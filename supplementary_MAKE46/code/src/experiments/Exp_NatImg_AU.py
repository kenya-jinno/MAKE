"""
実験 Exp-NatImg-AU: 自然画像 (CIFAR-10) における AU 飽和条件の探索
目的: β スケジューリング・アーキテクチャ深度を多様に変えて，
     AU 曲線が飽和するかどうかを確認し，飽和検出可能条件を整理する．

実験条件:
- Dataset: CIFAR-10 (n_train=20000, n_test=3000)
- Architecture A (Standard): EI と同一 (ch=[32,64,128])
- Architecture B (Deep):     ch=[64,128,256] でより深い表現力
- beta_max ∈ {10, 20, 40}
- m ∈ {32, 64, 96, 128, 192, 256, 384}
- n_cycles=3, epochs=200, seed=42
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import active_units

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


# ── Architecture A: Standard (same as EI) ─────────────────────────────────────
class ConvEncoderA(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 32→16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 16→8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8→4
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_lv = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoderA(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(F.relu(self.fc(z)).view(-1, 128, 4, 4))


# ── Architecture B: Deep (more channels) ──────────────────────────────────────
class ConvEncoderB(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),    # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8→4
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_lv = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoderB(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(F.relu(self.fc(z)).view(-1, 256, 4, 4))


class ConvVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta):
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


def cyclic_beta(epoch, total_epochs, n_cycles=3, beta_max=4.0, ratio=0.5):
    cycle_len = total_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    return beta_max * min(t / ratio, 1.0)


def load_cifar10(n_train=20000, n_test=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_train = torch.stack([train_ds[i][0] for i in range(n_train)])
    X_test  = torch.stack([test_ds[i][0]  for i in range(n_test)])
    return X_train, X_test


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def train_and_eval(model, X_train, X_test, epochs=200, beta_max=10.0,
                   n_cycles=3, batch_size=128, seed=42):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        beta = cyclic_beta(ep, epochs, n_cycles=n_cycles, beta_max=beta_max)
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        X_te = X_test.to(device)
        x_hat, mu_te, _ = model(X_te)
        mse = F.mse_loss(x_hat, X_te).item()
        Z = mu_te.cpu().numpy()
    au = int(active_units(Z, threshold=1e-2))
    id_2nn = float(twonn_estimate(Z))
    return {'mse': float(mse), 'au': au, 'twonn': id_2nn}


def compute_mknee(m_list, au_list, tau=0.25):
    for i in range(1, len(m_list)):
        dm  = m_list[i] - m_list[i-1]
        dau = au_list[i] - au_list[i-1]
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return m_list[i]
    return m_list[-1]


def main():
    print("=" * 60)
    print("Exp-NatImg-AU: CIFAR-10 AU Saturation Condition Exploration")
    print("=" * 60)

    X_train, X_test = load_cifar10(n_train=20000, n_test=3000)
    print(f"Data: X_train={X_train.shape}, X_test={X_test.shape}")

    # Input-space TwoNN (reference)
    X_flat = X_test[:1000].reshape(1000, -1).numpy()
    raw_id = twonn_estimate(X_flat)
    print(f"Raw CIFAR-10 TwoNN ID (pixel space, n=1000): {raw_id:.2f}")

    m_list = [32, 64, 96, 128, 192, 256, 384]

    configs = [
        ('A', 'Standard', [10, 20, 40]),
        ('B', 'Deep',     [20, 40]),
    ]

    all_results = {'raw_id': raw_id, 'configs': {}}
    seed = 42

    for arch_key, arch_name, betas in configs:
        all_results['configs'][arch_key] = {'name': arch_name, 'betas': {}}
        for beta_max in betas:
            print(f"\n--- Arch={arch_name}, beta_max={beta_max} ---")
            res_list = []
            for m in m_list:
                print(f"  m={m}...", end='', flush=True)
                if arch_key == 'A':
                    enc = ConvEncoderA(m).to(device)
                    dec = ConvDecoderA(m).to(device)
                else:
                    enc = ConvEncoderB(m).to(device)
                    dec = ConvDecoderB(m).to(device)
                model = ConvVAE(enc, dec).to(device)
                res = train_and_eval(
                    model, X_train, X_test,
                    epochs=200, beta_max=beta_max,
                    n_cycles=3, seed=seed
                )
                res['m'] = m
                res_list.append(res)
                print(f" AU={res['au']}/{m}, TwoNN={res['twonn']:.2f}", flush=True)

            au_list = [r['au'] for r in res_list]
            mknee = compute_mknee(m_list, au_list)
            print(f"  => mknee={mknee}, AU at m=384: {au_list[-1]}/{m_list[-1]}")

            all_results['configs'][arch_key]['betas'][str(beta_max)] = {
                'results': res_list,
                'mknee': mknee
            }

    # Save
    with open('results/tables/Exp_NatImg_AU.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/tables/Exp_NatImg_AU.json")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'10': 'tab:blue', '20': 'tab:orange', '40': 'tab:red'}

    for ax_idx, (arch_key, arch_cfg) in enumerate(all_results['configs'].items()):
        ax = axes[ax_idx]
        arch_name = arch_cfg['name']
        for beta_str, bdata in arch_cfg['betas'].items():
            res_list = bdata['results']
            ms  = [r['m']  for r in res_list]
            aus = [r['au'] for r in res_list]
            mk  = bdata['mknee']
            col = colors.get(beta_str, 'gray')
            ax.plot(ms, aus, 'o-', color=col, label=f'$\\beta_{{\\max}}={beta_str}$')
            ax.plot(ms, aus, 'o', color=col)
            if mk < m_list[-1]:
                ax.axvline(mk, color=col, linestyle='--', alpha=0.5)
        ax.plot(m_list, m_list, 'k:', alpha=0.4, label='AU=m (no saturation)')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Active Units (AU)')
        ax.set_title(f'CIFAR-10 AU Dynamics\nArch-{arch_key} ({arch_name})')
        ax.legend(fontsize=9)
        ax.set_xticks(m_list)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/Exp_NatImg_AU.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('results/figures/Exp_NatImg_AU.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved results/figures/Exp_NatImg_AU.{pdf,png}")

    # Summary
    print("\n=== Summary ===")
    print(f"CIFAR-10 raw TwoNN ID: {raw_id:.2f}")
    for arch_key, arch_cfg in all_results['configs'].items():
        for beta_str, bdata in arch_cfg['betas'].items():
            mk = bdata['mknee']
            last = bdata['results'][-1]
            print(f"Arch-{arch_key}, β={beta_str}: mknee={mk}, "
                  f"AU@m=384={last['au']}/{last['m']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
