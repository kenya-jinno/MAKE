"""
実験 EH: Conv-VAE 300エポック多シード検証
目的: Conv-VAEで十分収束した場合のAU Dynamics・mkneeを検証し
     FC-VAE（E4）との整合性を確認する
Architecture: Conv-VAE (EGと同一アーキテクチャ)
m: [4, 8, 12, 16, 20, 32, 64]
epochs: 300  beta: 4.0  seeds: [42, 123, 777]
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


# ── Conv-VAE (EGと同一アーキテクチャ) ────────────────────────────────────────
class ConvVAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_lv = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14->28
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, 64, 7, 7)
        return self.deconv(h).flatten(1)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvVAEEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def conv_vae_loss(x_hat, x, mu, logvar, beta=4.0):
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


# ── Data ──────────────────────────────────────────────────────────────────────
def load_mnist(n_train=20000, n_test=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_train = train_ds.data[:n_train].float().view(-1, 784) / 255.0
    X_test  = test_ds.data[:n_test].float().view(-1, 784) / 255.0
    return X_train.numpy(), X_test.numpy()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_conv_vae(m, X_train, X_test, epochs=300, beta=4.0, batch_size=128):
    model = ConvVAE(m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = conv_vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = model(X_te)
        mse = F.mse_loss(x_hat, X_te).item()
        Z = mu_te.cpu().numpy()
    au     = active_units(Z, threshold=1e-2)
    id_2nn = twonn_estimate(Z)
    id_mle = mle_estimate(Z, k=10)
    return {'m': m, 'mse': float(mse), 'au': int(au),
            'twonn': float(id_2nn), 'mle': float(id_mle)}


def compute_mknee(results, tau=0.25):
    ms  = [r['m']  for r in results]
    aus = [r['au'] for r in results]
    for i in range(1, len(ms)):
        dm  = ms[i] - ms[i-1]
        dau = aus[i] - aus[i-1]
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def main():
    seeds  = [42, 123, 777]
    m_list = [4, 8, 12, 16, 20, 32, 64]
    epochs = 300
    beta   = 4.0

    X_train, X_test = load_mnist()
    print(f"Data: train={len(X_train)}, test={len(X_test)}")

    all_results = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Conv-VAE Seed {seed} (300 epochs) ===")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_conv_vae(m, X_train, X_test, epochs=epochs, beta=beta)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}")
        mknee = compute_mknee(res, tau=0.25)
        print(f"  => mknee(tau=0.25) = {mknee}")
        all_results[str(seed)] = res
        with open('results/tables/EH_conv_vae_300ep.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial results")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Conv-VAE 300ep Multi-seed Summary ===")
    print(f"{'m':>4} | {'AU (mean±std)':>18} | {'TwoNN (mean±std)':>18}")
    for i, m in enumerate(m_list):
        aus   = [all_results[str(s)][i]['au']    for s in seeds]
        twons = [all_results[str(s)][i]['twonn'] for s in seeds]
        print(f"{m:4d} | {np.mean(aus):6.1f}±{np.std(aus):4.1f}        | "
              f"{np.mean(twons):6.2f}±{np.std(twons):4.2f}")

    mknees = [compute_mknee(all_results[str(s)], tau=0.25) for s in seeds]
    print(f"\nmknee per seed: {mknees}")
    print(f"mknee mean={np.mean(mknees):.1f}, std={np.std(mknees):.1f}")

    # ── Figure: compare Conv-VAE (300ep) vs FC-VAE (300ep) ───────────────────
    fc_vae_ref = {
        4: {'au': 4.0, 'twonn': 3.98},
        8: {'au': 8.0, 'twonn': 6.67},
        12: {'au': 10.0, 'twonn': 7.41},
        16: {'au': 11.7, 'twonn': 7.95},
        20: {'au': 13.3, 'twonn': 8.37},
        32: {'au': 15.7, 'twonn': 9.07},
        64: {'au': 22.3, 'twonn': 10.03},
    }

    all_aus   = np.array([[all_results[str(s)][i]['au']    for i in range(len(m_list))] for s in seeds])
    all_twons = np.array([[all_results[str(s)][i]['twonn'] for i in range(len(m_list))] for s in seeds])
    mu_au = all_aus.mean(0);   sd_au = all_aus.std(0)
    mu_id = all_twons.mean(0); sd_id = all_twons.std(0)
    fc_aus   = [fc_vae_ref[m]['au']    for m in m_list]
    fc_twons = [fc_vae_ref[m]['twonn'] for m in m_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Left: AU per seed
    ax = axes[0]
    for seed, col in zip(seeds, colors):
        res = all_results[str(seed)]
        aus_s = [r['au'] for r in res]
        mk = compute_mknee(res, tau=0.25)
        ax.plot(m_list, aus_s, 'o-', color=col, ms=5,
                label=f'Conv-VAE seed={seed} ($m_{{\\mathrm{{knee}}}}={mk}$)')
        ax.axvline(mk, color=col, ls=':', lw=1)
    ax.plot(m_list, fc_aus, 'k--s', ms=5, lw=1.5, alpha=0.7, label='FC-VAE mean (300ep, ref)')
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m (AE upper bound)')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('Conv-VAE AU vs $m$ (3 seeds, 300 ep)')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right: mean±std + FC-VAE comparison
    ax2  = axes[1]
    ax2r = ax2.twinx()
    ax2.fill_between(m_list, mu_au - sd_au, mu_au + sd_au, alpha=0.2, color='tab:blue')
    ax2.plot(m_list, mu_au, 'b-o', ms=5, label='Conv-VAE AU mean±std')
    ax2.plot(m_list, fc_aus, 'k--s', ms=5, lw=1.5, alpha=0.7, label='FC-VAE AU mean (ref)')
    ax2r.fill_between(m_list, mu_id - sd_id, mu_id + sd_id, alpha=0.2, color='tab:red')
    ax2r.plot(m_list, mu_id, 'r--^', ms=5, label='Conv-VAE TwoNN ID')
    ax2r.plot(m_list, fc_twons, 'm:^', ms=4, alpha=0.7, label='FC-VAE TwoNN ID (ref)')
    mk_mean = int(round(np.mean(mknees)))
    ax2.axvline(mk_mean, color='purple', ls='--', lw=1.5,
                label=f'Conv-VAE $m_{{\\mathrm{{knee}}}}$ mean={mk_mean}')
    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('AU', color='b')
    ax2r.set_ylabel('TwoNN ID', color='r')
    ax2.set_title('Conv-VAE vs FC-VAE (300 ep, MNIST)')
    ax2.set_xscale('log')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2r.tick_params(axis='y', labelcolor='r')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EH_conv_vae_300ep.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EH_conv_vae_300ep.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EH_conv_vae_300ep.{png,pdf}")


if __name__ == '__main__':
    main()
