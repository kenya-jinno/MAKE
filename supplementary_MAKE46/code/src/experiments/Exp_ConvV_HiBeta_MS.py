"""
実験 Exp-ConvV-HiBeta-MS: Exp-ConvV-HiBeta の多シード化（査読対応：単一シード → 3シード）
目的: β_max ∈ {10, 20} の高 β Conv-VAE による AU 遮断の回復が
     シードに対して頑健であることを検証する．
     seed 42 は既存の Exp_ConvV_HiBeta.json を再利用し，seed 123, 777 を追加学習する．

Architecture: Exp_ConvV_HighBeta.py と完全同一（ConvVAE + KL サイクリックアニーリング）
β_max: [10, 20]  seeds: [42(既存), 123, 777]  m: [16, 32, 64, 96, 128, 192, 256]  epochs: 150
出力: results/tables/Exp_ConvV_HiBeta_MS.json，results/figures/Exp_ConvV_HiBeta.{png,pdf}
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

from src.metrics.intrinsic_dim import twonn_estimate
from src.metrics.structure import active_units

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


class ConvVAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
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
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
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


def vae_loss(x_hat, x, mu, logvar, beta=4.0):
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


def cyclic_beta(epoch, total_epochs, n_cycles=3, beta_max=4.0, ratio=0.5):
    cycle_len = total_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    return beta_max * min(1.0, t / ratio)


def train_conv_vae(model, loader, epochs, device, beta_max=4.0, n_cycles=3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        beta = cyclic_beta(ep, epochs, n_cycles=n_cycles, beta_max=beta_max)
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            opt.step()
        if (ep + 1) % 50 == 0:
            print(f"      ep {ep+1}/{epochs}, beta={beta:.2f}", flush=True)


def encode_mu(model, X_t, device):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(X_t.to(device))
    return mu.cpu().numpy()


def load_mnist_tensors():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_train = train_ds.data[:20000].float().view(-1, 784) / 255.0
    X_test  = test_ds.data[:3000].float().view(-1, 784) / 255.0
    return X_train, X_test


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_hibeta_ms():
    print("=== Exp-ConvV-HiBeta-MS: 高 beta Conv-VAE の多シード検証 ===")

    X_train, X_test = load_mnist_tensors()

    m_list = [16, 32, 64, 96, 128, 192, 256]
    beta_list = [10, 20]
    new_seeds = [123, 777]

    # seed 42 は既存結果を再利用
    all_results = {}
    with open('results/tables/Exp_ConvV_HiBeta.json') as f:
        seed42 = json.load(f)
    all_results['42'] = seed42

    out_path = 'results/tables/Exp_ConvV_HiBeta_MS.json'
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results.update(json.load(f))

    for seed in new_seeds:
        if str(seed) in all_results:
            print(f"  seed {seed}: already done, skipping")
            continue
        set_seed(seed)
        loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
        results = {}
        for beta_max in beta_list:
            print(f"\n  seed={seed}, beta_max={beta_max}:")
            results[str(beta_max)] = []
            for m in m_list:
                print(f"    Training ConvVAE m={m} beta_max={beta_max} seed={seed} ...", flush=True)
                torch.manual_seed(seed)
                model = ConvVAE(latent_dim=m).to(device)
                train_conv_vae(model, loader, epochs=150, device=device,
                               beta_max=beta_max, n_cycles=3)

                Z = encode_mu(model, X_test, device)
                au = active_units(Z, threshold=0.01)
                tw = twonn_estimate(Z)

                model.eval()
                with torch.no_grad():
                    X_te = X_test.to(device)
                    x_hat, _, _ = model(X_te)
                    mse = float(F.mse_loss(x_hat, X_te).item())

                print(f"    m={m}: AU={au}, TwoNN={tw:.3f}, MSE={mse:.5f}", flush=True)
                results[str(beta_max)].append(
                    {'m': m, 'au': int(au), 'twonn': float(tw), 'mse': mse})
        all_results[str(seed)] = results
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial results -> {out_path}")

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Summary (paper table) ----
    seeds = ['42', '123', '777']
    print("\n=== Exp-ConvV-HiBeta-MS Summary (mean±std, 3 seeds) ===")
    summary = {}
    for beta_max in beta_list:
        print(f"  beta_max={beta_max}:")
        summary[str(beta_max)] = []
        for i, m in enumerate(m_list):
            aus = [all_results[s][str(beta_max)][i]['au'] for s in seeds]
            mu, sd = np.mean(aus), np.std(aus)
            summary[str(beta_max)].append({'m': m, 'au_mean': float(mu), 'au_std': float(sd)})
            print(f"    m={m:4d}: AU={mu:6.1f}±{sd:4.1f} ({mu/m*100:.0f}%)  seeds={aus}")

    # ---- Figure (B/W-friendly: distinct linestyles, larger fonts) ----
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    styles = {'4':  dict(color='tab:blue',   ls='-',  marker='o'),
              '10': dict(color='tab:orange', ls='--', marker='s'),
              '20': dict(color='tab:red',    ls='-.', marker='^')}

    # beta=4 reference: 3-seed Exp-Conv-M256 (EN)
    try:
        with open('results/tables/EN_conv_vae_extended_m.json') as f:
            en = json.load(f)
        en_m = [r['m'] for r in en['42']['results'] if 16 <= r['m'] <= 256]
        en_au = np.array([[r['au'] for r in en[s]['results'] if 16 <= r['m'] <= 256]
                          for s in ['42', '123', '777']])
        st = styles['4']
        axes[0].errorbar(en_m, en_au.mean(0), yerr=en_au.std(0), lw=2, ms=6,
                         capsize=3, label=r'$\beta_{\max}=4$ (Exp-Conv-M256)', **st)
        axes[1].errorbar(en_m, (en_au / np.array(en_m)).mean(0),
                         yerr=(en_au / np.array(en_m)).std(0), lw=2, ms=6,
                         capsize=3, label=r'$\beta_{\max}=4$', **st)
    except Exception as e:
        print("beta=4 reference not plotted:", e)

    for beta_max in beta_list:
        au_mat = np.array([[all_results[s][str(beta_max)][i]['au']
                            for i in range(len(m_list))] for s in seeds])
        st = styles[str(beta_max)]
        axes[0].errorbar(m_list, au_mat.mean(0), yerr=au_mat.std(0), lw=2, ms=6,
                         capsize=3, label=rf'$\beta_{{\max}}={beta_max}$', **st)
        ratio = au_mat / np.array(m_list)
        axes[1].errorbar(m_list, ratio.mean(0), yerr=ratio.std(0), lw=2, ms=6,
                         capsize=3, label=rf'$\beta_{{\max}}={beta_max}$', **st)

    axes[0].plot([0, 260], [0, 260], 'k:', lw=1.5, alpha=0.6, label='AU $=m$')
    axes[0].set_xlabel('Bottleneck dimension $m$')
    axes[0].set_ylabel('Active Units (AU)')
    axes[0].set_title(r'MNIST Conv-VAE: AU vs. $m$ (mean$\pm$std, 3 seeds)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color='k', ls=':', lw=1.5, alpha=0.6, label='AU$/m=1$')
    axes[1].set_xlabel('Bottleneck dimension $m$')
    axes[1].set_ylabel('AU $/\\, m$')
    axes[1].set_title('Active ratio AU$/m$ vs. $m$')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('results/figures/Exp_ConvV_HiBeta.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/Exp_ConvV_HiBeta.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/Exp_ConvV_HiBeta.{png,pdf}")

    with open('results/tables/Exp_ConvV_HiBeta_MS_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved: results/tables/Exp_ConvV_HiBeta_MS_summary.json")

    return all_results


if __name__ == '__main__':
    run_hibeta_ms()
