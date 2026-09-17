"""
実験 EI: CIFAR-10 Conv-VAE with KL Cyclic Annealing
目的: 自然画像（CIFAR-10）における多指標フレームワークの汎化性検証
     Pope et al. (2021) が報告する CIFAR-10 ID ≈ 26 との整合性確認
     KLアニーリングによる Conv-VAE の正常収束を実証
Architecture: Conv-VAE (32x32x3 入力)
m: [16, 32, 64, 128, 256]
epochs: 200  beta_max: 4.0  n_cycles: 4  seeds: [42, 123, 777]
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


# ── Conv-VAE for 32x32x3 (CIFAR-10 / SVHN) ───────────────────────────────────
class ConvEncoder32(nn.Module):
    def __init__(self, latent_dim, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # 16→8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 8→4
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_lv  = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoder32(nn.Module):
    def __init__(self, latent_dim, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 16→32
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, 128, 4, 4)
        return self.deconv(h)


class ConvVAE32(nn.Module):
    def __init__(self, latent_dim, in_channels=3):
        super().__init__()
        self.encoder = ConvEncoder32(latent_dim, in_channels)
        self.decoder = ConvDecoder32(latent_dim, in_channels)
        self.in_channels = in_channels

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


# ── Cyclic KL Annealing ────────────────────────────────────────────────────────
def cyclic_beta(epoch, total_epochs, n_cycles=4, beta_max=4.0, ratio=0.5):
    """Cyclic annealing schedule (Fu et al. 2019)"""
    cycle_len = total_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    if t < ratio:
        return beta_max * t / ratio
    else:
        return beta_max


# ── Data ───────────────────────────────────────────────────────────────────────
def load_cifar10(n_train=20000, n_test=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_train = torch.stack([train_ds[i][0] for i in range(n_train)])  # (N,3,32,32)
    X_test  = torch.stack([test_ds[i][0]  for i in range(n_test)])
    return X_train, X_test


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def train_conv_vae_annealing(m, X_train, X_test,
                              epochs=200, beta_max=4.0, n_cycles=4,
                              batch_size=128):
    model = ConvVAE32(m).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        X_te   = X_test.to(device)
        x_hat, mu_te, _ = model(X_te)
        mse    = F.mse_loss(x_hat, X_te).item()
        Z      = mu_te.cpu().numpy()
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


def estimate_raw_id(X_sample, n_sample=2000):
    """Estimate TwoNN ID on raw pixel space (flattened)."""
    X_flat = X_sample[:n_sample].reshape(n_sample, -1).numpy()
    return twonn_estimate(X_flat), mle_estimate(X_flat, k=10)


def main():
    seeds   = [42, 123, 777]
    m_list  = [16, 32, 64, 128, 256]
    epochs  = 150
    beta_max = 4.0
    n_cycles = 3

    print("Loading CIFAR-10 ...")
    X_train, X_test = load_cifar10(n_train=20000, n_test=3000)
    print(f"Data: train={len(X_train)}, test={len(X_test)}, shape={X_train.shape}")

    # ── Raw data ID estimate ────────────────────────────────────────────────
    print("\nEstimating raw CIFAR-10 ID (TwoNN / MLE) ...")
    raw_2nn, raw_mle = estimate_raw_id(X_train, n_sample=2000)
    print(f"  Raw CIFAR-10: TwoNN={raw_2nn:.2f}, MLE={raw_mle:.2f}")
    print(f"  (Pope et al. 2021 report ID ≈ 26 for CIFAR-10)")

    all_results = {'raw_id': {'twonn': float(raw_2nn), 'mle': float(raw_mle)}}

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Conv-VAE Cyclic Annealing  Seed {seed}  CIFAR-10 ===")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_conv_vae_annealing(m, X_train, X_test,
                                         epochs=epochs, beta_max=beta_max,
                                         n_cycles=n_cycles)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}")
        mknee = compute_mknee(res, tau=0.25)
        print(f"  => mknee(tau=0.25) = {mknee}")
        all_results[str(seed)] = res
        with open('results/tables/EI_cifar10_conv_vae.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== CIFAR-10 Conv-VAE Cyclic Annealing Summary ===")
    print(f"Raw data: TwoNN={raw_2nn:.2f}, MLE={raw_mle:.2f}")
    print(f"{'m':>4} | {'AU (mean±std)':>18} | {'TwoNN (mean±std)':>18}")
    for i, m in enumerate(m_list):
        aus   = [all_results[str(s)][i]['au']    for s in seeds]
        twons = [all_results[str(s)][i]['twonn'] for s in seeds]
        print(f"{m:4d} | {np.mean(aus):6.1f}±{np.std(aus):4.1f}        | "
              f"{np.mean(twons):6.2f}±{np.std(twons):4.2f}")

    mknees = [compute_mknee(all_results[str(s)], tau=0.25) for s in seeds]
    print(f"\nmknee per seed: {mknees}")
    print(f"mknee mean={np.mean(mknees):.1f}, std={np.std(mknees):.1f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    all_aus   = np.array([[all_results[str(s)][i]['au']    for i in range(len(m_list))] for s in seeds])
    all_twons = np.array([[all_results[str(s)][i]['twonn'] for i in range(len(m_list))] for s in seeds])
    mu_au = all_aus.mean(0);   sd_au = all_aus.std(0)
    mu_id = all_twons.mean(0); sd_id = all_twons.std(0)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: AU per seed
    ax = axes[0]
    for seed, col in zip(seeds, colors):
        res = all_results[str(seed)]
        aus_s = [r['au'] for r in res]
        mk = compute_mknee(res, tau=0.25)
        ax.plot(m_list, aus_s, 'o-', color=col, ms=5,
                label=f'seed={seed} ($m_{{\\mathrm{{knee}}}}={mk}$)')
        ax.axvline(mk, color=col, ls=':', lw=1, alpha=0.6)
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m')
    ax.axhline(raw_2nn, color='red', ls='-.', lw=1.5,
               label=f'Raw TwoNN ID={raw_2nn:.1f}')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('CIFAR-10 Conv-VAE AU (Cyclic Annealing, 3 seeds)')
    ax.set_xscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Right: mean±std AU and TwoNN ID
    ax2  = axes[1]
    ax2r = ax2.twinx()
    ax2.fill_between(m_list, mu_au - sd_au, mu_au + sd_au, alpha=0.2, color='tab:blue')
    ax2.plot(m_list, mu_au, 'b-o', ms=5, label='AU mean±std')
    ax2r.fill_between(m_list, mu_id - sd_id, mu_id + sd_id, alpha=0.15, color='tab:red')
    ax2r.plot(m_list, mu_id, 'r--^', ms=5, label='TwoNN ID mean±std')
    ax2r.axhline(raw_2nn, color='darkred', ls='-.', lw=1.5,
                 label=f'Raw TwoNN ID={raw_2nn:.1f}')
    mk_mean = int(round(np.mean(mknees)))
    ax2.axvline(mk_mean, color='purple', ls='--', lw=1.5,
                label=f'$m_{{\\mathrm{{knee}}}}$ mean={mk_mean}')
    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('AU', color='b')
    ax2r.set_ylabel('TwoNN ID', color='r')
    ax2.set_title('CIFAR-10: AU & TwoNN ID vs $m$ (Cyclic Annealing)')
    ax2.set_xscale('log')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2r.tick_params(axis='y', labelcolor='r')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EI_cifar10_conv_vae.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EI_cifar10_conv_vae.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EI_cifar10_conv_vae.{png,pdf}")


if __name__ == '__main__':
    main()
