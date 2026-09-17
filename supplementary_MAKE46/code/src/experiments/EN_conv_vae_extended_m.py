"""
実験 EN: MNIST Conv-VAE — 拡張 m グリッドでの AU 飽和探索
目的: EK（m≤64）では AU 飽和プラトーが未到達だったが，
     m を 256（または 512）まで拡張すると Conv-VAE でも AU 飽和が現れるかを検証する．
     これにより「Conv-VAE でも十分大きな m があれば AU 飽和が起きる」ことを実証し，
     Theorem 1（水充填原理）が Conv-VAE でも定性的に成立することを確認する．
Architecture: Conv-VAE（EK と同一）+ KL サイクリックアニーリング
m: [4, 8, 16, 32, 64, 96, 128, 192, 256]
epochs: 150  seeds: [42, 123, 777]  (3シードで統計的信頼性を確保)
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


class ConvVAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
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
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
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


def vae_loss(x_hat, x, mu, logvar, beta=4.0):
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


def cyclic_beta(epoch, total_epochs, n_cycles=3, beta_max=4.0, ratio=0.5):
    cycle_len = total_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    return beta_max * min(1.0, t / ratio)


def load_mnist(n_train=20000, n_test=3000):
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True,
        transform=transforms.ToTensor())
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True,
        transform=transforms.ToTensor())
    X_train = train_ds.data[:n_train].float().view(-1, 784) / 255.0
    X_test  = test_ds.data[:n_test].float().view(-1, 784) / 255.0
    return X_train, X_test


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def train(m, X_train, X_test, epochs=150, beta_max=4.0, n_cycles=3, batch_size=128):
    model = ConvVAE(m).to(device)
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
        X_te = X_test.to(device)
        x_hat, mu_te, _ = model(X_te)
        mse = F.mse_loss(x_hat, X_te).item()
        Z   = mu_te.cpu().numpy()
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
        r   = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def main():
    seeds  = [42, 123, 777]
    m_list = [4, 8, 16, 32, 64, 96, 128, 192, 256]
    epochs = 150
    beta_max = 4.0

    print(f"=== EN: Conv-VAE Extended m Grid (m up to 256) ===")
    print(f"Device: {device}, epochs={epochs}, beta_max={beta_max}, seeds={seeds}")

    print("Loading MNIST ...")
    X_train, X_test = load_mnist()

    all_results = {}
    for seed in seeds:
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train(m, X_train, X_test, epochs=epochs, beta_max=beta_max)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}")
        mknee = compute_mknee(res, tau=0.25)
        print(f"  mknee(tau=0.25) = {mknee}")
        all_results[str(seed)] = {'results': res, 'mknee': mknee}

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== EN Summary ===")
    print(f"{'m':>5} | " + " | ".join(f"AU(s={s})" for s in seeds) + " | mean AU | TwoNN(s42)")
    for i, m in enumerate(m_list):
        aus = [all_results[str(s)]['results'][i]['au'] for s in seeds]
        twonn42 = all_results['42']['results'][i]['twonn']
        print(f"{m:5d} | " + " | ".join(f"{a:8d}" for a in aus) + f" | {np.mean(aus):7.1f} | {twonn42:9.2f}")

    mknees = [all_results[str(s)]['mknee'] for s in seeds]
    print(f"\nmknee per seed: {mknees}, mean={np.mean(mknees):.1f}±{np.std(mknees):.1f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open('results/tables/EN_conv_vae_extended_m.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Figure ───────────────────────────────────────────────────────────────
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: AU vs m
    ax = axes[0]
    for seed, col in zip(seeds, colors):
        res = all_results[str(seed)]['results']
        aus_s = [r['au'] for r in res]
        mk = all_results[str(seed)]['mknee']
        ax.plot(m_list, aus_s, 'o-', color=col, ms=5, lw=1.5,
                label=f'seed={seed} ($m_{{\\mathrm{{knee}}}}={mk}$)')
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m (diagonal)')
    ax.axvline(10, color='red', ls='-.', lw=1.5, alpha=0.7,
               label='$\\hat{d}_{\\mathrm{ID}} \\approx 10$ (FC-VAE)')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('EN: Conv-VAE extended $m$ grid (KL annealing, MNIST)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 270)

    # Right: TwoNN ID vs m
    ax2 = axes[1]
    for seed, col in zip(seeds, colors):
        res = all_results[str(seed)]['results']
        t_s = [r['twonn'] for r in res]
        ax2.plot(m_list, t_s, 'o-', color=col, ms=5, lw=1.5, label=f'TwoNN (seed={seed})')
    ax2.axhline(10, color='red', ls='-.', lw=1.5, alpha=0.7,
                label='$\\hat{d}_{\\mathrm{ID}} \\approx 10$ (FC-VAE)')
    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('TwoNN ID estimate')
    ax2.set_title('EN: Conv-VAE — TwoNN ID vs $m$')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 270)

    plt.tight_layout()
    plt.savefig('results/figures/EN_conv_vae_extended_m.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EN_conv_vae_extended_m.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EN_conv_vae_extended_m.{png,pdf}")
    print("Saved: results/tables/EN_conv_vae_extended_m.json")


if __name__ == '__main__':
    main()
