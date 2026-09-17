"""
実験 EG: Conv-AE/VAE による MNIST m スイープ
目的: 全結合型に限定されないアーキテクチャへの汎化性を示す
Architecture: Conv-AE/VAE (Encoder: Conv+FC, Decoder: FC+ConvTranspose)
m: [4, 8, 12, 16, 20, 32, 64]
epochs: 100
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Conv-AE
# ──────────────────────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.conv(x).flatten(1)
        return self.fc(h)


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
        return self.deconv(h).flatten(1)  # (N, 784)


class ConvAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ──────────────────────────────────────────────────────────────────────────────
# Conv-VAE
# ──────────────────────────────────────────────────────────────────────────────
class ConvVAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_lv  = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


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


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
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


def train_conv_ae(m, X_train, X_test, epochs=100, batch_size=128):
    model = ConvAE(m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, z = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test).to(device)
        x_hat, Z = model(X_te)
        mse = F.mse_loss(x_hat, X_te).item()
        Z = Z.cpu().numpy()
    au    = active_units(Z, threshold=1e-2)
    id_2nn = twonn_estimate(Z)
    id_mle = mle_estimate(Z, k=10)
    return {'m': m, 'mse': float(mse), 'au': int(au),
            'twonn': float(id_2nn), 'mle': float(id_mle)}


def train_conv_vae(m, X_train, X_test, epochs=100, beta=4.0, batch_size=128):
    model = ConvVAE(m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
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
    au    = active_units(Z, threshold=1e-2)
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
    m_list = [4, 8, 12, 16, 20, 32, 64]
    epochs = 100
    beta   = 4.0

    X_train, X_test = load_mnist()
    print(f"Data: train={len(X_train)}, test={len(X_test)}")

    ae_results  = []
    vae_results = []

    print("\n=== Conv-AE ===")
    for m in m_list:
        print(f"  m={m} ...", end=" ", flush=True)
        r = train_conv_ae(m, X_train, X_test, epochs=epochs)
        ae_results.append(r)
        print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}")
        with open('results/tables/EG_conv.json', 'w') as f:
            json.dump({'ae': ae_results, 'vae': vae_results}, f, indent=2)

    print("\n=== Conv-VAE ===")
    for m in m_list:
        print(f"  m={m} ...", end=" ", flush=True)
        r = train_conv_vae(m, X_train, X_test, epochs=epochs, beta=beta)
        vae_results.append(r)
        print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}")
        with open('results/tables/EG_conv.json', 'w') as f:
            json.dump({'ae': ae_results, 'vae': vae_results}, f, indent=2)

    mknee_ae  = compute_mknee(ae_results,  tau=0.25)
    mknee_vae = compute_mknee(vae_results, tau=0.25)
    print(f"\nmknee: AE={mknee_ae}, VAE={mknee_vae}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: AE
    ax  = axes[0]
    axr = ax.twinx()
    ms  = [r['m']    for r in ae_results]
    mses = [r['mse'] for r in ae_results]
    ids  = [r['twonn'] for r in ae_results]
    ax.semilogy(ms, mses, 'b-o', ms=5, label='MSE (Conv-AE)')
    axr.plot(ms, ids, 'r--s', ms=5, label='TwoNN ID')
    axr.axhline(ids[-1], color='gray', ls=':', lw=1, label=f'$d_{{ID}}\\approx{ids[-1]:.1f}$')
    ax.set_xlabel('Bottleneck dim $m$'); ax.set_ylabel('MSE', color='b')
    axr.set_ylabel('TwoNN ID', color='r')
    ax.set_title('Conv-AE on MNIST', fontsize=11)
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor='b'); axr.tick_params(axis='y', labelcolor='r')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8); ax.grid(True, alpha=0.3)

    # Right: VAE
    ax2  = axes[1]
    ax2r = ax2.twinx()
    ms_v  = [r['m']    for r in vae_results]
    mses_v = [r['mse'] for r in vae_results]
    aus_v  = [r['au']  for r in vae_results]
    ids_v  = [r['twonn'] for r in vae_results]
    ax2.semilogy(ms_v, mses_v, 'b-o', ms=5, label='MSE (Conv-VAE)')
    ax2.plot(ms_v, aus_v, 'g-^', ms=5, label='AU (Conv-VAE)')
    ax2r.plot(ms_v, ids_v, 'r--s', ms=5, label='TwoNN ID')
    if mknee_vae:
        ax2.axvline(mknee_vae, color='purple', ls='--', lw=1.5,
                    label=f'$m_{{\\mathrm{{knee}}}}={mknee_vae}$')
    ax2.set_xlabel('Bottleneck dim $m$'); ax2.set_ylabel('MSE / AU', color='b')
    ax2r.set_ylabel('TwoNN ID', color='r')
    ax2.set_title(f'Conv-VAE ($\\beta={beta}$) on MNIST', fontsize=11)
    ax2.set_xscale('log')
    ax2.tick_params(axis='y', labelcolor='b'); ax2r.tick_params(axis='y', labelcolor='r')
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EG_conv_mnist.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EG_conv_mnist.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EG_conv_mnist.{png,pdf}")

    # Summary
    print("\n=== Summary ===")
    print("Conv-AE:")
    for r in ae_results:
        print(f"  m={r['m']:3d}: MSE={r['mse']:.5f}, AU={r['au']}, TwoNN={r['twonn']:.2f}")
    print("Conv-VAE:")
    for r in vae_results:
        print(f"  m={r['m']:3d}: MSE={r['mse']:.5f}, AU={r['au']}, TwoNN={r['twonn']:.2f}")


if __name__ == '__main__':
    main()
