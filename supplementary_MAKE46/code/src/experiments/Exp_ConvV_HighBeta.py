"""
実験 Exp-ConvV-HiBeta: MNIST Conv-VAE — 高 β によるAU飽和の人為的誘発
目的: β_max ∈ {10, 20} の強い KL 正則化により AU 飽和プラトーを観測できるか確認する．
     § の理論的考察（τ²_eff が小さいため d*(β)が高次元側に押しやられる）の逆証明：
     β を十分大きくすれば conv-VAE でも明確な AU 飽和が生じることを示す．

Architecture: EN と完全同一（ConvVAE + KL サイクリックアニーリング）
β_max: [10, 20]  seeds: [42]  m: [16, 32, 64, 96, 128, 192, 256]  epochs: 150
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
            print(f"      ep {ep+1}/{epochs}, beta={beta:.2f}")


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


def run_hibeta():
    print("=== Exp-ConvV-HiBeta: MNIST Conv-VAE 高 beta によるAU飽和 ===")

    X_train, X_test = load_mnist_tensors()
    loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)

    seed = 42
    m_list = [16, 32, 64, 96, 128, 192, 256]
    beta_list = [10, 20]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results = {}

    for beta_max in beta_list:
        print(f"\n  beta_max={beta_max}:")
        results[str(beta_max)] = []
        for m in m_list:
            print(f"    Training ConvVAE m={m} beta_max={beta_max} ...")
            torch.manual_seed(seed)
            model = ConvVAE(latent_dim=m).to(device)
            train_conv_vae(model, loader, epochs=150, device=device,
                           beta_max=beta_max, n_cycles=3)

            Z = encode_mu(model, X_test, device)
            au  = active_units(Z, threshold=0.01)
            tw  = twonn_estimate(Z)

            model.eval()
            with torch.no_grad():
                X_te = X_test.to(device)
                x_hat, _, _ = model(X_te)
                mse = float(F.mse_loss(x_hat, X_te).item())

            print(f"    m={m}: AU={au}, TwoNN={tw:.3f}, MSE={mse:.5f}")
            results[str(beta_max)].append({'m': m, 'au': au, 'twonn': tw, 'mse': mse})

    with open('results/tables/Exp_ConvV_HiBeta.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/tables/Exp_ConvV_HiBeta.json")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'4': 'tab:blue', '10': 'tab:orange', '20': 'tab:red'}
    labels = {'4': 'beta_max=4 (EN)', '10': 'beta_max=10', '20': 'beta_max=20'}

    # Load existing beta=4 results for comparison
    try:
        with open('results/tables/EN_ext_m512.json') as f:
            en_data = json.load(f)
        beta4_m = [r['m'] for r in en_data['42']['results'] if r['m'] <= 256]
        beta4_au = [r['au'] for r in en_data['42']['results'] if r['m'] <= 256]
        axes[0].plot(beta4_m, beta4_au, 'o-', color='tab:blue', lw=1.5, ms=4,
                     label='$\\beta_{\\max}=4$ (Exp-Conv-M256, seed=42)')
        axes[1].plot(beta4_m, [a/m for a, m in zip(beta4_au, beta4_m)],
                     'o-', color='tab:blue', lw=1.5, ms=4, label='$\\beta_{\\max}=4$')
    except Exception:
        pass

    for beta_max in beta_list:
        data = results[str(beta_max)]
        m_vals = [r['m'] for r in data]
        au_vals = [r['au'] for r in data]
        ratio = [r['au'] / r['m'] for r in data]
        c = colors[str(beta_max)]
        axes[0].plot(m_vals, au_vals, 'o-', color=c, lw=1.5, ms=4,
                     label=f'$\\beta_{{\\max}}={beta_max}$')
        axes[1].plot(m_vals, ratio, 'o-', color=c, lw=1.5, ms=4,
                     label=f'$\\beta_{{\\max}}={beta_max}$')

    axes[0].plot([0, 260], [0, 260], 'k--', lw=1, alpha=0.3, label='AU=m')
    axes[0].set_xlabel('Bottleneck dim $m$')
    axes[0].set_ylabel('Active Units (AU)')
    axes[0].set_title('MNIST Conv-VAE: AU vs $m$ for different $\\beta_{\\max}$\n(seed=42, 150ep, cyclic annealing)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color='k', ls='--', lw=1, alpha=0.3, label='AU/m=1')
    axes[1].set_xlabel('Bottleneck dim $m$')
    axes[1].set_ylabel('AU / m')
    axes[1].set_title('Active ratio AU/$m$ vs $m$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('results/figures/Exp_ConvV_HiBeta.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/Exp_ConvV_HiBeta.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/Exp_ConvV_HiBeta.{png,pdf}")

    # Summary
    print("\n=== Exp-ConvV-HiBeta Summary ===")
    for beta_max in beta_list:
        data = results[str(beta_max)]
        print(f"  beta_max={beta_max}:")
        for r in data:
            print(f"    m={r['m']:4d}: AU={r['au']:4d} ({r['au']/r['m']*100:.0f}%), TwoNN={r['twonn']:.2f}")

    return results


if __name__ == '__main__':
    run_hibeta()
