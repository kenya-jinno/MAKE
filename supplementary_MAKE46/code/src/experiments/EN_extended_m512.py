"""
実験 EN_ext: MNIST Conv-VAE — m を 256 超（384, 512）まで拡張
目的: EN 実験（m≤256）で mknee=256（グリッド端）だった Conv-VAE の AU 曲線を
     m=384, 512 まで伸ばし，AU 飽和点が実際に存在するかを確認する．

EN の既存結果（results/tables/EN_conv_vae_extended_m.json）を読み込み，
新たに m={384, 512} のみ 3シードで訓練して結合する．

Architecture: EN と完全に同一（ConvVAE + KL サイクリックアニーリング）
epochs: 150  seeds: [42, 123, 777]
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


# ─── EN と同一の ConvVAE 構造 ───────────────────────────────────────────────
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
    """EN オリジナルと完全同一の損失関数（MSE + β*KL）"""
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


def cyclic_beta(epoch, total_epochs, n_cycles=3, beta_max=4.0, ratio=0.5):
    """EN オリジナルと完全同一のサイクリック β スケジュール"""
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
            print(f"      ep {ep+1}/{epochs}, β={beta:.2f}")


def encode_mu(model, X_t, device):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(X_t.to(device))
    return mu.cpu().numpy()


def load_mnist_tensors():
    """EN オリジナルと同じ: train=20000, test=3000"""
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


def run_en_ext():
    print("=== EN_ext: Conv-VAE m=384, 512 拡張 ===")

    # 既存 EN 結果を読み込む
    with open('results/tables/EN_conv_vae_extended_m.json') as f:
        en_existing = json.load(f)
    print(f"  Loaded existing EN results for m ≤ 256 (seeds: {list(en_existing.keys())})")

    X_train, X_test = load_mnist_tensors()
    loader = DataLoader(
        TensorDataset(X_train), batch_size=128, shuffle=True
    )

    seeds = [42, 123, 777]
    new_m_list = [384, 512]

    # 新規実験
    new_results = {str(s): [] for s in seeds}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"\n  Seed {seed}:")
        for m in new_m_list:
            print(f"    Training ConvVAE m={m} ...")
            model = ConvVAE(latent_dim=m).to(device)
            train_conv_vae(model, loader, epochs=150, device=device,
                           beta_max=4.0, n_cycles=3)

            Z = encode_mu(model, X_test, device)
            au  = active_units(Z, threshold=0.01)
            tw  = twonn_estimate(Z)
            mle = mle_estimate(Z, k=10)

            # MSE（EN オリジナルと同様の計算）
            model.eval()
            with torch.no_grad():
                X_te = X_test.to(device)
                x_hat, _, _ = model(X_te)
                mse = float(F.mse_loss(x_hat, X_te).item())

            print(f"    m={m}: AU={au}, TwoNN={tw:.3f}, MLE={mle:.3f}, MSE={mse:.5f}")
            new_results[str(seed)].append({'m': m, 'mse': mse, 'au': au,
                                           'twonn': tw, 'mle': mle})

    # 既存 + 新規を結合してJSON保存
    combined = {}
    for seed in seeds:
        old = en_existing[str(seed)]['results']
        new = new_results[str(seed)]
        all_results = sorted(old + new, key=lambda r: r['m'])

        # mknee 再計算（τ=0.25）
        m_vals = [r['m'] for r in all_results]
        au_vals = [r['au'] for r in all_results]
        au_arr = np.array(au_vals, dtype=float)
        delta_au = np.diff(au_arr)
        delta_m  = np.diff(m_vals)
        rates = delta_au / delta_m
        max_rate = rates.max() if len(rates) > 0 else 1.0
        mknee = m_vals[-1]
        for i, rate in enumerate(rates):
            if rate < 0.25 * max_rate:
                mknee = m_vals[i + 1]
                break

        combined[str(seed)] = {'results': all_results, 'mknee': mknee}
        print(f"  Seed {seed}: mknee={mknee}")

    with open('results/tables/EN_ext_m512.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print("Saved: results/tables/EN_ext_m512.json")

    return combined


def make_en_ext_figure(combined):
    seeds = [42, 123, 777]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # 左: AU vs m（全データ, 3シード）
    ax = axes[0]
    for seed, color in zip(seeds, colors):
        data = combined[str(seed)]['results']
        m_vals = [r['m'] for r in data]
        au_vals = [r['au'] for r in data]
        mknee = combined[str(seed)]['mknee']
        ax.plot(m_vals, au_vals, 'o-', color=color, lw=1.5, ms=4,
                label=f'seed={seed} ($m_\\mathrm{{knee}}$={mknee})')
    ax.plot([0, 520], [0, 520], 'k--', lw=1, alpha=0.4, label='AU$=m$（参照）')
    ax.axvline(256, color='gray', ls=':', lw=1, alpha=0.7, label='既存 EN 上限（$m=256$）')
    ax.set_xlabel('ボトルネック次元 $m$')
    ax.set_ylabel('Active Units（AU）')
    ax.set_title('EN_ext: MNIST Conv-VAE AU vs $m$（$m$≤512）\n'
                 '（3シード，150ep，KL サイクリック，$\\beta_\\max=4$）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 530)

    # 右: AU/m 比率（飽和の度合い）
    ax2 = axes[1]
    for seed, color in zip(seeds, colors):
        data = combined[str(seed)]['results']
        m_vals = [r['m'] for r in data]
        ratio  = [r['au'] / r['m'] for r in data]
        ax2.plot(m_vals, ratio, 'o-', color=color, lw=1.5, ms=4, label=f'seed={seed}')
    ax2.axhline(1.0, color='k', ls='--', lw=1, alpha=0.4, label='AU/m=1（完全活性）')
    ax2.axhline(0.25, color='red', ls=':', lw=1.5, label='25% 閾値（参照）')
    ax2.axvline(256, color='gray', ls=':', lw=1, alpha=0.7, label='既存 EN 上限')
    ax2.set_xlabel('ボトルネック次元 $m$')
    ax2.set_ylabel('AU / $m$（活性率）')
    ax2.set_title('EN_ext: 活性率 AU/$m$ vs $m$\n（完全飽和なら AU/$m\\to0$）')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 530)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('results/figures/EN_ext_m512.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EN_ext_m512.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EN_ext_m512.{png,pdf}")


if __name__ == '__main__':
    combined = run_en_ext()
    make_en_ext_figure(combined)

    # サマリ表示
    print("\n=== EN_ext Summary ===")
    seeds = [42, 123, 777]
    for seed in seeds:
        data = combined[str(seed)]['results']
        mknee = combined[str(seed)]['mknee']
        print(f"  Seed {seed} (mknee={mknee}):")
        for r in data[-5:]:
            print(f"    m={r['m']:4d}: AU={r['au']:4d}, TwoNN={r['twonn']:.2f}")
