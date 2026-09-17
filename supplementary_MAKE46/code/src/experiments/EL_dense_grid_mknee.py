"""
実験 EL: 密グリッド mknee 安定化実験
目的: グリッド密度・シード・エポック数の3要因を分離し，mknee の不安定性の原因を定量的に検証
     - 密グリッド（m ∈ {8,9,10,11,12,13,14,15,16,18,20,24,28,32,40,48,64}）で
       命題2のグリッド感度上界を実証的に検証する
     - 疎グリッド（既存 E4_300ep_multiseed）との比較により，
       グリッド密度要因を単離する
Architecture: FC-VAE（E4 と同一）
Dense grid m: [8,9,10,11,12,13,14,15,16,18,20,24,28,32,40,48,64]
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


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_lv = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim=784):
        super().__init__()
        self.encoder = Encoder(latent_dim, input_dim)
        self.decoder = Decoder(latent_dim, input_dim)

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


def train_vae(m, X_train, X_test, epochs=300, beta=4.0, batch_size=128):
    model = VAE(m).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
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
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def compute_mknee_second_diff(results):
    """離散二階差分の絶対値最大点をmkneeとする代替手法"""
    ms  = [r['m']  for r in results]
    aus = [r['au'] for r in results]
    if len(ms) < 3:
        return ms[-1]
    best_m, best_val = ms[1], 0
    for i in range(1, len(ms)-1):
        d2 = abs(aus[i+1] - 2*aus[i] + aus[i-1])
        if d2 > best_val:
            best_val = d2
            best_m = ms[i]
    return best_m


def main():
    seeds   = [42, 123, 777]
    # 密グリッド：m=8〜16 に細密なポイントを追加
    m_list_dense  = [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32, 40, 48, 64]
    # 疎グリッド（E4 との比較用）
    m_list_sparse = [4, 8, 12, 16, 20, 32, 64]
    epochs  = 300
    beta    = 4.0

    print("Loading MNIST ...")
    X_train, X_test = load_mnist(n_train=20000, n_test=3000)
    X_train_t = X_train.to(device) if device == 'cpu' else X_train
    X_test_t  = X_test.to(device)  if device == 'cpu' else X_test

    all_results = {}

    print("\n=== EL: Dense grid (m ∈ {8,...,64}) ===")
    for seed in seeds:
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        res = []
        for m in m_list_dense:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_vae(m, X_train, X_test, epochs=epochs, beta=beta)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}")
        mknee_tau = compute_mknee(res, tau=0.25)
        mknee_d2  = compute_mknee_second_diff(res)
        print(f"  Dense grid mknee(tau=0.25)={mknee_tau}, mknee(2nd-diff)={mknee_d2}")
        all_results[str(seed)] = {
            'dense': res,
            'mknee_tau': mknee_tau,
            'mknee_d2': mknee_d2,
        }

    with open('results/tables/EL_dense_grid_mknee.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== EL Summary ===")
    mk_tau = [all_results[str(s)]['mknee_tau'] for s in seeds]
    mk_d2  = [all_results[str(s)]['mknee_d2']  for s in seeds]
    print(f"Dense grid mknee (tau=0.25):  {mk_tau}  mean={np.mean(mk_tau):.1f}±{np.std(mk_tau):.1f}")
    print(f"Dense grid mknee (2nd-diff):  {mk_d2}   mean={np.mean(mk_d2):.1f}±{np.std(mk_d2):.1f}")

    print(f"\n3要因比較（グリッド密度×エポック；シードは共通 [42,123,777]）：")
    print(f"  疎グリッド 150ep（E4）:        mknee = [64, 32, 64] → mean=53.3±15.1")
    print(f"  疎グリッド 300ep（E4-300ep）:  mknee = [64, 32, 32] → mean=42.7±15.1")
    print(f"  密グリッド 300ep（EL）:        mknee = {mk_tau} → mean={np.mean(mk_tau):.1f}±{np.std(mk_tau):.1f}")

    # ── AU vs m table ─────────────────────────────────────────────────────────
    print(f"\n{'m':>4} | " + " | ".join(f"AU(s={s})" for s in seeds))
    for i, m in enumerate(m_list_dense):
        aus = [all_results[str(s)]['dense'][i]['au'] for s in seeds]
        print(f"{m:4d} | " + " | ".join(f"{a:8d}" for a in aus))

    # ── Figure ───────────────────────────────────────────────────────────────
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: dense grid AU per seed
    ax = axes[0]
    for seed, col in zip(seeds, colors):
        res = all_results[str(seed)]['dense']
        aus_s = [r['au'] for r in res]
        mk = all_results[str(seed)]['mknee_tau']
        ax.plot(m_list_dense, aus_s, 'o-', color=col, ms=4, lw=1.5,
                label=f'seed={seed} ($m_{{\\mathrm{{knee}}}}={mk}$)')
        ax.axvline(mk, color=col, ls=':', lw=1, alpha=0.5)
    ax.plot(m_list_dense, m_list_dense, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m')
    ax.axvline(10, color='red', ls='-.', lw=1.5, alpha=0.7, label='$\\hat{d}_{\\mathrm{ID}}=10$')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('EL: Dense grid — AU vs $m$ (3 seeds, 300ep)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: comparison sparse vs dense mknee
    ax2 = axes[1]
    conditions = ['150ep sparse\n(E4)', '300ep sparse\n(E4)', '300ep dense\n(EL)']
    mk_sparse_150 = [64, 32, 64]
    mk_sparse_300 = [64, 32, 32]
    mk_dense_300  = mk_tau

    x = np.arange(3)
    width = 0.25
    for i, (seed, col) in enumerate(zip(seeds, colors)):
        vals = [mk_sparse_150[i], mk_sparse_300[i], mk_dense_300[i]]
        ax2.bar(x + i*width, vals, width, color=col, alpha=0.8, label=f'seed={seed}')
    ax2.axhline(10, color='red', ls='-.', lw=1.5, label='$\\hat{d}_{\\mathrm{ID}}=10$')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(conditions, fontsize=9)
    ax2.set_ylabel('$m^{\\mathrm{knee}}$')
    ax2.set_title('$m_{\\mathrm{knee}}$: grid density × epochs comparison')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/EL_dense_grid_mknee.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EL_dense_grid_mknee.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EL_dense_grid_mknee.{png,pdf}")


if __name__ == '__main__':
    main()
