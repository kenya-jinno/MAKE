"""
実験 EZ: τ_knee 閾値のロバスト性検証
目的: Fashion-MNIST および合成 3 クラス Gaussian 混合分布を用いて、
      τ_knee = 0.25 が MNIST 以外のデータでも機能するかを検証する。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.vae import VAE, vae_loss
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


# -------------------------------------------------------
# Dataset loaders
# -------------------------------------------------------

def load_fashion_mnist(n_train=10000, n_test=2000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_tr = np.array([img.view(-1).numpy() for img, _ in train_ds], dtype=np.float32)[:n_train]
    X_te = np.array([img.view(-1).numpy() for img, _ in test_ds],  dtype=np.float32)[:n_test]
    print(f"  Fashion-MNIST: train={len(X_tr)}, test={len(X_te)}")
    return X_tr, X_te


def make_gaussian_mixture(n_components=5, dim=50, n_per_comp=800, sigma=0.5):
    """低次元 Gaussian 混合を高次元空間に埋め込む合成データ。
    各クラスセンターは dim 次元の d_true=5 次元部分空間上に置く。"""
    rng = np.random.RandomState(SEED)
    d_true = 5
    # d_true 次元の部分空間基底
    basis = rng.randn(dim, d_true)
    basis, _ = np.linalg.qr(basis)  # orthonormal
    centers = rng.randn(n_components, d_true) * 3.0
    X = []
    for c in range(n_components):
        pts = rng.randn(n_per_comp, d_true) * sigma + centers[c]
        X.append(pts @ basis.T)
    X = np.vstack(X).astype(np.float32)
    # normalize to [0,1] for VAE sigmoid
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    np.random.shuffle(X)
    n_train = int(len(X) * 0.8)
    print(f"  GaussianMixture({n_components} comp, dim={dim}): train={n_train}, test={len(X)-n_train}, d_true={d_true}")
    return X[:n_train], X[n_train:], d_true


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_vae(vae, X_train, X_test, epochs=150, batch_size=128, beta=4.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    vae.to(device); vae.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
    # evaluate
    vae.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = vae(X_te_t)
        mse = nn.functional.mse_loss(x_hat, X_te_t).item()
        Z = mu_te.cpu().numpy()
    au = active_units(Z, threshold=1e-2)
    return mse, au, Z


def run_sweep_dataset(X_train, X_test, input_dim, m_list, hidden_dims,
                      epochs=150, label=''):
    results = []
    prev_au = None
    for m in m_list:
        vae = VAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=m)
        mse, au, Z = train_vae(vae, X_train, X_test, epochs=epochs, beta=4.0)
        id_2nn = twonn_estimate(Z) if len(Z) > 10 else float('nan')
        results.append({'m': m, 'mse': mse, 'au': au, 'twonn': id_2nn})
        print(f"  [{label}] m={m:3d}: AU={au:3d}, TwoNN ID={id_2nn:.2f}, MSE={mse:.4f}")
    return results


# -------------------------------------------------------
# τ_knee identification
# -------------------------------------------------------

def compute_r(results, m_grid):
    """Normalized AU growth rate r(m)."""
    aus = {r['m']: r['au'] for r in results}
    rates = {}
    for i in range(1, len(m_grid)):
        m, m_prev = m_grid[i], m_grid[i-1]
        rates[m] = (aus[m] - aus[m_prev]) / (m - m_prev) if m != m_prev else 0.0
    return rates


def find_knee(results, m_grid, tau=0.25):
    rates = compute_r(results, m_grid)
    if not rates:
        return None, None, None
    r_max = max(rates.values())
    if r_max == 0:
        return m_grid[0], rates, r_max
    for m in m_grid[1:]:
        if m in rates and rates[m] < tau * r_max:
            return m, rates, r_max
    return m_grid[-1], rates, r_max


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    m_list = [1, 2, 4, 8, 12, 16, 20, 32, 64]
    all_results = {}

    # 1) Fashion-MNIST
    print("\n=== Fashion-MNIST ===")
    X_tr, X_te = load_fashion_mnist(n_train=10000, n_test=2000)
    res_fmnist = run_sweep_dataset(X_tr, X_te, input_dim=784,
                                   m_list=m_list, hidden_dims=[512, 256],
                                   epochs=150, label='FashionMNIST')
    all_results['FashionMNIST'] = res_fmnist

    # 2) Gaussian Mixture (synthetic, d_true=5, dim=50, 5 components)
    print("\n=== GaussianMixture (5 comp, dim=50, d_true=5) ===")
    X_tr_g, X_te_g, d_true_g = make_gaussian_mixture(n_components=5, dim=50,
                                                       n_per_comp=800, sigma=0.5)
    res_gmix = run_sweep_dataset(X_tr_g, X_te_g, input_dim=50,
                                  m_list=[1,2,3,4,5,6,8,10,12,16,20],
                                  hidden_dims=[64, 32],
                                  epochs=200, label='GaussianMix')
    all_results['GaussianMix'] = res_gmix
    all_results['GaussianMix_dtrue'] = d_true_g

    # 3) Gaussian Mixture (10 components, d_true=5) — stronger entanglement
    print("\n=== GaussianMixture (10 comp, dim=50, d_true=5) ===")
    X_tr_g10, X_te_g10, _ = make_gaussian_mixture(n_components=10, dim=50,
                                                     n_per_comp=500, sigma=0.5)
    res_gmix10 = run_sweep_dataset(X_tr_g10, X_te_g10, input_dim=50,
                                    m_list=[1,2,3,4,5,6,8,10,12,16,20],
                                    hidden_dims=[64, 32],
                                    epochs=200, label='GaussianMix10')
    all_results['GaussianMix10'] = res_gmix10

    # Save
    with open('results/tables/EZ_tau_robustness.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ---- τ_knee analysis ----
    print("\n=== τ_knee analysis ===")
    tau_vals = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    summary = {}
    for ds_label, res in [('FashionMNIST', res_fmnist),
                           ('GaussianMix',  res_gmix),
                           ('GaussianMix10',res_gmix10)]:
        mg = [r['m'] for r in res]
        knees = {}
        for tau in tau_vals:
            knee_m, rates, rmax = find_knee(res, mg, tau=tau)
            knees[tau] = knee_m
        summary[ds_label] = knees
        print(f"\n{ds_label}:")
        print(f"  AU_sat = {max(r['au'] for r in res)}")
        print(f"  TwoNN ID (large m) = {res[-1]['twonn']:.2f}")
        print(f"  m_knee vs tau:", {f"{tau:.2f}": knees[tau] for tau in tau_vals})

    all_results['tau_summary'] = {
        k: {str(t): v for t, v in vv.items()} for k, vv in summary.items()
    }
    with open('results/tables/EZ_tau_robustness.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    datasets = [
        ('FashionMNIST', res_fmnist, 'tab:blue',   m_list),
        ('GaussianMix',  res_gmix,   'tab:green',  [1,2,3,4,5,6,8,10,12,16,20]),
        ('GaussianMix10',res_gmix10, 'tab:orange', [1,2,3,4,5,6,8,10,12,16,20]),
    ]
    for col, (ds_lbl, res, color, mg) in enumerate(datasets):
        ms  = [r['m']    for r in res]
        aus = [r['au']   for r in res]
        ids = [r['twonn'] for r in res]
        axes[0, col].plot(ms, aus, 'o-', color=color)
        axes[0, col].set_title(ds_lbl)
        axes[0, col].set_xlabel('m'); axes[0, col].set_ylabel('AU (VAE, β=4)')
        axes[0, col].set_xscale('log'); axes[0, col].grid(True, alpha=0.4)

        # r(m) curves for different τ
        rates = compute_r(res, mg)
        r_max = max(rates.values()) if rates else 1.0
        ms2 = [m for m in mg[1:] if m in rates]
        rs  = [rates[m] / (r_max + 1e-9) for m in ms2]
        axes[1, col].plot(ms2, rs, 'ko-', label='r(m)/r_max', linewidth=2)
        for tau, ls in [(0.1,'--'),(0.25,'-'),(0.5,':')]:
            axes[1, col].axhline(tau, linestyle=ls, color='red', alpha=0.7, label=f'τ={tau}')
        axes[1, col].set_title(f'{ds_lbl}: normalized AU growth rate')
        axes[1, col].set_xlabel('m'); axes[1, col].set_ylabel('r(m) / r_max')
        axes[1, col].grid(True, alpha=0.4)
        if col == 0:
            axes[1, col].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('results/figures/EZ_tau_robustness.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EZ_tau_robustness.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/EZ_tau_robustness.{png,pdf}")


if __name__ == '__main__':
    main()
