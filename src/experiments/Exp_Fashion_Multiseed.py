"""
実験 Exp-Fashion-MS: Fashion-MNIST FC-VAE の Step 3 多シード検証（査読対応）
目的: Fashion-MNIST の Step 3（AU 検証）を単一シード予備観察（Exp-Fashion, 100 ep）から
     MNIST 主実験（Exp-Mnist-Multi-300）と同一プロトコルの 3 シード検証に格上げする．

Model: FC-VAE 784-[512,256]-m（E4_300ep_multiseed.py と同一）
β = 4, epochs = 300, seeds = [42, 123, 777], m = [4, 8, 12, 16, 20, 32, 64]
検証点 m_ver = 2 * dhat_ID = 20（Fashion-MNIST の dhat_ID ≈ 10; Exp-Real-Geom）
出力: results/tables/Exp_Fashion_MS.json，results/figures/Exp_Fashion_MS.{png,pdf}
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

DHAT = 10          # Fashion-MNIST dhat_ID (TwoNN, Exp-Real-Geom, 3 seeds)
M_VER = 20         # 検証点 2*dhat_ID


def load_fashion(n_train=20000, n_test=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(
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


def train_one(m, X_train, X_test, epochs=300, beta=4.0, batch_size=128):
    vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=m)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    vae.to(device); vae.train()
    for ep in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
    vae.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = vae(X_te_t)
        mse = nn.functional.mse_loss(x_hat, X_te_t).item()
        Z = mu_te.cpu().numpy()
    au     = active_units(Z, threshold=1e-2)
    id_2nn = twonn_estimate(Z)
    id_mle = mle_estimate(Z, k=10)
    return {'m': m, 'mse': float(mse), 'au': int(au),
            'twonn': float(id_2nn), 'mle': float(id_mle)}


def main():
    seeds  = [42, 123, 777]
    m_list = [4, 8, 12, 16, 20, 32, 64]
    epochs = 300

    X_train, X_test = load_fashion()
    print(f"Data: Fashion-MNIST train={len(X_train)}, test={len(X_test)}")

    out_path = 'results/tables/Exp_Fashion_MS.json'
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    for seed in seeds:
        if str(seed) in all_results and len(all_results[str(seed)]) == len(m_list):
            print(f"Seed {seed}: already done, skipping")
            continue
        set_seed(seed)
        print(f"\n=== Seed {seed} (300 epochs, beta=4) ===")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_one(m, X_train, X_test, epochs=epochs, beta=4.0)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}", flush=True)
        all_results[str(seed)] = res
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print("  Saved partial results")

    # ---- Summary ----
    print("\n=== Exp-Fashion-MS Summary (3 seeds, 300 ep) ===")
    print(f"{'m':>4} | {'AU (mean±std)':>16} | {'TwoNN (mean±std)':>18} | {'MSE (mean)':>10}")
    for i, m in enumerate(m_list):
        aus   = [all_results[str(s)][i]['au']    for s in seeds]
        twons = [all_results[str(s)][i]['twonn'] for s in seeds]
        mses  = [all_results[str(s)][i]['mse']   for s in seeds]
        print(f"{m:4d} | {np.mean(aus):6.1f}±{np.std(aus):4.1f}      | "
              f"{np.mean(twons):6.2f}±{np.std(twons):4.2f}      | {np.mean(mses):.4f}")

    # ---- V1/V2 verdict at m_ver ----
    i_ver = m_list.index(M_VER)
    aus_ver = [all_results[str(s)][i_ver]['au'] for s in seeds]
    au_mean = np.mean(aus_ver)
    v1 = au_mean / M_VER <= 0.9
    v2 = DHAT <= au_mean <= 3 * DHAT
    print(f"\nVerification at m_ver={M_VER} (dhat={DHAT}):")
    print(f"  AU(m_ver) per seed: {aus_ver}, mean={au_mean:.1f}")
    print(f"  V1 (truncation): AU/m = {au_mean/M_VER:.3f} <= 0.9 ? {'PASS' if v1 else 'FAIL'}")
    print(f"  V2 (order match): {DHAT} <= {au_mean:.1f} <= {3*DHAT} ? {'PASS' if v2 else 'FAIL'}")
    per_seed_v1 = [au / M_VER <= 0.9 for au in aus_ver]
    per_seed_v2 = [DHAT <= au <= 3 * DHAT for au in aus_ver]
    print(f"  per-seed V1: {per_seed_v1}, per-seed V2: {per_seed_v2}")

    # ---- Figure (B/W-friendly) ----
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    styles = [dict(color='tab:blue', ls='-', marker='o'),
              dict(color='tab:orange', ls='--', marker='s'),
              dict(color='tab:green', ls='-.', marker='^')]

    ax = axes[0]
    for seed, st in zip(seeds, styles):
        res = all_results[str(seed)]
        ms  = [r['m'] for r in res]
        aus = [r['au'] for r in res]
        ax.plot(ms, aus, lw=2, ms=6, label=f'seed={seed}', **st)
    ax.plot(m_list, m_list, 'k:', lw=1.5, alpha=0.6, label='AU $=m$')
    ax.axvspan(DHAT, 2 * DHAT, color='gray', alpha=0.15,
               label=r'$[\hat{d}_{\mathrm{ID}},\ 2\hat{d}_{\mathrm{ID}}]$')
    ax.set_xlabel('Bottleneck dimension $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('Fashion-MNIST FC-VAE: AU vs. $m$ (3 seeds, 300 ep)')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    au_mat = np.array([[all_results[str(s)][i]['au'] for i in range(len(m_list))] for s in seeds])
    mse_mat = np.array([[all_results[str(s)][i]['mse'] for i in range(len(m_list))] for s in seeds])
    ax2.errorbar(m_list, au_mat.mean(0), yerr=au_mat.std(0), color='tab:blue',
                 ls='-', marker='o', lw=2, ms=6, capsize=3, label='AU mean$\\pm$std')
    ax2.axhline(DHAT, color='purple', ls='--', lw=2,
                label=r'$\hat{d}_{\mathrm{ID}} \approx 10$ (Step 1)')
    ax2r = ax2.twinx()
    ax2r.errorbar(m_list, mse_mat.mean(0), yerr=mse_mat.std(0), color='tab:red',
                  ls='-.', marker='s', lw=2, ms=6, capsize=3, label='MSE mean$\\pm$std')
    ax2.set_xlabel('Bottleneck dimension $m$')
    ax2.set_ylabel('AU')
    ax2r.set_ylabel('Test MSE')
    ax2.set_title('Mean$\\pm$std over 3 seeds (Fashion-MNIST)')
    ax2.set_xscale('log')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/Exp_Fashion_MS.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/Exp_Fashion_MS.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/Exp_Fashion_MS.{png,pdf}")


if __name__ == '__main__':
    main()
