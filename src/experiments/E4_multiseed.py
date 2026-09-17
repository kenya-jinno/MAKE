"""
実験 E4_multiseed: MNIST VAE/AE の多シード再現性実験
目的: mknee=12 の統計的頑健性を複数シードで検証
Seeds: 42, 123, 777
m: [4, 8, 12, 16, 20, 32, 64]
epochs: 150 (E4 の 300 よりやや短縮)
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


def train_one(m, X_train, X_test, epochs=150, beta=4.0, batch_size=128):
    vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=m)
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
    vae.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = vae(X_te_t)
        mse = nn.functional.mse_loss(x_hat, X_te_t).item()
        Z = mu_te.cpu().numpy()
    au = active_units(Z, threshold=1e-2)
    id_2nn = twonn_estimate(Z)
    id_mle  = mle_estimate(Z, k=10)
    return {'m': m, 'mse': mse, 'au': au, 'twonn': id_2nn, 'mle': id_mle}


def compute_mknee(results, tau=0.25):
    """AU増加率がτを下回る最初のmを返す"""
    ms  = [r['m']  for r in results]
    aus = [r['au'] for r in results]
    for i in range(1, len(ms)):
        dm  = ms[i]  - ms[i-1]
        dau = aus[i] - aus[i-1]
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def main():
    seeds  = [42, 123, 777]
    m_list = [4, 8, 12, 16, 20, 32, 64]
    epochs = 150

    X_train, X_test = load_mnist()
    print(f"Data: train={len(X_train)}, test={len(X_test)}")

    all_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_one(m, X_train, X_test, epochs=epochs, beta=4.0)
            res.append(r)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, MSE={r['mse']:.4f}")
        all_seed_results[str(seed)] = res
        # partial save
        with open('results/tables/E4_multiseed.json', 'w') as f:
            json.dump(all_seed_results, f, indent=2)
        mknee = compute_mknee(res, tau=0.25)
        print(f"  mknee(τ=0.25)={mknee}")

    # Summary
    print("\n=== Multi-seed Summary ===")
    print(f"{'m':>4} | {'AU (mean±std)':>16} | {'TwoNN (mean±std)':>18} | {'mknee per seed':>20}")
    for i, m in enumerate(m_list):
        aus   = [all_seed_results[str(s)][i]['au']    for s in seeds]
        twons = [all_seed_results[str(s)][i]['twonn'] for s in seeds]
        print(f"{m:4d} | {np.mean(aus):6.1f}±{np.std(aus):4.1f}  | "
              f"{np.mean(twons):6.2f}±{np.std(twons):4.2f}       | "
              f"{[compute_mknee(all_seed_results[str(s)], tau=0.25) for s in seeds]}")

    mknees = [compute_mknee(all_seed_results[str(s)], tau=0.25) for s in seeds]
    print(f"\nmknee per seed: {mknees}, mean={np.mean(mknees):.1f}, std={np.std(mknees):.1f}")

    # Figure: AU vs m with error bands
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Left: AU vs m per seed
    ax = axes[0]
    for seed, col in zip(seeds, colors):
        res = all_seed_results[str(seed)]
        ms  = [r['m']  for r in res]
        aus = [r['au'] for r in res]
        ax.plot(ms, aus, 'o-', color=col, label=f'seed={seed}', ms=5)
    ax.plot(ms, ms, 'k--', lw=1, alpha=0.5, label='AU=m (AE)')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('VAE AU vs $m$ (3 seeds)')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for seed, col in zip(seeds, colors):
        mk = compute_mknee(all_seed_results[str(seed)], tau=0.25)
        ax.axvline(mk, color=col, ls=':', lw=1)

    # Right: mean±std AU
    ax2 = axes[1]
    all_aus   = np.array([[all_seed_results[str(s)][i]['au']    for i in range(len(m_list))] for s in seeds])
    all_twons = np.array([[all_seed_results[str(s)][i]['twonn'] for i in range(len(m_list))] for s in seeds])
    mu_au  = all_aus.mean(0);   sd_au  = all_aus.std(0)
    mu_id  = all_twons.mean(0); sd_id  = all_twons.std(0)

    ax2r = ax2.twinx()
    ax2.fill_between(m_list, mu_au-sd_au, mu_au+sd_au, alpha=0.2, color='tab:blue')
    ax2.plot(m_list, mu_au, 'b-o', ms=5, label='AU (mean±std)')
    ax2r.fill_between(m_list, mu_id-sd_id, mu_id+sd_id, alpha=0.2, color='tab:red')
    ax2r.plot(m_list, mu_id, 'r--s', ms=5, label='TwoNN ID (mean±std)')
    ax2.axvline(np.mean(mknees), color='purple', ls='--', lw=1.5,
                label=f'$m_{{\\mathrm{{knee}}}}={int(np.mean(mknees))}$')
    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('AU', color='b')
    ax2r.set_ylabel('TwoNN ID', color='r')
    ax2.set_title('Mean±Std over 3 seeds (VAE, MNIST)')
    ax2.set_xscale('log')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2r.tick_params(axis='y', labelcolor='r')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/E4_multiseed.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/E4_multiseed.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E4_multiseed.{png,pdf}")


if __name__ == '__main__':
    main()
