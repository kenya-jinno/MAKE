"""
EX_mnist_multiseed: MNIST 多シードロバスト性検証
3シード × 主要 m 値でのAE/VAE学習による mean±std 推定
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder
from src.models.vae import VAE, vae_loss
from src.metrics.intrinsic_dim import twonn_estimate
from src.metrics.structure import active_units, centered_kernel_alignment

SEEDS = [42, 123, 456]
M_VALUES = [8, 12, 16, 20, 32]
EPOCHS = 150  # 安定性確認用（収束後の分散推定）
BETA = 4.0
N_TRAIN = 20000
N_TEST = 3000
BATCH_SIZE = 128
HIDDEN_DIMS = [512, 256]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_mnist(seed=42):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform
    )
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(train_ds), N_TRAIN, replace=False)
    test_idx = rng.choice(len(test_ds), N_TEST, replace=False)
    X_train = np.stack([train_ds[i][0].numpy().flatten() for i in train_idx]).astype('float32')
    X_test = np.stack([test_ds[i][0].numpy().flatten() for i in test_idx]).astype('float32')
    return X_train, X_test


def train_ae_model(m, X_train, X_test, epochs):
    model = AutoEncoder(784, m, HIDDEN_DIMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), x).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(device)
        mse = loss_fn(model(X_test_t), X_test_t).item()
        Z_test = model.encode(X_test_t).cpu().numpy()
    au = float(Z_test.shape[1])  # AE always AU = m
    id_est = twonn_estimate(Z_test)
    return {'mse': mse, 'au': au, 'twonn_id': id_est}


def train_vae_model(m, X_train, X_test, epochs):
    model = VAE(784, m, HIDDEN_DIMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(device)
        recon, mu, logvar = model(X_test_t)
        mse = nn.MSELoss()(recon, X_test_t).item()
        # AU: use all training data for variance estimate
        X_all_t = torch.tensor(X_train).to(device)
        all_mus = []
        for i in range(0, len(X_all_t), 512):
            batch = X_all_t[i:i+512]
            _, mu_b, _ = model(batch)
            all_mus.append(mu_b.cpu())
        all_mus = torch.cat(all_mus, dim=0).numpy()
    au_count = active_units(all_mus)
    id_est = twonn_estimate(mu.cpu().numpy())
    return {'mse': mse, 'au': au_count, 'twonn_id': id_est}


def run_multiseed():
    print(f"Running multi-seed MNIST stability check: {len(SEEDS)} seeds × {len(M_VALUES)} m-values × 2 models")
    ae_results = {m: [] for m in M_VALUES}
    vae_results = {m: [] for m in M_VALUES}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
        set_seed(seed)
        X_train, X_test = load_mnist(seed=seed)

        for m in M_VALUES:
            print(f"  AE  m={m}...", end=' ', flush=True)
            set_seed(seed)
            r = train_ae_model(m, X_train, X_test, EPOCHS)
            ae_results[m].append(r)
            print(f"MSE={r['mse']:.4f}")

            print(f"  VAE m={m}...", end=' ', flush=True)
            set_seed(seed)
            r = train_vae_model(m, X_train, X_test, EPOCHS)
            vae_results[m].append(r)
            print(f"MSE={r['mse']:.4f} AU={r['au']}")

    # Compute mean ± std
    ae_summary = {}
    vae_summary = {}
    for m in M_VALUES:
        ae_vals = ae_results[m]
        ae_summary[m] = {
            'mse_mean': np.mean([v['mse'] for v in ae_vals]),
            'mse_std': np.std([v['mse'] for v in ae_vals]),
            'au_mean': np.mean([v['au'] for v in ae_vals]),
            'au_std': np.std([v['au'] for v in ae_vals]),
            'twonn_mean': np.mean([v['twonn_id'] for v in ae_vals]),
            'twonn_std': np.std([v['twonn_id'] for v in ae_vals]),
        }
        vae_vals = vae_results[m]
        vae_summary[m] = {
            'mse_mean': np.mean([v['mse'] for v in vae_vals]),
            'mse_std': np.std([v['mse'] for v in vae_vals]),
            'au_mean': np.mean([v['au'] for v in vae_vals]),
            'au_std': np.std([v['au'] for v in vae_vals]),
            'twonn_mean': np.mean([v['twonn_id'] for v in vae_vals]),
            'twonn_std': np.std([v['twonn_id'] for v in vae_vals]),
        }

    # Print table
    print("\n\n=== AE Results (mean ± std) ===")
    print(f"{'m':>4} | {'MSE':>15} | {'AU':>10} | {'TwoNN ID':>10}")
    print("-" * 50)
    for m in M_VALUES:
        s = ae_summary[m]
        print(f"{m:>4} | {s['mse_mean']:.4f}±{s['mse_std']:.4f} | {s['au_mean']:.1f}±{s['au_std']:.1f} | {s['twonn_mean']:.2f}±{s['twonn_std']:.2f}")

    print("\n=== VAE Results (mean ± std) ===")
    print(f"{'m':>4} | {'MSE':>15} | {'AU':>10} | {'TwoNN ID':>10}")
    print("-" * 50)
    for m in M_VALUES:
        s = vae_summary[m]
        print(f"{m:>4} | {s['mse_mean']:.4f}±{s['mse_std']:.4f} | {s['au_mean']:.1f}±{s['au_std']:.1f} | {s['twonn_mean']:.2f}±{s['twonn_std']:.2f}")

    # Save JSON
    output = {
        'ae': {str(m): ae_summary[m] for m in M_VALUES},
        'vae': {str(m): vae_summary[m] for m in M_VALUES},
        'ae_raw': {str(m): ae_results[m] for m in M_VALUES},
        'vae_raw': {str(m): vae_results[m] for m in M_VALUES},
        'config': {'seeds': SEEDS, 'm_values': M_VALUES, 'epochs': EPOCHS, 'n_train': N_TRAIN}
    }
    with open('results/tables/EX_mnist_multiseed.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/tables/EX_mnist_multiseed.json")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    m_arr = np.array(M_VALUES)

    for ax, key, ylabel in zip(axes, ['mse', 'au', 'twonn'], ['MSE', 'AU', 'TwoNN ID']):
        ae_means = [ae_summary[m][f'{key}_mean'] for m in M_VALUES]
        ae_stds  = [ae_summary[m][f'{key}_std']  for m in M_VALUES]
        vae_means = [vae_summary[m][f'{key}_mean'] for m in M_VALUES]
        vae_stds  = [vae_summary[m][f'{key}_std']  for m in M_VALUES]

        ax.errorbar(m_arr, ae_means, yerr=ae_stds, fmt='o-', label='AE', capsize=4, color='steelblue')
        ax.errorbar(m_arr + 0.3, vae_means, yerr=vae_stds, fmt='s--', label='VAE', capsize=4, color='tomato')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel(ylabel)
        ax.set_title(f'MNIST {ylabel} (mean±std, 3 seeds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EX_mnist_multiseed.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EX_mnist_multiseed.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figure to results/figures/EX_mnist_multiseed.png")

    return ae_summary, vae_summary


if __name__ == '__main__':
    run_multiseed()
