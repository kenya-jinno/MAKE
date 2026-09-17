"""
実験 EX: AU閾値δの感度分析
異なる閾値 δ ∈ {0.001, 0.01, 0.05, 0.1} に対して
Swiss Roll の VAE (β=4) で AU カウントがどう変化するかを検証する。
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.vae import VAE, train_vae
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


def make_dataloader(X, batch_size=64):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_delta_sensitivity(manifold_name, X_train, X_test, d_true,
                           m_values, delta_values, epochs=200):
    """
    複数の δ 値に対して VAE の AU カウントを計算する。
    Returns:
        results[m][delta] = AU count
    """
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # m ごとに VAE を1回だけ学習し，複数の δ で AU をカウント
    results = {}
    latent_dict = {}

    print(f"\n=== EX Delta Sensitivity: {manifold_name} ===")
    for m in m_values:
        print(f"  Training VAE m={m} ...")
        torch.manual_seed(SEED)
        vae = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[64, 32])
        train_vae(vae, train_loader, epochs=epochs, lr=1e-3, beta=4.0, device=device)

        vae.eval()
        with torch.no_grad():
            mu, _ = vae.encode(X_test_tensor)
            Z = mu.cpu().numpy()

        latent_dict[m] = Z
        results[m] = {}
        variances = np.var(Z, axis=0)
        for delta in delta_values:
            au = int(np.sum(variances > delta))
            results[m][delta] = au
            print(f"    m={m}, δ={delta:.3f}: AU={au}  (variances range: {variances.min():.4f}~{variances.max():.4f})")

    return results


def plot_delta_sensitivity(results_sr, results_tor, m_values, delta_values, d_true=2):
    """AU閾値感度分析のプロット。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for ax, (results, title) in zip(axes, [
        (results_sr, 'Swiss Roll ($d_{true}=2$)'),
        (results_tor, 'Torus ($d_{true}=2$)')
    ]):
        for (delta, color, marker) in zip(delta_values, colors, markers):
            au_vals = [results[m][delta] for m in m_values]
            ax.plot(m_values, au_vals, marker=marker, linestyle='-',
                    color=color, linewidth=2, markersize=8,
                    label=f'δ = {delta}')

        ax.axhline(y=d_true, color='red', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'$d_{{true}}={d_true}$')
        ax.set_xlabel('Bottleneck dim $m$', fontsize=13)
        ax.set_ylabel('Active Units (AU)', fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)
        ax.set_ylim(0, max(m_values) + 1)

    plt.suptitle('AU Threshold Sensitivity Analysis (VAE, β=4)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/EX_au_delta_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('results/figures/EX_au_delta_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EX_au_delta_sensitivity.pdf, .png")


def main():
    n_train = 3000
    n_test = 1000
    epochs = 200
    m_values = [2, 3, 4, 6, 8, 10, 12, 16]
    delta_values = [0.001, 0.01, 0.05, 0.1]

    # Swiss Roll
    print("Generating Swiss Roll data ...")
    X_train_sr, d_true_sr, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test_sr, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)
    results_sr = run_delta_sensitivity(
        'SwissRoll', X_train_sr, X_test_sr, d_true_sr,
        m_values, delta_values, epochs=epochs
    )

    # Torus
    print("\nGenerating Torus data ...")
    X_train_tor, d_true_tor, _ = generate_torus(n_train, R=3.0, r=1.0, noise=0.01, seed=SEED)
    X_test_tor, _, _ = generate_torus(n_test, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)
    results_tor = run_delta_sensitivity(
        'Torus', X_train_tor, X_test_tor, d_true_tor,
        m_values, delta_values, epochs=epochs
    )

    # 結果保存
    save_data = {
        'swiss_roll': {str(m): {str(d): v for d, v in dv.items()}
                       for m, dv in results_sr.items()},
        'torus': {str(m): {str(d): v for d, v in dv.items()}
                  for m, dv in results_tor.items()},
        'm_values': m_values,
        'delta_values': delta_values,
    }
    with open('results/tables/EX_au_delta_sensitivity.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("Saved: results/tables/EX_au_delta_sensitivity.json")

    # プロット
    plot_delta_sensitivity(results_sr, results_tor, m_values, delta_values)

    # サマリー表示
    print("\n=== EX Summary: Swiss Roll (VAE, β=4) ===")
    header = "  m  | " + " | ".join([f"δ={d}" for d in delta_values])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in m_values:
        row = f"  {m:2d} | " + " | ".join([f"AU={results_sr[m][d]:2d}" for d in delta_values])
        print(row)

    print("\n=== EX Summary: Torus (VAE, β=4) ===")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in m_values:
        row = f"  {m:2d} | " + " | ".join([f"AU={results_tor[m][d]:2d}" for d in delta_values])
        print(row)


if __name__ == '__main__':
    main()
