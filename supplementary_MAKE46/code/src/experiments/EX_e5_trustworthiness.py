"""
EX_e5_trustworthiness: 実験 E5
Swiss Roll と Torus において Trustworthiness の m 依存性を計測し，
m = d_true での急激な改善を確認する（定理5：近傍構造保存）
3シード × m=1,2,3,4 × 2多様体
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

from src.models.ae import AutoEncoder, train_ae
from src.data.synthetic import generate_swiss_roll, generate_torus
from src.metrics.structure import trustworthiness

SEEDS = [42, 123, 456]
M_VALUES = [1, 2, 3, 4]
EPOCHS = 200
N_TRAIN = 4000
N_TEST = 1000
BATCH_SIZE = 64
HIDDEN_DIMS = [64, 32]
K_TRUST = 10

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


def run_e5_single(manifold_name, gen_func, seed):
    set_seed(seed)
    X_all, _, _ = gen_func(N_TRAIN + N_TEST, seed=seed)
    X_train, X_test = X_all[:N_TRAIN], X_all[N_TRAIN:]
    input_dim = X_train.shape[1]

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    results = {}
    for m in M_VALUES:
        set_seed(seed)
        model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=HIDDEN_DIMS).to(device)
        train_ae(model, loader, epochs=EPOCHS, lr=1e-3, device=device)
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            Z_test = model.encode(X_test_t).cpu().numpy()
        trust = float(trustworthiness(X_test, Z_test, k=K_TRUST))
        results[m] = trust
        print(f"  [{manifold_name}] seed={seed}, m={m}: Trust={trust:.4f}")
    return results


def run_e5():
    print("=" * 60)
    print("実験 E5: Trustworthiness の m 依存性（3シード）")
    print("=" * 60)

    manifolds = [
        ('Swiss Roll', generate_swiss_roll, 2),
        ('Torus',      generate_torus,      2),
    ]

    all_results = {}

    for name, gen_func, dtrue in manifolds:
        print(f"\n--- {name} (d_true={dtrue}) ---")
        seed_data = {m: [] for m in M_VALUES}
        for seed in SEEDS:
            r = run_e5_single(name, gen_func, seed)
            for m in M_VALUES:
                seed_data[m].append(r[m])

        stats = {}
        for m in M_VALUES:
            vals = seed_data[m]
            stats[m] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
            }
        all_results[name] = {'stats': stats, 'dtrue': dtrue}

        print(f"\n  {name} 結果まとめ（3シード mean±std）：")
        print(f"  {'m':>4}  {'Trustworthiness':>20}")
        for m in M_VALUES:
            s = stats[m]
            star = ' ← d_true' if m == dtrue else ''
            print(f"  {m:>4}  {s['mean']:.4f}±{s['std']:.4f}{star}")

    # Save JSON
    with open('results/tables/E5_trustworthiness.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/tables/E5_trustworthiness.json")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m_arr = np.array(M_VALUES)

    for ax, (name, _, dtrue) in zip(axes, manifolds):
        stats = all_results[name]['stats']
        means = [stats[m]['mean'] for m in M_VALUES]
        stds  = [stats[m]['std']  for m in M_VALUES]
        ax.errorbar(m_arr, means, yerr=stds, fmt='o-', capsize=5,
                    color='steelblue', linewidth=2, markersize=7)
        ax.axvline(x=dtrue, color='tomato', linestyle='--', linewidth=1.5,
                   label=f'$m = d_{{\\rm true}} = {dtrue}$')
        ax.set_xlabel('Bottleneck dim $m$', fontsize=12)
        ax.set_ylabel('Trustworthiness ($k=10$)', fontsize=12)
        ax.set_title(f'{name} (3 seeds mean±std)', fontsize=12)
        ax.set_xticks(M_VALUES)
        ax.set_ylim(0.85, 1.01)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/E5_trustworthiness.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/E5_trustworthiness.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E5_trustworthiness.png")

    return all_results


if __name__ == '__main__':
    results = run_e5()
