"""
実験 EB: 位相的AU実験
対応定理: 定理6（AU飽和）の精密化
検証仮説: VAEのAU飽和値は多様体の位相的最小埋め込み次元を推定する
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

from src.models.vae import VAE, vae_loss, train_vae
from src.metrics.structure import active_units

# 乱数シードの固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


# =====================================================================
# 多様体データ生成関数
# =====================================================================

def generate_circle(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    円 S^1 を生成する（d=1, ambient=2）。
    位相的最小埋め込み次元 = 2。
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    return X.astype(np.float32)


def generate_swiss_roll_2d(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Swiss Roll (d=2, simply connected, ambient=3)。
    位相的最小埋め込み次元 = 2。
    """
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n_samples))
    height = rng.uniform(0, 1, n_samples)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    # 正規化
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X.astype(np.float32)


def generate_torus_3d(n_samples: int, R: float = 3.0, r: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    トーラス T^2 (d=2, π₁=Z², ambient=3)。
    位相的最小埋め込み次元 = 3。
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X = np.column_stack([x, y, z])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X.astype(np.float32)


def generate_mobius_band(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    メビウスの帯（非可向, d=2, ambient=3）。
    位相的最小埋め込み次元 = 3。
    """
    rng = np.random.RandomState(seed)
    t = rng.uniform(0, 2 * np.pi, n_samples)
    s = rng.uniform(-0.5, 0.5, n_samples)
    x1 = (1 + s * np.cos(t / 2)) * np.cos(t)
    x2 = (1 + s * np.cos(t / 2)) * np.sin(t)
    x3 = s * np.sin(t / 2)
    X = np.column_stack([x1, x2, x3])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X.astype(np.float32)


def make_dataloader(X: np.ndarray, batch_size: int = 64) -> DataLoader:
    """numpy 配列から DataLoader を作成する。"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_latent_repr_vae(model: VAE, X: np.ndarray) -> np.ndarray:
    """VAEの潜在表現（mu）を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
    return mu.cpu().numpy()


def find_au_saturation(au_values: list, m_values: list) -> int:
    """
    AUが飽和するボトルネック次元を見つける。
    AU(m) == m となる最初のmを飽和点とする。
    """
    for m, au in zip(m_values, au_values):
        if au >= m:
            return m
    return m_values[-1]


def run_manifold_experiment(
    name: str,
    X: np.ndarray,
    expected_min_embed_dim: int,
    m_values: list,
    epochs: int = 200,
    beta: float = 4.0
) -> dict:
    """
    1つの多様体に対してVAEのAU飽和実験を実行する。

    Args:
        name: 多様体名
        X: データ配列
        expected_min_embed_dim: 期待される最小埋め込み次元
        m_values: ボトルネック次元のリスト
        epochs: エポック数
        beta: beta-VAEのKL重み
    Returns:
        results: 実験結果
    """
    print(f"\n=== EB_topology: {name} (expected_min_embed_dim={expected_min_embed_dim}) ===")
    input_dim = X.shape[1]
    train_loader = make_dataloader(X, batch_size=64)

    au_values = []
    mse_values = []

    for m in m_values:
        print(f"\n  m={m} ...")
        model = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[128, 64])
        train_vae(model, train_loader, epochs=epochs, lr=1e-3, beta=beta, device=device)

        # 評価
        model.eval()
        Z = get_latent_repr_vae(model, X)
        au = active_units(Z, threshold=0.01)
        au_values.append(au)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon, mu, logvar = model(X_tensor)
            mse = nn.MSELoss()(recon, X_tensor).item()
        mse_values.append(mse)

        print(f"    AU={au}, MSE={mse:.6f}")

    saturation_m = find_au_saturation(au_values, m_values)
    print(f"\n  AU saturation at m={saturation_m} (expected={expected_min_embed_dim})")

    return {
        'name': name,
        'expected_min_embed_dim': expected_min_embed_dim,
        'm_values': m_values,
        'au_values': au_values,
        'mse_values': mse_values,
        'saturation_m': saturation_m,
    }


def plot_results(all_results: list):
    """AU飽和実験の結果をプロットする。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(1, n_manifolds, figsize=(5 * n_manifolds, 5))
    if n_manifolds == 1:
        axes = [axes]

    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    for ax, res in zip(axes, all_results):
        m_values = res['m_values']
        au_values = res['au_values']
        expected = res['expected_min_embed_dim']
        saturation = res['saturation_m']

        ax.plot(m_values, m_values, 'k--', linewidth=1, alpha=0.5, label='AU=m (diagonal)')
        ax.plot(m_values, au_values, 'o-', color='steelblue', linewidth=2,
                markersize=8, label='Active Units')
        ax.axvline(x=expected, color='red', linestyle='--', linewidth=2,
                   label=f'Expected min embed={expected}')
        ax.axvline(x=saturation, color='green', linestyle=':', linewidth=2,
                   label=f'AU saturation={saturation}')

        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Active Units', fontsize=12)
        ax.set_title(f"{res['name']}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)

    plt.suptitle('EB: AU Saturation vs Topological Min Embedding Dim\n(β-VAE, β=4)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/EB_topology_au.pdf', bbox_inches='tight')
    plt.savefig('results/figures/EB_topology_au.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/EB_topology_au.pdf, .png")


def main():
    """実験 EB のメイン関数。"""
    n_samples = 3000
    epochs = 200
    beta = 4.0
    m_values = [1, 2, 3, 4, 6, 8, 10]

    manifolds = [
        {
            'name': 'Circle_S1',
            'data': generate_circle(n_samples, seed=SEED),
            'expected_min_embed_dim': 2,
            'description': 'Circle S^1 (d=1, topological min embed=2)',
        },
        {
            'name': 'SwissRoll',
            'data': generate_swiss_roll_2d(n_samples, seed=SEED),
            'expected_min_embed_dim': 2,
            'description': 'Swiss Roll (d=2, simply connected, min embed=2)',
        },
        {
            'name': 'Torus_T2',
            'data': generate_torus_3d(n_samples, seed=SEED),
            'expected_min_embed_dim': 3,
            'description': 'Torus T^2 (d=2, π₁=Z², min embed=3)',
        },
        {
            'name': 'MobiusBand',
            'data': generate_mobius_band(n_samples, seed=SEED),
            'expected_min_embed_dim': 3,
            'description': 'Möbius Band (d=2, non-orientable, min embed=3)',
        },
    ]

    all_results = []
    for mf in manifolds:
        res = run_manifold_experiment(
            name=mf['name'],
            X=mf['data'],
            expected_min_embed_dim=mf['expected_min_embed_dim'],
            m_values=m_values,
            epochs=epochs,
            beta=beta
        )
        res['description'] = mf['description']
        all_results.append(res)

    # 結果保存
    with open('results/tables/EB_topology.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/tables/EB_topology.json")

    # プロット
    plot_results(all_results)

    # サマリー
    print("\n=== EB Topology Summary ===")
    print(f"{'Manifold':<20} {'Expected':>10} {'AU_sat':>8} {'Match':>8}")
    print("-" * 50)
    for res in all_results:
        match = "YES" if res['saturation_m'] == res['expected_min_embed_dim'] else "NO"
        print(f"{res['name']:<20} {res['expected_min_embed_dim']:>10} {res['saturation_m']:>8} {match:>8}")
        print(f"  AU values: {list(zip(res['m_values'], res['au_values']))}")


if __name__ == '__main__':
    main()
