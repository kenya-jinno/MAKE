"""
実験 E3: ノイズ方向選択性の定量化
対応定理: 定理4（スコアマッチング）
検証仮説: 法線方向ノイズがAEに除去される（NSRが高い）
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.synthetic import generate_swiss_roll
from src.models.ae import AutoEncoder, train_ae

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


def estimate_tangent_space_local_pca(X: np.ndarray, idx: int,
                                      neighbors: np.ndarray, d: int) -> np.ndarray:
    """
    局所 PCA で点 X[idx] の接空間を推定する。

    Args:
        X: データ配列 (n_samples, N)
        idx: 注目点のインデックス
        neighbors: 近傍点インデックス配列
        d: 接空間の次元
    Returns:
        T: 接空間基底 (N, d)
    """
    local_data = X[neighbors] - X[idx]  # (k, N)
    _, _, Vt = np.linalg.svd(local_data, full_matrices=False)
    return Vt[:d, :].T  # (N, d)


def inject_noise_tangent(x: np.ndarray, T: np.ndarray, sigma: float, rng) -> np.ndarray:
    """
    接空間方向にノイズを加える。

    Args:
        x: 点 (N,)
        T: 接空間基底 (N, d)
        sigma: ノイズ強度
        rng: 乱数生成器
    Returns:
        x_noisy: ノイズ付きデータ (N,)
    """
    d = T.shape[1]
    coeffs = rng.randn(d) * sigma
    noise = T @ coeffs  # 接空間内のノイズ
    return x + noise


def inject_noise_normal(x: np.ndarray, T: np.ndarray, sigma: float, rng) -> np.ndarray:
    """
    法線空間方向にノイズを加える。

    Args:
        x: 点 (N,)
        T: 接空間基底 (N, d)
        sigma: ノイズ強度
        rng: 乱数生成器
    Returns:
        x_noisy: ノイズ付きデータ (N,)
    """
    N = len(x)
    # 接空間への射影行列 P = T @ T^T
    P = T @ T.T  # (N, N)
    # 法線空間への射影 (I - P)
    raw_noise = rng.randn(N) * sigma
    normal_noise = raw_noise - P @ raw_noise  # 法線成分のみ
    # 正規化してスケールを合わせる
    norm = np.linalg.norm(normal_noise) + 1e-8
    normal_noise = normal_noise / norm * (np.linalg.norm(T @ (rng.randn(T.shape[1]) * sigma)) + 1e-8)
    return x + normal_noise


def load_or_train_model(manifold_name: str, m: int, input_dim: int,
                        X_train: np.ndarray, epochs: int = 100) -> AutoEncoder:
    """E1 の学習済みモデルを読み込む。なければ再訓練する。"""
    save_path = f'results/E1_{manifold_name}_m{m}.pth'
    model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"  Loaded model from {save_path}")
    else:
        print(f"  Model not found, retraining ...")
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        train_ae(model, train_loader, epochs=epochs, lr=1e-3, device=device)
        torch.save(model.state_dict(), save_path)

    return model.to(device)


def compute_nsr(model: AutoEncoder, X_test: np.ndarray, T_bases: np.ndarray,
                sigma: float, n_points: int = 200) -> dict:
    """
    NSR (Normal-to-Tangent Suppression Ratio) を計算する。
    NSR = ||rec(x + eps_tangent) - rec(x)|| / ||rec(x + eps_normal) - rec(x)||

    Args:
        model: 学習済み AE
        X_test: テストデータ
        T_bases: 接空間基底配列 (n_samples, N, d)
        sigma: ノイズ強度
        n_points: 評価点数
    Returns:
        dict: NSR, tangent_err, normal_err の統計
    """
    model.eval()
    rng = np.random.RandomState(SEED)
    n = min(n_points, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)

    tangent_errs = []
    normal_errs = []

    for i in idx:
        x = X_test[i]
        T = T_bases[i]  # (N, d)

        # 接空間ノイズ
        x_tan = inject_noise_tangent(x, T, sigma, rng)
        # 法線ノイズ
        x_nor = inject_noise_normal(x, T, sigma, rng)

        # テンソル変換
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        x_tan_t = torch.tensor(x_tan, dtype=torch.float32).unsqueeze(0).to(device)
        x_nor_t = torch.tensor(x_nor, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            rec_x = model(x_t)
            rec_tan = model(x_tan_t)
            rec_nor = model(x_nor_t)

        err_tan = (rec_tan - rec_x).norm().item()
        err_nor = (rec_nor - rec_x).norm().item()
        tangent_errs.append(err_tan)
        normal_errs.append(err_nor)

    tangent_errs = np.array(tangent_errs)
    normal_errs = np.array(normal_errs)
    nsr = tangent_errs / (normal_errs + 1e-8)

    return {
        'nsr_mean': float(np.mean(nsr)),
        'nsr_median': float(np.median(nsr)),
        'nsr_std': float(np.std(nsr)),
        'tangent_err_mean': float(np.mean(tangent_errs)),
        'normal_err_mean': float(np.mean(normal_errs))
    }


def run_noise_direction_experiment(X_train: np.ndarray, X_test: np.ndarray,
                                    T_basis_test: np.ndarray, d_true: int) -> dict:
    """
    ノイズ方向選択性実験を実行する。

    - m = 1, 2 (d_true), 3, 4 のモデルを評価
    - sigma = 0.01, 0.05, 0.1, 0.2 を試す
    """
    input_dim = X_train.shape[1]
    m_values = [1, 2, 3, 4]
    sigma_values = [0.01, 0.05, 0.1, 0.2]

    print("\n=== E3: Noise Direction Selectivity ===")

    results = {
        'd_true': d_true,
        'm_values': m_values,
        'sigma_values': sigma_values,
        'nsr_by_m_sigma': {}
    }

    # 接空間基底の計算（局所 PCA で補完）
    print("\n  Estimating tangent spaces via local PCA ...")
    k_pca = 2 * d_true
    nbrs = NearestNeighbors(n_neighbors=k_pca + 1, algorithm='auto').fit(X_test)
    _, neighbor_indices = nbrs.kneighbors(X_test)

    T_pca = np.zeros_like(T_basis_test)
    for i in range(len(X_test)):
        T_pca[i] = estimate_tangent_space_local_pca(
            X_test, i, neighbor_indices[i, 1:], d_true
        )

    for m in m_values:
        results['nsr_by_m_sigma'][str(m)] = {}
        model = load_or_train_model('SwissRoll', m, input_dim, X_train, epochs=100)
        print(f"\n  m={m}:")

        for sigma in sigma_values:
            nsr_stats = compute_nsr(model, X_test, T_pca, sigma=sigma, n_points=200)
            results['nsr_by_m_sigma'][str(m)][str(sigma)] = nsr_stats
            print(f"    sigma={sigma:.2f}: NSR={nsr_stats['nsr_mean']:.3f} "
                  f"(tan={nsr_stats['tangent_err_mean']:.4f}, "
                  f"nor={nsr_stats['normal_err_mean']:.4f})")

    return results


def plot_nsr_results(results: dict):
    """NSR 結果をプロットする。"""
    m_values = results['m_values']
    sigma_values = results['sigma_values']
    d_true = results['d_true']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # NSR vs m（各 sigma ごと）
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))
    for j, sigma in enumerate(sigma_values):
        nsr_vals = []
        for m in m_values:
            stats = results['nsr_by_m_sigma'][str(m)][str(sigma)]
            nsr_vals.append(stats['nsr_mean'])
        ax.plot(m_values, nsr_vals, 'o-', color=colors[j],
                linewidth=2, label=f'sigma={sigma}')

    ax.axvline(x=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('NSR (tangent/normal response ratio)')
    ax.set_title('Swiss Roll: NSR vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(m_values)

    # Tangent vs Normal error bars（m=d_true のとき）
    ax = axes[1]
    m_sel = d_true  # m = d_true
    tan_errs = []
    nor_errs = []
    for sigma in sigma_values:
        stats = results['nsr_by_m_sigma'][str(m_sel)][str(sigma)]
        tan_errs.append(stats['tangent_err_mean'])
        nor_errs.append(stats['normal_err_mean'])

    x = np.arange(len(sigma_values))
    width = 0.35
    bars1 = ax.bar(x - width/2, tan_errs, width, color='steelblue', label='Tangent noise response')
    bars2 = ax.bar(x + width/2, nor_errs, width, color='coral', label='Normal noise response')
    ax.set_xlabel('Noise level sigma')
    ax.set_ylabel('Reconstruction change norm')
    ax.set_title(f'Swiss Roll (m={m_sel}): Tangent vs Normal Response')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigma_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/E3_nsr_analysis.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E3_nsr_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E3_nsr_analysis.pdf, .png")

    # 追加: NSR vs sigma（各 m ごと）
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_m = plt.cm.plasma(np.linspace(0, 0.8, len(m_values)))
    for j, m in enumerate(m_values):
        nsr_vals = []
        for sigma in sigma_values:
            stats = results['nsr_by_m_sigma'][str(m)][str(sigma)]
            nsr_vals.append(stats['nsr_mean'])
        linestyle = '-' if m == d_true else '--'
        ax.plot(sigma_values, nsr_vals, 'o' + linestyle, color=colors_m[j],
                linewidth=2, label=f'm={m}{"*" if m == d_true else ""}')

    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='NSR=1 (no selectivity)')
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('NSR (mean)')
    ax.set_title('Swiss Roll: NSR vs sigma (by bottleneck dim)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/E3_nsr_vs_sigma.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E3_nsr_vs_sigma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E3_nsr_vs_sigma.pdf, .png")


def main():
    """実験 E3 のメイン関数。"""
    n_train = 3000
    n_test = 1000

    print("Generating Swiss Roll data ...")
    X_train, d_true, T_basis_train = generate_swiss_roll(n_train, noise=0.0, seed=SEED)
    X_test, _, T_basis_test = generate_swiss_roll(n_test, noise=0.0, seed=SEED + 1)

    results = run_noise_direction_experiment(X_train, X_test, T_basis_test, d_true)

    # 結果保存
    with open('results/tables/E3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/tables/E3_results.json")

    # プロット
    plot_nsr_results(results)

    # サマリー
    print("\n=== E3 Summary ===")
    print(f"d_true = {results['d_true']}")
    for m in results['m_values']:
        sigma = results['sigma_values'][1]  # sigma=0.05
        stats = results['nsr_by_m_sigma'][str(m)][str(sigma)]
        print(f"  m={m}, sigma={sigma}: NSR={stats['nsr_mean']:.3f}")


if __name__ == '__main__':
    main()
