"""
実験 E5: 構造保存メトリクス評価
対応定理: 定理7（近傍構造保存）
検証仮説: m=d_true のとき Trustworthiness/Continuity が最大化される
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

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder, train_ae
from src.metrics.structure import (
    active_units, trustworthiness, continuity, geodesic_distance_correlation
)

# 乱数シードの固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# デバイス設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 結果保存先
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def make_dataloader(X: np.ndarray, batch_size: int = 64, train: bool = True) -> DataLoader:
    """numpy 配列から DataLoader を作成する。"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_latent_repr(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """潜在表現を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(X_tensor)
    return Z.cpu().numpy()


def evaluate_mse(model: nn.Module, X_test: np.ndarray) -> float:
    """テストデータの MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return mse


def run_structure_experiment(
    manifold_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    d_true: int,
    m_values: list,
    n_geo_samples: int = 500,
    k: int = 10,
    epochs: int = 200
) -> dict:
    """
    構造保存メトリクス実験を実行する。

    Args:
        manifold_name: 多様体名
        X_train: 訓練データ
        X_test: テストデータ
        d_true: 真の内在次元
        m_values: ボトルネック次元のリスト
        n_geo_samples: 測地線距離計算に使うサンプル数
        k: 近傍数
        epochs: 訓練エポック数
    Returns:
        results: 実験結果の辞書
    """
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64, train=True)

    results = {
        'manifold': manifold_name,
        'd_true': d_true,
        'm_values': m_values,
        'mse': [],
        'au': [],
        'trustworthiness': [],
        'continuity': [],
        'geodesic_corr': [],
    }

    print(f"\n=== Experiment E5: {manifold_name} (d_true={d_true}) ===")

    # 測地線距離計算用のサブサンプル
    n_test = X_test.shape[0]
    geo_idx = np.random.choice(n_test, min(n_geo_samples, n_test), replace=False)
    X_geo = X_test[geo_idx]

    for m in m_values:
        print(f"\n  m={m} ...")
        torch.manual_seed(SEED)

        # モデルをロードするか、訓練する
        model_path = f'results/E1_{manifold_name}_m{m}.pth'
        model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])

        if os.path.exists(model_path):
            print(f"    Loading from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device)
        else:
            print(f"    Model not found, training from scratch for {epochs} epochs ...")
            model = model.to(device)
            train_ae(model, train_loader, epochs=epochs, lr=1e-3, device=device)

        # MSE
        mse = evaluate_mse(model, X_test)
        results['mse'].append(mse)
        print(f"    MSE: {mse:.6f}")

        # 潜在表現
        Z_test = get_latent_repr(model, X_test)
        Z_geo = get_latent_repr(model, X_geo)

        # Active Units
        au = active_units(Z_test)
        results['au'].append(au)
        print(f"    AU: {au}")

        # Trustworthiness
        trust = trustworthiness(X_test, Z_test, k=k)
        results['trustworthiness'].append(trust)
        print(f"    Trustworthiness: {trust:.4f}")

        # Continuity
        cont = continuity(X_test, Z_test, k=k)
        results['continuity'].append(cont)
        print(f"    Continuity: {cont:.4f}")

        # 測地線距離相関（サブサンプルで計算）
        geo_corr = geodesic_distance_correlation(X_geo, Z_geo, k=k)
        results['geodesic_corr'].append(geo_corr)
        print(f"    Geodesic corr: {geo_corr:.4f}")

    return results


def plot_structure_metrics(all_results: list):
    """構造保存メトリクスのプロット。"""
    n_manifolds = len(all_results)
    metrics = ['trustworthiness', 'continuity', 'geodesic_corr', 'mse']
    metric_labels = ['Trustworthiness', 'Continuity', 'Geodesic Corr (Spearman)', 'Test MSE']
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    fig, axes = plt.subplots(len(metrics), n_manifolds,
                             figsize=(7 * n_manifolds, 4 * len(metrics)))
    if n_manifolds == 1:
        axes = axes.reshape(-1, 1)

    for col, res in enumerate(all_results):
        m_values = res['m_values']
        d_true = res['d_true']
        name = res['manifold']

        for row, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            ax = axes[row, col]
            values = res[metric]
            ax.plot(m_values, values, 'o-', color=color, linewidth=2, markersize=8)
            ax.axvline(x=d_true, color='red', linestyle='--', linewidth=1.5,
                       label=f'd_true={d_true}')
            ax.set_xlabel('Bottleneck dim m', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{name}: {label}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(m_values)

    plt.tight_layout()
    plt.savefig('results/figures/E5_structure_metrics.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E5_structure_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E5_structure_metrics.pdf, .png")


def plot_combined_bar(all_results: list):
    """メトリクスの棒グラフ（各多様体・各m値の比較）。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(1, n_manifolds, figsize=(8 * n_manifolds, 6))
    if n_manifolds == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        m_values = res['m_values']
        d_true = res['d_true']
        name = res['manifold']
        n_m = len(m_values)

        x = np.arange(n_m)
        width = 0.25

        bars1 = ax.bar(x - width, res['trustworthiness'], width, label='Trustworthiness',
                       color='steelblue', alpha=0.85)
        bars2 = ax.bar(x, res['continuity'], width, label='Continuity',
                       color='coral', alpha=0.85)
        bars3 = ax.bar(x + width, res['geodesic_corr'], width, label='Geodesic Corr',
                       color='seagreen', alpha=0.85)

        # d_true のインデックスに縦線
        if d_true in m_values:
            d_idx = m_values.index(d_true)
            ax.axvline(x=d_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'd_true={d_true}')

        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Metric value', fontsize=12)
        ax.set_title(f'{name}: Structure Metrics vs m', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'm={m}' for m in m_values])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/E5_bar_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E5_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E5_bar_comparison.pdf, .png")


def main():
    """実験 E5 のメイン関数。"""
    n_train = 3000
    n_test = 1000
    # E1と同じm値 + 追加
    m_values = [1, 2, 3, 4]
    k = 10
    epochs = 200

    all_results = []

    # Swiss Roll
    print("Generating Swiss Roll data ...")
    X_train_sr, d_true_sr, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test_sr, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)

    res_sr = run_structure_experiment(
        'SwissRoll', X_train_sr, X_test_sr, d_true_sr,
        m_values, n_geo_samples=500, k=k, epochs=epochs
    )
    all_results.append(res_sr)

    # Torus
    print("\nGenerating Torus data ...")
    X_train_tor, d_true_tor, _ = generate_torus(n_train, R=3.0, r=1.0, noise=0.01, seed=SEED)
    X_test_tor, _, _ = generate_torus(n_test, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)

    res_tor = run_structure_experiment(
        'Torus', X_train_tor, X_test_tor, d_true_tor,
        m_values, n_geo_samples=500, k=k, epochs=epochs
    )
    all_results.append(res_tor)

    # 結果保存
    with open('results/tables/E5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/tables/E5_results.json")

    # プロット
    plot_structure_metrics(all_results)
    plot_combined_bar(all_results)

    # サマリー表示
    print("\n=== E5 Summary ===")
    for res in all_results:
        print(f"\n{res['manifold']} (d_true={res['d_true']}):")
        print(f"  {'m':>4} | {'Trust':>7} | {'Cont':>7} | {'GeoCor':>7} | {'AU':>4} | {'MSE':>10}")
        print(f"  {'-'*55}")
        for i, m in enumerate(res['m_values']):
            print(f"  {m:>4} | {res['trustworthiness'][i]:>7.4f} | "
                  f"{res['continuity'][i]:>7.4f} | "
                  f"{res['geodesic_corr'][i]:>7.4f} | "
                  f"{res['au'][i]:>4} | "
                  f"{res['mse'][i]:>10.6f}")


if __name__ == '__main__':
    main()
