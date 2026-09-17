"""
実験 E5_extended: 測地線距離相関の k-NN パラメータ感度分析
対応定理: 定理7（近傍構造保存）
検証仮説: k値を変えても測地線相関の順序関係（m=d_trueで最高）は安定する
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

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder
from src.metrics.structure import geodesic_distance_correlation, trustworthiness

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


def get_latent_repr(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """潜在表現を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(X_tensor)
    return Z.cpu().numpy()


def load_model(manifold_name: str, m: int, input_dim: int) -> AutoEncoder:
    """E1で学習済みのモデルをロードする。"""
    model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
    save_path = f'results/E1_{manifold_name}_m{m}.pth'
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model not found: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def run_k_sensitivity(manifold_name: str, X: np.ndarray, input_dim: int,
                      m_values: list, k_values_geo: list, k_values_trust: list) -> dict:
    """
    k値の感度分析を実行する。

    Args:
        manifold_name: 多様体名
        X: テストデータ
        input_dim: 入力次元
        m_values: ボトルネック次元のリスト
        k_values_geo: 測地線計算に使うk値のリスト
        k_values_trust: Trustworthinessに使うk値のリスト
    Returns:
        results: 実験結果の辞書
    """
    print(f"\n=== E5_extended: {manifold_name} ===")

    # サブサンプル（計算コスト削減）
    n_eval = min(500, len(X))
    idx = np.random.choice(len(X), n_eval, replace=False)
    X_sub = X[idx]

    results = {
        'manifold': manifold_name,
        'm_values': m_values,
        'k_values_geo': k_values_geo,
        'k_values_trust': k_values_trust,
        'geo_corr': {},   # geo_corr[m][k] = correlation
        'trust': {},       # trust[m][k] = trustworthiness
    }

    for m in m_values:
        print(f"\n  m={m}:")
        try:
            model = load_model(manifold_name, m, input_dim)
        except FileNotFoundError as e:
            print(f"  Skipping m={m}: {e}")
            continue

        Z_sub = get_latent_repr(model, X_sub)

        results['geo_corr'][str(m)] = {}
        results['trust'][str(m)] = {}

        for k in k_values_geo:
            k_eff = min(k, n_eval - 1)
            corr = geodesic_distance_correlation(X_sub, Z_sub, k=k_eff)
            results['geo_corr'][str(m)][str(k)] = float(corr)
            print(f"    k={k}: geo_corr={corr:.4f}")

        for k in k_values_trust:
            k_eff = min(k, n_eval - 1)
            tw = trustworthiness(X_sub, Z_sub, k=k_eff)
            results['trust'][str(m)][str(k)] = float(tw)
            print(f"    k={k}: trust={tw:.4f}")

    return results


def plot_results(all_results: list):
    """k感度分析の結果をプロットする。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(2, n_manifolds, figsize=(7 * n_manifolds, 12))
    if n_manifolds == 1:
        axes = axes.reshape(-1, 1)

    colors = ['steelblue', 'coral', 'seagreen', 'orchid']
    markers = ['o', 's', '^', 'D']

    for col, res in enumerate(all_results):
        manifold_name = res['manifold']
        m_values = res['m_values']
        k_values_geo = res['k_values_geo']
        k_values_trust = res['k_values_trust']

        # 測地線相関 vs k（各m）
        ax = axes[0, col]
        for i, m in enumerate(m_values):
            if str(m) not in res['geo_corr']:
                continue
            corrs = [res['geo_corr'][str(m)].get(str(k), np.nan) for k in k_values_geo]
            ax.plot(k_values_geo, corrs, f'{markers[i]}-', color=colors[i],
                    linewidth=2, label=f'm={m}', markersize=8)
        ax.set_xlabel('k (neighbors)', fontsize=12)
        ax.set_ylabel('Geodesic Correlation (Spearman)', fontsize=11)
        ax.set_title(f'{manifold_name}: Geodesic Correlation vs k', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Trustworthiness vs k（各m）
        ax = axes[1, col]
        for i, m in enumerate(m_values):
            if str(m) not in res['trust']:
                continue
            trusts = [res['trust'][str(m)].get(str(k), np.nan) for k in k_values_trust]
            ax.plot(k_values_trust, trusts, f'{markers[i]}-', color=colors[i],
                    linewidth=2, label=f'm={m}', markersize=8)
        ax.set_xlabel('k (neighbors)', fontsize=12)
        ax.set_ylabel('Trustworthiness', fontsize=12)
        ax.set_title(f'{manifold_name}: Trustworthiness vs k', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/figures/E5_extended_k_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E5_extended_k_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/E5_extended_k_sensitivity.pdf, .png")


def main():
    """実験 E5_extended のメイン関数。"""
    k_values_geo = [5, 10, 15, 20, 30]
    k_values_trust = [5, 10, 15, 20]

    all_results = []

    # Swiss Roll (d_true=2, ambient=3, m_values=1..4)
    print("Generating Swiss Roll data ...")
    X_sr, d_sr, _ = generate_torus(1000, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)
    X_sr, d_sr, _ = generate_swiss_roll(1000, noise=0.01, seed=SEED + 1)
    input_dim_sr = X_sr.shape[1]
    m_values_sr = [1, 2, 3, 4]

    res_sr = run_k_sensitivity(
        'SwissRoll', X_sr, input_dim_sr, m_values_sr, k_values_geo, k_values_trust
    )
    all_results.append(res_sr)

    # Torus (d_true=2, ambient=3, m_values=1..4)
    print("\nGenerating Torus data ...")
    X_tor, d_tor, _ = generate_torus(1000, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)
    input_dim_tor = X_tor.shape[1]
    m_values_tor = [1, 2, 3, 4]

    res_tor = run_k_sensitivity(
        'Torus', X_tor, input_dim_tor, m_values_tor, k_values_geo, k_values_trust
    )
    all_results.append(res_tor)

    # 結果保存
    with open('results/tables/E5_extended.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/tables/E5_extended.json")

    # プロット
    plot_results(all_results)

    # サマリー
    print("\n=== E5_extended Summary ===")
    for res in all_results:
        print(f"\n{res['manifold']}:")
        for m in res['m_values']:
            if str(m) not in res['geo_corr']:
                continue
            corrs = res['geo_corr'][str(m)]
            trusts = res['trust'][str(m)]
            geo_str = ', '.join(f"{corrs.get(str(k), float('nan')):.3f}" for k in res['k_values_geo'])
            trust_str = ', '.join(f"{trusts.get(str(k), float('nan')):.3f}" for k in res['k_values_trust'])
            print(f"  m={m}: geo_corr@k=[{geo_str}]")
            print(f"       trust@k=[{trust_str}]")


if __name__ == '__main__':
    main()
