"""
実験 E1: 合成多様体での基礎検証
対応定理: 定理1（m >= d 下界）, 定理6（AU飽和）
検証仮説: m = d_true付近でMSEの肘点が現れる
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder, train_ae
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate, pca_estimate
from src.metrics.structure import active_units

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


def evaluate_mse(model: nn.Module, X_test: np.ndarray) -> float:
    """テストデータの MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return mse


def get_latent_repr(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """潜在表現を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(X_tensor)
    return Z.cpu().numpy()


def run_experiment(manifold_name: str, X_train: np.ndarray, X_test: np.ndarray,
                   d_true: int, epochs: int = 100) -> dict:
    """
    指定した多様体に対してボトルネック次元スイープを実行する。

    Args:
        manifold_name: 多様体名（ログ用）
        X_train: 訓練データ
        X_test: テストデータ
        d_true: 真の内在次元
        epochs: 訓練エポック数
    Returns:
        results: 実験結果の辞書
    """
    input_dim = X_train.shape[1]
    m_values = list(range(1, 2 * d_true + 1))  # m = 1, 2, ..., 2*d_true
    train_loader = make_dataloader(X_train, batch_size=64, train=True)

    results = {
        'manifold': manifold_name,
        'd_true': d_true,
        'm_values': m_values,
        'mse': [],
        'au': [],
        'id_twonn': [],
        'id_mle': [],
        'id_pca': [],
        'train_losses': {}
    }

    print(f"\n=== Experiment E1: {manifold_name} (d_true={d_true}) ===")

    for m in m_values:
        print(f"\n  m={m} ...")
        model = AutoEncoder(
            input_dim=input_dim,
            latent_dim=m,
            hidden_dims=[256, 128]
        )
        train_losses = train_ae(model, train_loader, epochs=epochs, lr=1e-3, device=device)
        results['train_losses'][str(m)] = train_losses

        # MSE 計算
        mse = evaluate_mse(model, X_test)
        results['mse'].append(mse)
        print(f"    MSE: {mse:.6f}")

        # 潜在表現取得
        Z_test = get_latent_repr(model, X_test)

        # Active Units
        au = active_units(Z_test)
        results['au'].append(au)
        print(f"    AU: {au}")

        # 内在次元推定
        id_twonn = twonn_estimate(Z_test)
        id_mle = mle_estimate(Z_test, k=10)
        id_pca = pca_estimate(Z_test, threshold=0.95)
        results['id_twonn'].append(id_twonn)
        results['id_mle'].append(id_mle)
        results['id_pca'].append(id_pca)
        print(f"    ID (TwoNN/MLE/PCA): {id_twonn:.2f} / {id_mle:.2f} / {id_pca}")

        # モデルを保存
        save_path = f'results/E1_{manifold_name}_m{m}.pth'
        torch.save(model.state_dict(), save_path)

    return results


def plot_results(all_results: list):
    """実験結果をプロットする。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(2, n_manifolds, figsize=(6 * n_manifolds, 10))
    if n_manifolds == 1:
        axes = axes.reshape(-1, 1)

    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    for col, res in enumerate(all_results):
        m_values = res['m_values']
        d_true = res['d_true']
        name = res['manifold']

        # MSE vs m
        ax = axes[0, col]
        ax.plot(m_values, res['mse'], 'o-', color=colors[0], linewidth=2, label='MSE')
        ax.axvline(x=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title(f'{name}: MSE vs m', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)

        # AU, ID vs m
        ax = axes[1, col]
        ax.plot(m_values, res['au'], 's-', color=colors[1], linewidth=2, label='Active Units')
        ax.plot(m_values, res['id_twonn'], '^-', color=colors[2], linewidth=2, label='ID (TwoNN)')
        ax.plot(m_values, res['id_mle'], 'D-', color=colors[3], linewidth=2, label='ID (MLE)')
        ax.axvline(x=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{name}: AU & ID vs m', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)

    plt.tight_layout()
    plt.savefig('results/figures/E1_mse_vs_m.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E1_mse_vs_m.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/E1_mse_vs_m.pdf, .png")


def main():
    """実験 E1 のメイン関数。"""
    n_train = 3000
    n_test = 1000
    epochs = 100

    all_results = []

    # Swiss Roll
    print("Generating Swiss Roll data ...")
    X_train_sr, d_true_sr, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test_sr, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)

    res_sr = run_experiment('SwissRoll', X_train_sr, X_test_sr, d_true_sr, epochs=epochs)
    all_results.append(res_sr)

    # Torus
    print("\nGenerating Torus data ...")
    X_train_tor, d_true_tor, _ = generate_torus(n_train, R=3.0, r=1.0, noise=0.01, seed=SEED)
    X_test_tor, _, _ = generate_torus(n_test, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)

    res_tor = run_experiment('Torus', X_train_tor, X_test_tor, d_true_tor, epochs=epochs)
    all_results.append(res_tor)

    # 結果保存
    save_data = []
    for res in all_results:
        save_res = {k: v for k, v in res.items() if k != 'train_losses'}
        save_data.append(save_res)

    with open('results/tables/E1_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/E1_results.json")

    # プロット
    plot_results(all_results)

    # サマリー表示
    print("\n=== E1 Summary ===")
    for res in all_results:
        print(f"\n{res['manifold']} (d_true={res['d_true']}):")
        for i, m in enumerate(res['m_values']):
            print(f"  m={m}: MSE={res['mse'][i]:.6f}, AU={res['au'][i]}, "
                  f"ID_TwoNN={res['id_twonn'][i]:.2f}, ID_MLE={res['id_mle'][i]:.2f}")


if __name__ == '__main__':
    main()
