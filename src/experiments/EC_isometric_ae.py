"""
実験 EC: 等長AE（デコーダヤコビアン正則化による等長性改善）
対応定理: 定理3（プルバック計量）
検証仮説: デコーダJ_g^T J_g ≈ I を強制するとCAEより条件数が低くなる
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

from src.data.synthetic import generate_swiss_roll
from src.models.ae import AutoEncoder, train_ae
from src.models.cae import ContractiveAE, train_cae
from src.models.dae import DenoisingAE, train_dae
from src.models.isometric_ae import IsometricAE, train_isometric_ae
from src.metrics.structure import trustworthiness, active_units
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

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


def make_dataloader(X: np.ndarray, batch_size: int = 64) -> DataLoader:
    """numpy 配列から DataLoader を作成する。"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate_mse(model: nn.Module, X_test: np.ndarray) -> float:
    """テストデータの MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        if isinstance(model, IsometricAE):
            recon, _ = model(X_tensor)
        else:
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


def compute_condition_number(model: nn.Module, Z: np.ndarray, n_eval: int = 50) -> float:
    """
    デコーダのヤコビアン条件数の平均を計算する。

    Args:
        model: AEモデル（decoderを持つ）
        Z: 潜在表現 (n_samples, latent_dim)
        n_eval: 評価点数
    Returns:
        mean_kappa: 平均条件数
    """
    n_eval = min(n_eval, len(Z))
    idx = np.random.choice(len(Z), n_eval, replace=False)
    condition_numbers = []

    model.eval()
    for i in idx:
        z_tensor = torch.tensor(Z[i], dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(model.decoder, z_tensor)
        J_np = J.cpu().numpy()
        _, _, kappa = analyze_singular_values(J_np, threshold_ratio=0.01)
        condition_numbers.append(kappa)

    return float(np.mean(condition_numbers))


def compute_nsr(model: nn.Module, X: np.ndarray, Z: np.ndarray,
                sigma_noise: float = 0.05) -> float:
    """
    ノイズ選択性比（NSR）: 法線方向ノイズが接空間方向ノイズより
    どれだけ抑制されるかを測定する（近似）。

    Args:
        model: AEモデル
        X: 元データ (n_samples, input_dim)
        Z: 潜在表現 (n_samples, latent_dim)
        sigma_noise: ノイズの標準偏差
    Returns:
        nsr: ノイズ選択性比（大きいほど法線ノイズを抑制している）
    """
    model.eval()
    n_eval = min(100, len(X))
    idx = np.random.choice(len(X), n_eval, replace=False)

    nsr_vals = []
    rng = np.random.RandomState(SEED)

    for i in idx:
        x_orig = X[i]
        z_orig = Z[i]

        # ランダムノイズを加えた入力の潜在表現変化
        noise = rng.randn(*x_orig.shape) * sigma_noise
        x_noisy = x_orig + noise

        X_orig_t = torch.tensor(x_orig, dtype=torch.float32).unsqueeze(0).to(device)
        X_noisy_t = torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            z_noisy = model.encode(X_noisy_t).squeeze(0).cpu().numpy()

        # 潜在空間での変化量 vs 入力空間での変化量の比
        delta_z = np.linalg.norm(z_noisy - z_orig)
        delta_x = np.linalg.norm(noise)

        if delta_x > 1e-10:
            nsr_vals.append(delta_z / delta_x)

    return float(np.mean(nsr_vals)) if nsr_vals else 0.0


def run_model_comparison(X_train: np.ndarray, X_test: np.ndarray,
                         latent_dim: int, epochs: int = 200) -> list:
    """
    AE, CAE, IsometricAE, DAE を比較する。

    Returns:
        results: 各モデルの結果リスト
    """
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64)

    models_config = [
        {
            'name': 'AE',
            'model': AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[128, 64]),
            'train_fn': lambda m, dl: train_ae(m, dl, epochs=epochs, lr=1e-3, device=device),
        },
        {
            'name': 'CAE',
            'model': ContractiveAE(input_dim=input_dim, latent_dim=latent_dim,
                                   hidden_dims=[128, 64], lambda_c=1e-3),
            'train_fn': lambda m, dl: train_cae(m, dl, epochs=epochs, lr=1e-3, device=device),
        },
        {
            'name': 'IsometricAE',
            'model': IsometricAE(input_dim=input_dim, latent_dim=latent_dim,
                                 hidden_dims=[128, 64], lambda_iso=1e-2),
            'train_fn': lambda m, dl: train_isometric_ae(
                m, dl, epochs=epochs, lr=1e-3, device=device, iso_batch_size=4),
        },
        {
            'name': 'DAE',
            'model': DenoisingAE(input_dim=input_dim, latent_dim=latent_dim,
                                 hidden_dims=[128, 64], noise_std=0.1),
            'train_fn': lambda m, dl: train_dae(m, dl, epochs=epochs, lr=1e-3, device=device),
        },
    ]

    results = []

    for cfg in models_config:
        name = cfg['name']
        model = cfg['model']
        print(f"\n  Training {name} ...")
        cfg['train_fn'](model, train_loader)

        # 評価
        mse = evaluate_mse(model, X_test)
        Z_test = get_latent_repr(model, X_test)

        # 条件数
        kappa = compute_condition_number(model, Z_test, n_eval=30)

        # NSR
        nsr = compute_nsr(model, X_test, Z_test, sigma_noise=0.05)

        # Trustworthiness
        n_trust = min(300, len(X_test))
        idx = np.random.choice(len(X_test), n_trust, replace=False)
        tw = trustworthiness(X_test[idx], Z_test[idx], k=10)

        # Active Units
        au = active_units(Z_test)

        res = {
            'model': name,
            'mse': float(mse),
            'condition_number': float(kappa),
            'nsr': float(nsr),
            'trustworthiness': float(tw),
            'active_units': int(au),
        }
        results.append(res)

        print(f"    {name}: MSE={mse:.6f}, kappa={kappa:.2f}, "
              f"NSR={nsr:.4f}, Trust={tw:.4f}, AU={au}")

    return results


def plot_results(results: list):
    """モデル比較の結果をプロットする。"""
    model_names = [r['model'] for r in results]
    metrics = ['mse', 'condition_number', 'nsr', 'trustworthiness']
    metric_labels = ['MSE', 'Condition Number κ', 'NSR', 'Trustworthiness']
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [r[metric] for r in results]
        bars = ax.bar(model_names, values, color=colors[:len(results)], alpha=0.8, edgecolor='black')
        ax.set_title(label, fontsize=13)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('EC: AE Variant Comparison on Swiss Roll (m=2, latent_dim=2)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/EC_isometric_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/EC_isometric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/EC_isometric_comparison.pdf, .png")


def main():
    """実験 EC のメイン関数。"""
    n_train = 3000
    n_test = 500
    epochs = 200
    latent_dim = 2  # Swiss Roll の d_true = 2

    print("Generating Swiss Roll data ...")
    X_train, _, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)

    print(f"\nInput dim: {X_train.shape[1]}, Latent dim: {latent_dim}")
    print(f"Train: {n_train}, Test: {n_test}, Epochs: {epochs}")

    print("\n=== EC: Isometric AE Comparison ===")
    results = run_model_comparison(X_train, X_test, latent_dim=latent_dim, epochs=epochs)

    # 結果保存
    save_data = {
        'dataset': 'SwissRoll',
        'latent_dim': latent_dim,
        'epochs': epochs,
        'results': results,
    }
    with open('results/tables/EC_isometric.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/EC_isometric.json")

    # プロット
    plot_results(results)

    # サマリー
    print("\n=== EC Summary ===")
    print(f"{'Model':<15} {'MSE':>10} {'kappa':>10} {'NSR':>10} {'Trust':>10} {'AU':>5}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<15} {r['mse']:>10.6f} {r['condition_number']:>10.2f} "
              f"{r['nsr']:>10.4f} {r['trustworthiness']:>10.4f} {r['active_units']:>5}")


if __name__ == '__main__':
    main()
