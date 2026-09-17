"""
実験 ED: モデル比較（AE/CAE/DAE/VAE の多様体抽出能力比較）
対応定理: 定理3（プルバック計量・等長性）, 定理4（ノイズ選択性）
検証仮説: CAEは条件数が最小（等長性が高い）、DAEはNSRが最大
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

from src.data.synthetic import generate_swiss_roll
from src.models.ae import AutoEncoder, train_ae
from src.models.cae import ContractiveAE, train_cae
from src.models.dae import DenoisingAE, train_dae
from src.models.vae import VAE, train_vae
from src.metrics.structure import active_units, trustworthiness
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

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


def evaluate_mse(model: nn.Module, X: np.ndarray, is_vae: bool = False) -> float:
    """テストデータの MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if is_vae:
            recon, _, _ = model(X_tensor)
        else:
            recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return mse


def get_latent_repr(model: nn.Module, X: np.ndarray, is_vae: bool = False) -> np.ndarray:
    """潜在表現を取得する（VAEはmuを返す）。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if is_vae:
            z, _ = model.encode(X_tensor)
        else:
            z = model.encode(X_tensor)
    return z.cpu().numpy()


def compute_mean_condition_number(
    model: nn.Module,
    X_eval: np.ndarray,
    n_points: int = 50,
    threshold_ratio: float = 0.1,
    is_vae: bool = False
) -> float:
    """
    デコーダのプルバック計量の平均条件数を計算する。

    Args:
        model: AutoEncoderモデル
        X_eval: 評価データ
        n_points: 評価点数
        threshold_ratio: 有意特異値の閾値
        is_vae: VAEフラグ（エンコーダインタフェースの違い）
    Returns:
        mean_kappa: 平均条件数
    """
    model.eval()
    n = len(X_eval)
    idx = np.random.choice(n, min(n_points, n), replace=False)
    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    condition_numbers = []
    for i in idx:
        with torch.no_grad():
            if is_vae:
                mu, _ = model.encode(X_tensor[i:i+1])
                z = mu.squeeze(0)
            else:
                z = model.encode(X_tensor[i:i+1]).squeeze(0)

        try:
            J = compute_decoder_jacobian(model.decoder, z.to(device))
            J_np = J.cpu().numpy()
            _, _, kappa = analyze_singular_values(J_np, threshold_ratio=threshold_ratio)
            condition_numbers.append(kappa)
        except Exception:
            continue

    return float(np.mean(condition_numbers)) if condition_numbers else float('nan')


def compute_nsr(
    model: nn.Module,
    X: np.ndarray,
    T_basis: np.ndarray,
    sigma: float = 0.1,
    n_points: int = 200,
    n_trials: int = 10,
    is_vae: bool = False
) -> float:
    """
    Normal-to-Tangent Space Ratio (NSR) を計算する。
    NSR = E[||z_noisy_normal - z_clean||] / E[||z_noisy_tangent - z_clean||]
    NSR > 1 → 法線ノイズが接線ノイズより潜在空間で大きく見える（選択的抑制あり）

    Args:
        model: モデル
        X: データ (n_samples, N)
        T_basis: 接空間基底 (n_samples, N, d)
        sigma: ノイズの標準偏差
        n_points: 評価点数
        n_trials: 各点でのノイズ試行数
        is_vae: VAEフラグ
    Returns:
        nsr: Normal-Tangent Space Ratio
    """
    model.eval()
    rng = np.random.RandomState(SEED)

    n = len(X)
    idx = rng.choice(n, min(n_points, n), replace=False)

    normal_diffs = []
    tangent_diffs = []

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    for i in idx:
        xi = X[i]  # (N,)
        Ti = T_basis[i]  # (N, d)
        N_dim = xi.shape[0]

        # 投影行列
        P_tangent = Ti @ Ti.T  # (N, N)
        P_normal = np.eye(N_dim) - P_tangent

        with torch.no_grad():
            if is_vae:
                mu, _ = model.encode(X_tensor[i:i+1])
                z_clean = mu.squeeze(0).cpu().numpy()
            else:
                z_clean = model.encode(X_tensor[i:i+1]).squeeze(0).cpu().numpy()

        for _ in range(n_trials):
            noise = rng.randn(N_dim) * sigma

            # 法線ノイズのみ
            xi_normal = xi + P_normal @ noise
            xi_normal_t = torch.tensor(xi_normal, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_vae:
                    mu_n, _ = model.encode(xi_normal_t)
                    z_normal = mu_n.squeeze(0).cpu().numpy()
                else:
                    z_normal = model.encode(xi_normal_t).squeeze(0).cpu().numpy()
            normal_diffs.append(np.linalg.norm(z_normal - z_clean))

            # 接空間ノイズのみ
            xi_tangent = xi + P_tangent @ noise
            xi_tangent_t = torch.tensor(xi_tangent, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_vae:
                    mu_t, _ = model.encode(xi_tangent_t)
                    z_tangent = mu_t.squeeze(0).cpu().numpy()
                else:
                    z_tangent = model.encode(xi_tangent_t).squeeze(0).cpu().numpy()
            tangent_diffs.append(np.linalg.norm(z_tangent - z_clean))

    mean_normal = np.mean(normal_diffs) if normal_diffs else float('nan')
    mean_tangent = np.mean(tangent_diffs) if tangent_diffs else float('nan')

    if mean_tangent < 1e-10:
        return float('nan')
    return float(mean_normal / mean_tangent)


def main():
    """実験 ED のメイン関数。"""
    n_train = 3000
    n_test = 1000
    epochs = 200
    d_true = 2
    m = 2  # m = d_true
    k = 10
    noise_sigma = 0.1

    print("Generating Swiss Roll data ...")
    X_train, _, T_train = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test, _, T_test = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)
    input_dim = X_train.shape[1]

    train_loader = make_dataloader(X_train, batch_size=64, train=True)

    print(f"\n=== Experiment ED: Swiss Roll (d_true={d_true}, m={m}) ===")

    results = {}

    # ===================== AE =====================
    print("\n--- Training Vanilla AE ---")
    torch.manual_seed(SEED)
    ae_model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
    train_ae(ae_model, train_loader, epochs=epochs, lr=1e-3, device=device)

    mse_ae = evaluate_mse(ae_model, X_test)
    Z_ae = get_latent_repr(ae_model, X_test)
    au_ae = active_units(Z_ae)
    trust_ae = trustworthiness(X_test, Z_ae, k=k)
    kappa_ae = compute_mean_condition_number(ae_model, X_test, n_points=50)
    nsr_ae = compute_nsr(ae_model, X_test, T_test, sigma=noise_sigma, n_points=200)

    results['AE'] = {
        'mse': mse_ae,
        'condition_number': kappa_ae,
        'nsr': nsr_ae,
        'trustworthiness': trust_ae,
        'au': au_ae
    }
    print(f"  MSE={mse_ae:.6f}, kappa={kappa_ae:.3f}, NSR={nsr_ae:.4f}, "
          f"Trust={trust_ae:.4f}, AU={au_ae}")

    # ===================== CAE =====================
    print("\n--- Training Contractive AE (lambda=1e-3) ---")
    torch.manual_seed(SEED)
    cae_model = ContractiveAE(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128],
                               lambda_c=1e-3)
    train_cae(cae_model, train_loader, epochs=epochs, lr=1e-3, device=device)

    mse_cae = evaluate_mse(cae_model, X_test)
    Z_cae = get_latent_repr(cae_model, X_test)
    au_cae = active_units(Z_cae)
    trust_cae = trustworthiness(X_test, Z_cae, k=k)
    kappa_cae = compute_mean_condition_number(cae_model, X_test, n_points=50)
    nsr_cae = compute_nsr(cae_model, X_test, T_test, sigma=noise_sigma, n_points=200)

    results['CAE'] = {
        'mse': mse_cae,
        'condition_number': kappa_cae,
        'nsr': nsr_cae,
        'trustworthiness': trust_cae,
        'au': au_cae
    }
    print(f"  MSE={mse_cae:.6f}, kappa={kappa_cae:.3f}, NSR={nsr_cae:.4f}, "
          f"Trust={trust_cae:.4f}, AU={au_cae}")

    # ===================== DAE =====================
    print("\n--- Training Denoising AE (noise_std=0.1) ---")
    torch.manual_seed(SEED)
    dae_model = DenoisingAE(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128],
                             noise_std=0.1)
    train_dae(dae_model, train_loader, epochs=epochs, lr=1e-3, device=device)

    mse_dae = evaluate_mse(dae_model, X_test)
    Z_dae = get_latent_repr(dae_model, X_test)
    au_dae = active_units(Z_dae)
    trust_dae = trustworthiness(X_test, Z_dae, k=k)
    kappa_dae = compute_mean_condition_number(dae_model, X_test, n_points=50)
    nsr_dae = compute_nsr(dae_model, X_test, T_test, sigma=noise_sigma, n_points=200)

    results['DAE'] = {
        'mse': mse_dae,
        'condition_number': kappa_dae,
        'nsr': nsr_dae,
        'trustworthiness': trust_dae,
        'au': au_dae
    }
    print(f"  MSE={mse_dae:.6f}, kappa={kappa_dae:.3f}, NSR={nsr_dae:.4f}, "
          f"Trust={trust_dae:.4f}, AU={au_dae}")

    # ===================== VAE =====================
    print("\n--- Training VAE (beta=1.0) ---")
    torch.manual_seed(SEED)
    vae_model = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
    train_vae(vae_model, train_loader, epochs=epochs, lr=1e-3, beta=1.0, device=device)

    mse_vae = evaluate_mse(vae_model, X_test, is_vae=True)
    Z_vae = get_latent_repr(vae_model, X_test, is_vae=True)
    au_vae = active_units(Z_vae)
    trust_vae = trustworthiness(X_test, Z_vae, k=k)
    kappa_vae = compute_mean_condition_number(vae_model, X_test, n_points=50, is_vae=True)
    nsr_vae = compute_nsr(vae_model, X_test, T_test, sigma=noise_sigma, n_points=200,
                          is_vae=True)

    results['VAE'] = {
        'mse': mse_vae,
        'condition_number': kappa_vae,
        'nsr': nsr_vae,
        'trustworthiness': trust_vae,
        'au': au_vae
    }
    print(f"  MSE={mse_vae:.6f}, kappa={kappa_vae:.3f}, NSR={nsr_vae:.4f}, "
          f"Trust={trust_vae:.4f}, AU={au_vae}")

    # 結果保存
    save_data = {
        'manifold': 'SwissRoll',
        'd_true': d_true,
        'm': m,
        'epochs': epochs,
        'results': results
    }

    with open('results/tables/ED_model_comparison.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/ED_model_comparison.json")

    # プロット
    plot_model_comparison(results)

    # サマリー表示
    print("\n=== ED Summary: Swiss Roll (d_true=2, m=2) ===")
    print(f"  {'Model':>6} | {'MSE':>10} | {'kappa':>8} | {'NSR':>7} | {'Trust':>7} | {'AU':>3}")
    print(f"  {'-'*55}")
    for model_name, r in results.items():
        print(f"  {model_name:>6} | {r['mse']:>10.6f} | {r['condition_number']:>8.3f} | "
              f"{r['nsr']:>7.4f} | {r['trustworthiness']:>7.4f} | {r['au']:>3}")

    # 仮説の検証
    print("\n  Hypothesis check:")
    kappas = {k: v['condition_number'] for k, v in results.items()}
    nsrs = {k: v['nsr'] for k, v in results.items()}

    min_kappa_model = min(kappas, key=kappas.get)
    max_nsr_model = max(nsrs, key=nsrs.get)

    print(f"  Lowest condition number (most isometric): {min_kappa_model} (kappa={kappas[min_kappa_model]:.3f})")
    print(f"  Highest NSR (best noise selectivity): {max_nsr_model} (NSR={nsrs[max_nsr_model]:.4f})")

    if min_kappa_model == 'CAE':
        print("  --> CONFIRMED: CAE has best isometry (kappa minimized)")
    else:
        print(f"  --> NOTE: Expected CAE, got {min_kappa_model} as most isometric")

    if max_nsr_model == 'DAE':
        print("  --> CONFIRMED: DAE has best noise selectivity (NSR maximized)")
    else:
        print(f"  --> NOTE: Expected DAE, got {max_nsr_model} as best noise selector")


def plot_model_comparison(results: dict):
    """モデル比較のグループ棒グラフ。"""
    models = list(results.keys())
    metrics = ['mse', 'condition_number', 'nsr', 'trustworthiness']
    metric_labels = ['MSE (lower=better)', 'Condition Number κ (lower=better)',
                     'NSR (higher=better)', 'Trustworthiness (higher=better)']
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    # 正規化（各メトリクスを[0,1]にスケール）
    vals = {m: [results[model][m] for model in models] for m in metrics}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ED: Model Comparison — Swiss Roll (d_true=2, m=2)', fontsize=14)

    # 左: 生の値（棒グラフ）
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.2

    for j, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[model][metric] for model in models]
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Raw Metric Values', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 右: 正規化した値（各メトリクスの相対比較）
    ax = axes[1]
    # Invert MSE and kappa for "higher is better" normalization
    normalized = {}
    for metric in metrics:
        raw = np.array([results[model][metric] for model in models], dtype=float)
        finite_mask = np.isfinite(raw)
        if not finite_mask.any():
            normalized[metric] = raw
            continue
        rng = raw[finite_mask].max() - raw[finite_mask].min()
        if rng < 1e-10:
            normalized[metric] = np.ones_like(raw)
        else:
            normalized[metric] = (raw - raw[finite_mask].min()) / rng

    # For MSE and kappa: lower is better, so invert
    for metric in ['mse', 'condition_number']:
        normalized[metric] = 1.0 - normalized[metric]

    for j, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [normalized[metric][i] for i in range(len(models))]
        offset = (j - 1.5) * width
        ax.bar(x + offset, values, width, label=label.split('(')[0].strip(),
               color=color, alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Normalized Score (higher=better)', fontsize=12)
    ax.set_title('Normalized Metric Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/ED_model_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/ED_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/ED_model_comparison.pdf, .png")


if __name__ == '__main__':
    main()
