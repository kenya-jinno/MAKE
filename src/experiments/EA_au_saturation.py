"""
実験 EA: AU飽和の確認（定理6の精密検証）
対応定理: 定理6（Active Units 飽和）
検証仮説: m >> d_ID のとき AU が d_ID 付近で頭打ちになる
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
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder, train_ae
from src.models.vae import VAE, train_vae
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


def get_latent_repr_ae(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """AEの潜在表現を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(X_tensor)
    return Z.cpu().numpy()


def get_latent_repr_vae(model: VAE, X: np.ndarray) -> np.ndarray:
    """VAEの潜在表現（mu）を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
    return mu.cpu().numpy()


def train_ae_with_l2_latent(
    model: AutoEncoder,
    train_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    latent_l2: float = 1e-3,
    device: str = 'cpu'
) -> list:
    """
    L2正則化付きAutoEncoderを訓練する。
    潜在表現に直接L2ペナルティを加えることで未使用次元の収縮を促す。

    Args:
        model: AutoEncoder モデル
        train_loader: 訓練データローダー
        epochs: エポック数
        lr: 学習率
        latent_l2: 潜在表現のL2正則化係数
        device: 計算デバイス
    Returns:
        losses: エポックごとの平均訓練損失のリスト
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device).float()

            optimizer.zero_grad()
            z = model.encode(x)
            recon = model.decode(z)
            rec_loss = criterion(recon, x)
            # 潜在表現のL2ペナルティ（小さい分散を持つ次元を0に収縮）
            l2_penalty = latent_l2 * (z ** 2).mean()
            loss = rec_loss + l2_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"    AE+L2 Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return losses


def run_au_saturation_experiment(
    manifold_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    d_true: int,
    m_values: list,
    epochs: int = 200
) -> dict:
    """
    AU飽和実験を実行する。

    Args:
        manifold_name: 多様体名
        X_train: 訓練データ
        X_test: テストデータ
        d_true: 真の内在次元
        m_values: ボトルネック次元のリスト
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
        'ae_au': [],
        'ae_mse': [],
        'ae_dim_variances': [],
        'ae_l2_au': [],
        'ae_l2_mse': [],
        'ae_l2_dim_variances': [],
        'vae_au': [],
        'vae_mse': [],
        'vae_dim_variances': [],
    }

    print(f"\n=== Experiment EA: {manifold_name} (d_true={d_true}) ===")

    for m in m_values:
        print(f"\n  m={m} ...")

        # --- Vanilla AE ---
        torch.manual_seed(SEED)
        ae_model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
        train_ae(ae_model, train_loader, epochs=epochs, lr=1e-3, device=device)

        ae_model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            Z_ae = ae_model.encode(X_test_tensor).cpu().numpy()
            recon_ae = ae_model(X_test_tensor)
            mse_ae = nn.MSELoss()(recon_ae, X_test_tensor).item()

        au_ae = active_units(Z_ae)
        var_ae = np.var(Z_ae, axis=0).tolist()
        results['ae_au'].append(au_ae)
        results['ae_mse'].append(mse_ae)
        results['ae_dim_variances'].append(var_ae)
        print(f"    AE:    MSE={mse_ae:.6f}, AU={au_ae}, dim_vars={[f'{v:.4f}' for v in var_ae]}")

        # --- AE with L2 latent regularization ---
        torch.manual_seed(SEED)
        ae_l2_model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
        train_ae_with_l2_latent(ae_l2_model, train_loader, epochs=epochs, lr=1e-3,
                                latent_l2=1e-3, device=device)

        ae_l2_model.eval()
        with torch.no_grad():
            Z_ae_l2 = ae_l2_model.encode(X_test_tensor).cpu().numpy()
            recon_ae_l2 = ae_l2_model(X_test_tensor)
            mse_ae_l2 = nn.MSELoss()(recon_ae_l2, X_test_tensor).item()

        au_ae_l2 = active_units(Z_ae_l2)
        var_ae_l2 = np.var(Z_ae_l2, axis=0).tolist()
        results['ae_l2_au'].append(au_ae_l2)
        results['ae_l2_mse'].append(mse_ae_l2)
        results['ae_l2_dim_variances'].append(var_ae_l2)
        print(f"    AE+L2: MSE={mse_ae_l2:.6f}, AU={au_ae_l2}, dim_vars={[f'{v:.4f}' for v in var_ae_l2]}")

        # --- VAE with beta=4.0 ---
        torch.manual_seed(SEED)
        vae_model = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])
        train_vae(vae_model, train_loader, epochs=epochs, lr=1e-3, beta=4.0, device=device)

        vae_model.eval()
        with torch.no_grad():
            mu, logvar = vae_model.encode(X_test_tensor)
            Z_vae = mu.cpu().numpy()
            recon_vae, _, _ = vae_model(X_test_tensor)
            mse_vae = nn.MSELoss()(recon_vae, X_test_tensor).item()

        au_vae = active_units(Z_vae)
        var_vae = np.var(Z_vae, axis=0).tolist()
        results['vae_au'].append(au_vae)
        results['vae_mse'].append(mse_vae)
        results['vae_dim_variances'].append(var_vae)
        print(f"    VAE:   MSE={mse_vae:.6f}, AU={au_vae}, dim_vars={[f'{v:.4f}' for v in var_vae]}")

    return results


def plot_au_saturation(all_results: list):
    """AU飽和実験のプロット。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(2, n_manifolds, figsize=(8 * n_manifolds, 12))
    if n_manifolds == 1:
        axes = axes.reshape(-1, 1)

    for col, res in enumerate(all_results):
        m_values = res['m_values']
        d_true = res['d_true']
        name = res['manifold']

        # AU vs m
        ax = axes[0, col]
        ax.plot(m_values, res['ae_au'], 'o-', color='steelblue', linewidth=2, label='AE')
        ax.plot(m_values, res['ae_l2_au'], 's--', color='darkorange', linewidth=2, label='AE+L2')
        ax.plot(m_values, res['vae_au'], '^-.', color='seagreen', linewidth=2, label='VAE (β=4)')
        ax.axhline(y=d_true, color='red', linestyle=':', linewidth=2, label=f'd_true={d_true}')
        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Active Units', fontsize=12)
        ax.set_title(f'{name}: AU vs m', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)
        ax.set_ylim(0, max(m_values) + 1)

        # Per-dim variance heatmap (VAE)
        ax = axes[1, col]
        # Build heatmap matrix: rows = m, cols = dim index
        max_m = max(m_values)
        heatmap = np.zeros((len(m_values), max_m))
        heatmap[:] = np.nan

        for i, (m, var_list) in enumerate(zip(m_values, res['vae_dim_variances'])):
            # Sort variances descending
            sorted_vars = sorted(var_list, reverse=True)
            for j, v in enumerate(sorted_vars):
                heatmap[i, j] = v

        # Normalize each row for visibility
        row_max = np.nanmax(heatmap, axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        heatmap_norm = heatmap / row_max

        im = ax.imshow(heatmap_norm, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f'm={m}' for m in m_values])
        ax.set_xlabel('Latent dim index (sorted by variance)', fontsize=11)
        ax.set_title(f'{name}: VAE per-dim variance heatmap\n(normalized, sorted desc)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Normalized variance')

    plt.tight_layout()
    plt.savefig('results/figures/EA_au_saturation.pdf', bbox_inches='tight')
    plt.savefig('results/figures/EA_au_saturation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EA_au_saturation.pdf, .png")


def plot_mse_comparison(all_results: list):
    """MSE比較プロット。"""
    n_manifolds = len(all_results)
    fig, axes = plt.subplots(1, n_manifolds, figsize=(7 * n_manifolds, 5))
    if n_manifolds == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        m_values = res['m_values']
        d_true = res['d_true']
        name = res['manifold']

        ax.plot(m_values, res['ae_mse'], 'o-', color='steelblue', linewidth=2, label='AE')
        ax.plot(m_values, res['ae_l2_mse'], 's--', color='darkorange', linewidth=2, label='AE+L2')
        ax.plot(m_values, res['vae_mse'], '^-.', color='seagreen', linewidth=2, label='VAE (β=4)')
        ax.axvline(x=d_true, color='red', linestyle=':', linewidth=2, label=f'd_true={d_true}')
        ax.set_xlabel('Bottleneck dim m', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title(f'{name}: MSE vs m', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_values)

    plt.tight_layout()
    plt.savefig('results/figures/EA_mse_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/EA_mse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EA_mse_comparison.pdf, .png")


def main():
    """実験 EA のメイン関数。"""
    n_train = 3000
    n_test = 1000
    epochs = 200
    m_values = [1, 2, 3, 4, 6, 8, 10, 12, 16]

    all_results = []

    # Swiss Roll (d_true=2)
    print("Generating Swiss Roll data ...")
    X_train_sr, d_true_sr, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test_sr, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)

    res_sr = run_au_saturation_experiment(
        'SwissRoll', X_train_sr, X_test_sr, d_true_sr, m_values, epochs=epochs
    )
    all_results.append(res_sr)

    # Torus (d_true=2)
    print("\nGenerating Torus data ...")
    X_train_tor, d_true_tor, _ = generate_torus(n_train, R=3.0, r=1.0, noise=0.01, seed=SEED)
    X_test_tor, _, _ = generate_torus(n_test, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)

    res_tor = run_au_saturation_experiment(
        'Torus', X_train_tor, X_test_tor, d_true_tor, m_values, epochs=epochs
    )
    all_results.append(res_tor)

    # 結果保存
    with open('results/tables/EA_au_saturation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/tables/EA_au_saturation.json")

    # プロット
    plot_au_saturation(all_results)
    plot_mse_comparison(all_results)

    # サマリー表示
    print("\n=== EA Summary ===")
    for res in all_results:
        print(f"\n{res['manifold']} (d_true={res['d_true']}):")
        print(f"  {'m':>4} | {'AE_AU':>6} | {'AEL2_AU':>8} | {'VAE_AU':>7} | {'AE_MSE':>10} | {'VAE_MSE':>10}")
        print(f"  {'-'*55}")
        for i, m in enumerate(res['m_values']):
            print(f"  {m:>4} | {res['ae_au'][i]:>6} | {res['ae_l2_au'][i]:>8} | "
                  f"{res['vae_au'][i]:>7} | {res['ae_mse'][i]:>10.6f} | {res['vae_mse'][i]:>10.6f}")


if __name__ == '__main__':
    main()
