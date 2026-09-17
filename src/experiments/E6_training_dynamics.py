"""
実験 E6: 学習ダイナミクス
対応定理: 定理2（接空間形成の過程）
検証仮説: 大域的構造（Trustworthiness）が局所的構造（TSA）より先に形成される
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
from src.models.ae import AutoEncoder
from src.metrics.structure import active_units, trustworthiness, continuity
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


def get_latent_repr(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """潜在表現を取得する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(X_tensor)
    return Z.cpu().numpy()


def evaluate_mse(model: nn.Module, X: np.ndarray) -> float:
    """MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return mse


def count_significant_singular_values(
    model: AutoEncoder,
    X_eval: np.ndarray,
    n_points: int = 10,
    threshold_ratio: float = 0.1
) -> float:
    """
    ランダムなテスト点でのデコーダヤコビアンの有意特異値数の平均を計算する。

    Args:
        model: AutoEncoder
        X_eval: 評価データ
        n_points: 評価点数
        threshold_ratio: 有意特異値の閾値（最大比）
    Returns:
        mean_n_sig: 平均有意特異値数
    """
    model.eval()
    n = len(X_eval)
    idx = np.random.choice(n, min(n_points, n), replace=False)

    n_sig_list = []
    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    for i in idx:
        with torch.no_grad():
            z = model.encode(X_tensor[i:i+1]).squeeze(0)

        try:
            J = compute_decoder_jacobian(model.decoder, z.to(device))
            J_np = J.cpu().numpy()
            _, n_sig, _ = analyze_singular_values(J_np, threshold_ratio=threshold_ratio)
            n_sig_list.append(n_sig)
        except Exception:
            continue

    return float(np.mean(n_sig_list)) if n_sig_list else 0.0


def train_with_checkpoints(
    model: AutoEncoder,
    train_loader: DataLoader,
    X_train: np.ndarray,
    X_test: np.ndarray,
    total_epochs: int,
    checkpoint_epochs: list,
    n_trust_samples: int = 500,
    n_jacobian_points: int = 10,
    k: int = 10
) -> dict:
    """
    チェックポイントで各種メトリクスを記録しながら訓練する。

    Args:
        model: AutoEncoder モデル
        train_loader: 訓練データローダー
        X_train: 訓練データ（MSE計算用）
        X_test: テストデータ
        total_epochs: 総エポック数
        checkpoint_epochs: メトリクスを記録するエポックのリスト
        n_trust_samples: Trustworthiness計算に使うサンプル数
        n_jacobian_points: ヤコビアン評価点数
        k: 近傍数
    Returns:
        history: エポックごとのメトリクス辞書
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Trustworthiness/Continuity計算用のサブサンプル
    n_test = X_test.shape[0]
    trust_idx = np.random.choice(n_test, min(n_trust_samples, n_test), replace=False)
    X_trust = X_test[trust_idx]

    history = {
        'epochs': [],
        'train_mse': [],
        'test_mse': [],
        'au': [],
        'trustworthiness': [],
        'continuity': [],
        'n_sig_sv': [],
    }

    checkpoint_set = set(checkpoint_epochs)

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device).float()

            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch in checkpoint_set:
            train_mse = epoch_loss / n_batches
            test_mse = evaluate_mse(model, X_test)

            Z_trust = get_latent_repr(model, X_trust)
            au = active_units(Z_trust)

            trust = trustworthiness(X_trust, Z_trust, k=k)
            cont = continuity(X_trust, Z_trust, k=k)
            n_sig = count_significant_singular_values(model, X_test, n_points=n_jacobian_points)

            history['epochs'].append(epoch)
            history['train_mse'].append(train_mse)
            history['test_mse'].append(test_mse)
            history['au'].append(au)
            history['trustworthiness'].append(trust)
            history['continuity'].append(cont)
            history['n_sig_sv'].append(n_sig)

            print(f"  Epoch {epoch:>4}: train_mse={train_mse:.6f}, test_mse={test_mse:.6f}, "
                  f"AU={au}, Trust={trust:.4f}, Cont={cont:.4f}, n_sig_SV={n_sig:.2f}")

    return history


def plot_dynamics(history: dict, manifold_name: str, d_true: int):
    """学習ダイナミクスのプロット。"""
    epochs = history['epochs']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'E6 Training Dynamics: {manifold_name} (m=d_true={d_true})', fontsize=14)

    # MSE（train & test）
    ax = axes[0, 0]
    ax.plot(epochs, history['train_mse'], 'o-', color='steelblue', label='Train MSE', linewidth=2)
    ax.plot(epochs, history['test_mse'], 's--', color='coral', label='Test MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Reconstruction MSE', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Active Units
    ax = axes[0, 1]
    ax.plot(epochs, history['au'], 'o-', color='seagreen', linewidth=2)
    ax.axhline(y=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Active Units', fontsize=11)
    ax.set_title('Active Units vs Epoch', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, d_true + 2)

    # Trustworthiness
    ax = axes[1, 0]
    ax.plot(epochs, history['trustworthiness'], 'o-', color='orchid', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Trustworthiness', fontsize=11)
    ax.set_title('Trustworthiness vs Epoch (k=10)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)

    # Continuity
    ax = axes[1, 1]
    ax.plot(epochs, history['continuity'], 'o-', color='darkorange', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Continuity', fontsize=11)
    ax.set_title('Continuity vs Epoch (k=10)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)

    # Number of significant singular values
    ax = axes[2, 0]
    ax.plot(epochs, history['n_sig_sv'], 'o-', color='teal', linewidth=2)
    ax.axhline(y=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean # significant SVs', fontsize=11)
    ax.set_title('Jacobian Significant Singular Values', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Trust & Continuity combined
    ax = axes[2, 1]
    ax.plot(epochs, history['trustworthiness'], 'o-', color='orchid', linewidth=2,
            label='Trustworthiness')
    ax.plot(epochs, history['continuity'], 's--', color='darkorange', linewidth=2,
            label='Continuity')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Trust & Continuity Combined', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('results/figures/E6_dynamics.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E6_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/E6_dynamics.pdf, .png")


def main():
    """実験 E6 のメイン関数。"""
    n_train = 3000
    n_test = 1000
    total_epochs = 300
    d_true = 2
    m = 2  # m = d_true

    checkpoint_epochs = [1, 5, 10, 20, 50, 100, 150, 200, 300]

    # Swiss Roll データ生成
    print("Generating Swiss Roll data ...")
    X_train, _, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test, _, _ = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)

    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64, train=True)

    print(f"\n=== Experiment E6: Swiss Roll (d_true={d_true}, m={m}) ===")
    print(f"  Checkpoints: {checkpoint_epochs}")

    model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])

    history = train_with_checkpoints(
        model, train_loader, X_train, X_test,
        total_epochs=total_epochs,
        checkpoint_epochs=checkpoint_epochs,
        n_trust_samples=500,
        n_jacobian_points=10,
        k=10
    )

    # 結果保存
    save_data = {
        'manifold': 'SwissRoll',
        'd_true': d_true,
        'm': m,
        'total_epochs': total_epochs,
        'history': history
    }

    with open('results/tables/E6_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/E6_results.json")

    # プロット
    plot_dynamics(history, 'SwissRoll', d_true)

    # サマリー表示
    print("\n=== E6 Summary ===")
    print("Swiss Roll (d_true=2, m=2):")
    print(f"  {'Epoch':>6} | {'Train MSE':>10} | {'Test MSE':>10} | {'AU':>3} | {'Trust':>7} | {'Cont':>7} | {'n_sig_SV':>9}")
    print(f"  {'-'*65}")
    for i, ep in enumerate(history['epochs']):
        print(f"  {ep:>6} | {history['train_mse'][i]:>10.6f} | {history['test_mse'][i]:>10.6f} | "
              f"{history['au'][i]:>3} | {history['trustworthiness'][i]:>7.4f} | "
              f"{history['continuity'][i]:>7.4f} | {history['n_sig_sv'][i]:>9.2f}")

    # グローバル vs ローカル構造の形成順序を分析
    print("\n  Analysis: Global vs Local structure formation:")
    trust_vals = history['trustworthiness']
    n_sig_vals = history['n_sig_sv']
    trust_thresh = 0.9
    sig_thresh = d_true - 0.5

    trust_epoch = None
    sig_epoch = None
    for i, ep in enumerate(history['epochs']):
        if trust_epoch is None and trust_vals[i] >= trust_thresh:
            trust_epoch = ep
        if sig_epoch is None and n_sig_vals[i] >= sig_thresh:
            sig_epoch = ep

    print(f"  Trustworthiness >= {trust_thresh} first reached at epoch: {trust_epoch}")
    print(f"  Jacobian n_sig_SV >= {sig_thresh} first reached at epoch: {sig_epoch}")
    if trust_epoch is not None and sig_epoch is not None:
        if trust_epoch < sig_epoch:
            print("  --> Global structure (Trustworthiness) formed BEFORE local structure (Jacobian)")
        elif trust_epoch > sig_epoch:
            print("  --> Local structure (Jacobian) formed BEFORE global structure (Trustworthiness)")
        else:
            print("  --> Global and local structure formed at same epoch")


if __name__ == '__main__':
    main()
