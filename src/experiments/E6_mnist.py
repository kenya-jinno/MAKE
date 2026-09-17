"""
実験 E6_mnist: MNISTでの学習ダイナミクス
対応定理: 定理2（接空間形成過程）
検証仮説: 複雑データでは「大域構造→局所構造」の時系列順序が現れる
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
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder
from src.metrics.intrinsic_dim import twonn_estimate
from src.metrics.structure import active_units, trustworthiness, continuity, centered_kernel_alignment

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

# 記録するエポック
RECORD_EPOCHS = [1, 2, 5, 10, 20, 50, 100, 150, 200]


def load_mnist(n_train: int = 10000, n_test: int = 1000) -> tuple:
    """
    MNIST データセットをロードする。

    Returns:
        X_train, X_test: フラット化された (n, 784) numpy 配列
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # データを ~/.cache にダウンロード（既存ならスキップ）
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform
    )

    # サブサンプル
    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(len(train_ds), n_train, replace=False)
    test_idx = rng.choice(len(test_ds), n_test, replace=False)

    X_train = np.stack([train_ds[i][0].numpy().flatten() for i in train_idx])
    X_test = np.stack([test_ds[i][0].numpy().flatten() for i in test_idx])

    return X_train.astype(np.float32), X_test.astype(np.float32)


def evaluate_all(model: nn.Module, X_train: np.ndarray, X_test: np.ndarray,
                 X_eval: np.ndarray, n_metrics: int = 1000) -> dict:
    """
    1エポック後の各種メトリクスを計算する。

    Args:
        model: AEモデル
        X_train: 訓練データ
        X_test: テストデータ（MSE用）
        X_eval: メトリクス評価用データ（サブサンプル）
        n_metrics: メトリクス評価点数
    Returns:
        metrics: メトリクス辞書
    """
    model.eval()

    # Train MSE
    X_train_t = torch.tensor(X_train[:2000], dtype=torch.float32).to(device)
    with torch.no_grad():
        recon_tr = model(X_train_t)
        train_mse = nn.MSELoss()(recon_tr, X_train_t).item()

    # Test MSE
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon_te = model(X_test_t)
        test_mse = nn.MSELoss()(recon_te, X_test_t).item()

    # 潜在表現
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z_eval = model.encode(X_eval_t).cpu().numpy()

    # Active Units
    au = active_units(Z_eval, threshold=0.01)

    # Trustworthiness (k=10)
    tw = trustworthiness(X_eval, Z_eval, k=10)

    # Continuity (k=10)
    cont = continuity(X_eval, Z_eval, k=10)

    # CKA
    cka = centered_kernel_alignment(X_eval, Z_eval)

    # TwoNN ID of latent space
    id_twonn = twonn_estimate(Z_eval)

    return {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'au': int(au),
        'trustworthiness': float(tw),
        'continuity': float(cont),
        'cka': float(cka),
        'id_twonn': float(id_twonn),
    }


def train_with_recording(
    model: nn.Module,
    train_loader: DataLoader,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_eval: np.ndarray,
    total_epochs: int = 200,
    lr: float = 1e-3,
    record_epochs: list = RECORD_EPOCHS
) -> dict:
    """
    学習ダイナミクスを記録しながら AE を訓練する。

    Returns:
        history: エポックごとのメトリクス履歴
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {epoch: {} for epoch in record_epochs}

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

        avg_loss = epoch_loss / n_batches

        if epoch in record_epochs:
            print(f"\n  [Epoch {epoch}] train_loss={avg_loss:.6f}")
            metrics = evaluate_all(model, X_train, X_test, X_eval)
            history[epoch] = metrics
            print(f"    train_mse={metrics['train_mse']:.6f}, test_mse={metrics['test_mse']:.6f}")
            print(f"    AU={metrics['au']}, Trust={metrics['trustworthiness']:.4f}, "
                  f"Cont={metrics['continuity']:.4f}, CKA={metrics['cka']:.4f}, "
                  f"ID_TwoNN={metrics['id_twonn']:.2f}")
        elif epoch % 10 == 0:
            print(f"  Epoch [{epoch}/{total_epochs}] Loss: {avg_loss:.6f}")

    return history


def plot_dynamics(history: dict, record_epochs: list):
    """学習ダイナミクスをプロットする。"""
    epochs = [e for e in record_epochs if history.get(e)]
    if not epochs:
        print("No history to plot!")
        return

    # メトリクス値を取得
    train_mse = [history[e]['train_mse'] for e in epochs]
    test_mse = [history[e]['test_mse'] for e in epochs]
    au = [history[e]['au'] for e in epochs]
    trust = [history[e]['trustworthiness'] for e in epochs]
    cont = [history[e]['continuity'] for e in epochs]
    cka = [history[e]['cka'] for e in epochs]
    id_twonn = [history[e]['id_twonn'] for e in epochs]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # MSE
    ax = axes[0, 0]
    ax.semilogy(epochs, train_mse, 'o-', color='steelblue', linewidth=2, label='Train MSE')
    ax.semilogy(epochs, test_mse, 's--', color='coral', linewidth=2, label='Test MSE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('MSE over Training', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AU
    ax = axes[0, 1]
    ax.plot(epochs, au, 'o-', color='seagreen', linewidth=2, markersize=8)
    ax.axhline(y=16, color='red', linestyle='--', alpha=0.5, label='latent_dim=16')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Active Units', fontsize=12)
    ax.set_title('Active Units over Training', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trustworthiness & Continuity
    ax = axes[0, 2]
    ax.plot(epochs, trust, 'o-', color='orchid', linewidth=2, label='Trustworthiness (k=10)')
    ax.plot(epochs, cont, 's--', color='orange', linewidth=2, label='Continuity (k=10)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Trustworthiness & Continuity', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # CKA
    ax = axes[1, 0]
    ax.plot(epochs, cka, 'o-', color='teal', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('CKA', fontsize=12)
    ax.set_title('CKA (Input vs Latent)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # TwoNN ID
    ax = axes[1, 1]
    ax.plot(epochs, id_twonn, 'o-', color='darkred', linewidth=2, markersize=8)
    ax.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Expected ID≈10')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('TwoNN ID', fontsize=12)
    ax.set_title('TwoNN ID of Latent Space', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normalized comparison (Global vs Local structure)
    ax = axes[1, 2]
    trust_norm = np.array(trust) / (max(trust) + 1e-8)
    cont_norm = np.array(cont) / (max(cont) + 1e-8)
    cka_norm = np.array(cka) / (max(cka) + 1e-8)
    ax.plot(epochs, trust_norm, 'o-', color='orchid', linewidth=2, label='Trustworthiness (norm)')
    ax.plot(epochs, cont_norm, 's--', color='orange', linewidth=2, label='Continuity (norm)')
    ax.plot(epochs, cka_norm, '^-', color='teal', linewidth=2, label='CKA (norm)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Global vs Local Structure Formation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('E6_mnist: MNIST Learning Dynamics (latent_dim=16, 200 epochs)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/E6_mnist_dynamics.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E6_mnist_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/E6_mnist_dynamics.pdf, .png")


def main():
    """実験 E6_mnist のメイン関数。"""
    n_train = 10000
    n_test = 1000
    n_metrics = 1000  # メトリクス評価点数
    latent_dim = 16   # MNISTのID≈10に近い
    epochs = 200

    print("Loading MNIST data ...")
    X_train, X_test = load_mnist(n_train=n_train, n_test=n_test)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # メトリクス評価用サブセット（テストデータから）
    rng = np.random.RandomState(SEED)
    eval_idx = rng.choice(n_test, min(n_metrics, n_test), replace=False)
    X_eval = X_test[eval_idx]

    # DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # モデル
    input_dim = X_train.shape[1]  # 784
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[256, 128])
    print(f"\nModel: AE(784 → 256 → 128 → {latent_dim} → 128 → 256 → 784)")

    print(f"\n=== E6_mnist: Training for {epochs} epochs ===")
    history = train_with_recording(
        model, train_loader, X_train, X_test, X_eval,
        total_epochs=epochs, lr=1e-3, record_epochs=RECORD_EPOCHS
    )

    # 結果保存
    # JSON シリアライズ可能な形式に変換
    save_history = {str(k): v for k, v in history.items()}
    save_data = {
        'dataset': 'MNIST',
        'n_train': n_train,
        'n_test': n_test,
        'latent_dim': latent_dim,
        'epochs': epochs,
        'record_epochs': RECORD_EPOCHS,
        'history': save_history,
    }
    with open('results/tables/E6_mnist.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/E6_mnist.json")

    # プロット
    plot_dynamics(history, RECORD_EPOCHS)

    # サマリー
    print("\n=== E6_mnist Summary ===")
    print(f"{'Epoch':>6} {'Train_MSE':>12} {'Test_MSE':>12} {'AU':>5} "
          f"{'Trust':>8} {'Cont':>8} {'CKA':>8} {'ID_2NN':>8}")
    print("-" * 75)
    for epoch in RECORD_EPOCHS:
        if epoch in history and history[epoch]:
            m = history[epoch]
            print(f"{epoch:>6} {m['train_mse']:>12.6f} {m['test_mse']:>12.6f} "
                  f"{m['au']:>5} {m['trustworthiness']:>8.4f} "
                  f"{m['continuity']:>8.4f} {m['cka']:>8.4f} "
                  f"{m['id_twonn']:>8.2f}")


if __name__ == '__main__':
    main()
