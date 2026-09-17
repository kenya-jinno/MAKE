"""
実験 E4_extended: MNIST ボトルネック次元スイープ（長期学習版）
対応定理: 定理5（レート歪み肘点）, 定理6（AU飽和）
検証仮説: 300エポック学習でMSE肘点とAU飽和が明確に現れる
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder
from src.models.vae import VAE, vae_loss
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import active_units, centered_kernel_alignment

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


def load_mnist_subset(n_train: int = 20000, n_test: int = 3000) -> tuple:
    """
    MNIST データセットのサブセットをロードする。

    Returns:
        X_train, X_test: フラット化された (n, 784) numpy 配列
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform
    )

    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(len(train_ds), min(n_train, len(train_ds)), replace=False)
    test_idx = rng.choice(len(test_ds), min(n_test, len(test_ds)), replace=False)

    X_train = np.stack([train_ds[i][0].numpy().flatten() for i in train_idx])
    X_test = np.stack([test_ds[i][0].numpy().flatten() for i in test_idx])

    return X_train.astype(np.float32), X_test.astype(np.float32)


def evaluate_mse(model: nn.Module, X_test: np.ndarray) -> float:
    """テストデータの MSE を計算する。"""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return float(mse)


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


def train_ae_model(input_dim: int, latent_dim: int, train_loader: DataLoader,
                   epochs: int = 300, lr: float = 1e-3) -> tuple:
    """
    AE を訓練して損失履歴を返す。

    Returns:
        model, losses
    """
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[256, 128])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
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
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"    AE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return model, losses


def train_vae_model(input_dim: int, latent_dim: int, train_loader: DataLoader,
                    epochs: int = 300, lr: float = 1e-3, beta: float = 4.0) -> tuple:
    """
    VAE を訓練して損失履歴を返す。

    Returns:
        model, losses
    """
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[256, 128])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
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
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"    VAE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return model, losses


def evaluate_ae(model: AutoEncoder, X_test: np.ndarray, X_eval: np.ndarray) -> dict:
    """AE モデルを評価する。"""
    mse = evaluate_mse(model, X_test)
    Z = get_latent_repr_ae(model, X_eval)
    au = active_units(Z, threshold=0.01)

    # サブサンプルでID推定（コスト削減）
    n_id = min(2000, len(Z))
    idx = np.random.choice(len(Z), n_id, replace=False)
    Z_sub = Z[idx]

    id_twonn = twonn_estimate(Z_sub)
    id_mle = mle_estimate(Z_sub, k=10)

    # CKA
    X_eval_sub = X_eval[idx]
    cka = centered_kernel_alignment(X_eval_sub, Z_sub)

    return {
        'mse': float(mse),
        'au': int(au),
        'id_twonn': float(id_twonn),
        'id_mle': float(id_mle),
        'cka': float(cka),
    }


def evaluate_vae(model: VAE, X_test: np.ndarray, X_eval: np.ndarray) -> dict:
    """VAE モデルを評価する。"""
    # Reconstruction MSE
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, mu_test, logvar_test = model(X_test_t)
        mse = nn.MSELoss()(recon, X_test_t).item()

    # 潜在表現
    Z = get_latent_repr_vae(model, X_eval)
    au = active_units(Z, threshold=0.01)

    n_id = min(2000, len(Z))
    idx = np.random.choice(len(Z), n_id, replace=False)
    Z_sub = Z[idx]

    id_twonn = twonn_estimate(Z_sub)
    id_mle = mle_estimate(Z_sub, k=10)

    X_eval_sub = X_eval[idx]
    cka = centered_kernel_alignment(X_eval_sub, Z_sub)

    return {
        'mse': float(mse),
        'au': int(au),
        'id_twonn': float(id_twonn),
        'id_mle': float(id_mle),
        'cka': float(cka),
    }


def plot_results(ae_results: list, vae_results: list, latent_dims: list):
    """結果をプロットする。"""
    # --- MSE plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ae_mse = [r['mse'] for r in ae_results]
    vae_mse = [r['mse'] for r in vae_results]
    ax.semilogy(latent_dims, ae_mse, 'o-', color='steelblue', linewidth=2, label='AE MSE')
    ax.semilogy(latent_dims, vae_mse, 's--', color='coral', linewidth=2, label='VAE MSE (β=4)')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7, label='Expected ID≈10')
    ax.set_xlabel('Bottleneck dim m', fontsize=13)
    ax.set_ylabel('Test MSE (log scale)', fontsize=13)
    ax.set_title('E4_extended: MSE vs Bottleneck Dim (MNIST, 300 epochs)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(latent_dims)
    plt.tight_layout()
    plt.savefig('results/figures/E4_extended_mse.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E4_extended_mse.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- AU plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ae_au = [r['au'] for r in ae_results]
    vae_au = [r['au'] for r in vae_results]
    ax.plot(latent_dims, latent_dims, 'k--', linewidth=1, alpha=0.4, label='AU=m (diagonal)')
    ax.plot(latent_dims, ae_au, 'o-', color='steelblue', linewidth=2, label='AE AU')
    ax.plot(latent_dims, vae_au, 's--', color='coral', linewidth=2, label='VAE AU (β=4)')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7, label='Expected ID≈10')
    ax.set_xlabel('Bottleneck dim m', fontsize=13)
    ax.set_ylabel('Active Units', fontsize=13)
    ax.set_title('E4_extended: Active Units vs Bottleneck Dim (MNIST)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(latent_dims)
    plt.tight_layout()
    plt.savefig('results/figures/E4_extended_au.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E4_extended_au.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- ID plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ae_id = [r['id_twonn'] for r in ae_results]
    ae_mle = [r['id_mle'] for r in ae_results]
    ax.plot(latent_dims, ae_id, 'o-', color='seagreen', linewidth=2, label='TwoNN ID (AE)')
    ax.plot(latent_dims, ae_mle, 's--', color='orchid', linewidth=2, label='MLE ID (AE)')
    ax.plot(latent_dims, latent_dims, 'k:', linewidth=1, alpha=0.5, label='ID=m')
    ax.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='Expected ID≈10')
    ax.set_xlabel('Bottleneck dim m', fontsize=13)
    ax.set_ylabel('Intrinsic Dimension', fontsize=13)
    ax.set_title('E4_extended: Intrinsic Dim of Latent Space (MNIST)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(latent_dims)
    plt.tight_layout()
    plt.savefig('results/figures/E4_extended_id.pdf', bbox_inches='tight')
    plt.savefig('results/figures/E4_extended_id.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSaved: E4_extended_mse, _au, _id (PDF and PNG)")


def main():
    """実験 E4_extended のメイン関数。"""
    n_train = 20000  # コスト削減のため20000を使用
    n_test = 3000
    epochs = 300
    beta_vae = 4.0
    latent_dims = [1, 2, 4, 8, 12, 16, 20, 32, 64, 128, 256]

    print(f"Loading MNIST (train={n_train}, test={n_test}) ...")
    X_train, X_test = load_mnist_subset(n_train=n_train, n_test=n_test)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    input_dim = X_train.shape[1]  # 784

    # 評価用サブセット
    X_eval = X_test[:2000]

    # DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    ae_results = []
    vae_results = []

    print("\n=== E4_extended: AE sweep ===")
    for m in latent_dims:
        print(f"\n--- AE m={m} ---")
        model, losses = train_ae_model(input_dim, m, train_loader, epochs=epochs, lr=1e-3)
        metrics = evaluate_ae(model, X_test, X_eval)
        ae_results.append({'m': m, **metrics})
        print(f"  Result: MSE={metrics['mse']:.6f}, AU={metrics['au']}, "
              f"ID_2NN={metrics['id_twonn']:.2f}, ID_MLE={metrics['id_mle']:.2f}, "
              f"CKA={metrics['cka']:.4f}")

        # モデル保存（オプション）
        torch.save(model.state_dict(), f'results/E4_extended_AE_m{m}.pth')

    print("\n=== E4_extended: VAE sweep (β=4) ===")
    for m in latent_dims:
        print(f"\n--- VAE m={m} ---")
        model, losses = train_vae_model(input_dim, m, train_loader,
                                        epochs=epochs, lr=1e-3, beta=beta_vae)
        metrics = evaluate_vae(model, X_test, X_eval)
        vae_results.append({'m': m, **metrics})
        print(f"  Result: MSE={metrics['mse']:.6f}, AU={metrics['au']}, "
              f"ID_2NN={metrics['id_twonn']:.2f}, ID_MLE={metrics['id_mle']:.2f}, "
              f"CKA={metrics['cka']:.4f}")

    # 結果保存 (JSON)
    save_data = {
        'dataset': 'MNIST',
        'n_train': n_train,
        'n_test': n_test,
        'epochs': epochs,
        'beta_vae': beta_vae,
        'latent_dims': latent_dims,
        'ae_results': ae_results,
        'vae_results': vae_results,
    }
    with open('results/tables/E4_extended.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/E4_extended.json")

    # CSV 保存
    csv_rows = []
    for r in ae_results:
        csv_rows.append({'model': 'AE', **r})
    for r in vae_results:
        csv_rows.append({'model': 'VAE', **r})

    with open('results/tables/E4_extended.csv', 'w', newline='') as f:
        fieldnames = ['model', 'm', 'mse', 'au', 'id_twonn', 'id_mle', 'cka']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print("Saved: results/tables/E4_extended.csv")

    # プロット
    plot_results(ae_results, vae_results, latent_dims)

    # サマリー
    print("\n=== E4_extended Summary (AE) ===")
    print(f"{'m':>5} {'MSE':>12} {'AU':>5} {'ID_2NN':>8} {'ID_MLE':>8} {'CKA':>8}")
    print("-" * 50)
    for r in ae_results:
        print(f"{r['m']:>5} {r['mse']:>12.6f} {r['au']:>5} "
              f"{r['id_twonn']:>8.2f} {r['id_mle']:>8.2f} {r['cka']:>8.4f}")

    print("\n=== E4_extended Summary (VAE, β=4) ===")
    print(f"{'m':>5} {'MSE':>12} {'AU':>5} {'ID_2NN':>8} {'ID_MLE':>8} {'CKA':>8}")
    print("-" * 50)
    for r in vae_results:
        print(f"{r['m']:>5} {r['mse']:>12.6f} {r['au']:>5} "
              f"{r['id_twonn']:>8.2f} {r['id_mle']:>8.2f} {r['cka']:>8.4f}")

    # MSE肘点の検出（二次差分）
    mse_vals = np.array([r['mse'] for r in ae_results])
    if len(mse_vals) >= 3:
        # 対数スケールで二次差分
        log_mse = np.log(mse_vals + 1e-10)
        second_diff = np.diff(np.diff(log_mse))
        elbow_idx = np.argmax(second_diff) + 1  # +1 for double diff offset
        elbow_m = latent_dims[elbow_idx]
        print(f"\n  MSE elbow detected at m={elbow_m} (expected ~10)")

    # VAE AU飽和点
    vae_au_vals = [r['au'] for r in vae_results]
    for i, (m, au) in enumerate(zip(latent_dims, vae_au_vals)):
        if au >= m:
            print(f"  VAE AU saturation at m={m} (AU={au})")
            break


if __name__ == '__main__':
    main()
