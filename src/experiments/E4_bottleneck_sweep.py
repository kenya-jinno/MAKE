"""
実験 E4: ボトルネック次元スイープ
対応定理: 定理5（レート歪み肘点）, 定理6（AU飽和）
検証仮説: MNISTのAUがd_ID付近で飽和する
"""

import sys
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.loaders import get_mnist, get_fashion_mnist
from src.models.ae import AutoEncoder, train_ae
from src.models.vae import VAE, train_vae
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate, pca_estimate
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


def get_subset_loader(train_loader, n_samples=10000, batch_size=128):
    """訓練データのサブセットを取得する。"""
    all_x = []
    all_y = []
    for batch in train_loader:
        x, y = batch
        all_x.append(x)
        all_y.append(y)
        if len(all_x) * x.size(0) >= n_samples:
            break

    X = torch.cat(all_x, dim=0)[:n_samples]
    X = X.view(X.size(0), -1)  # flatten

    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_test_embeddings(model, test_loader, n_samples=2000, is_vae=False):
    """テストデータの潜在表現を取得する。"""
    model.eval()
    all_z = []
    all_x = []
    n_collected = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            x = x.view(x.size(0), -1).to(device).float()

            if is_vae:
                mu, _ = model.encode(x)
                z = mu
            else:
                z = model.encode(x)

            all_z.append(z.cpu())
            all_x.append(x.cpu())
            n_collected += x.size(0)
            if n_collected >= n_samples:
                break

    Z = torch.cat(all_z, dim=0)[:n_samples].numpy()
    X = torch.cat(all_x, dim=0)[:n_samples].numpy()
    return Z, X


def compute_test_mse(model, test_loader, n_samples=2000, is_vae=False):
    """テストデータの MSE を計算する。"""
    model.eval()
    total_loss = 0.0
    n_total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            x = x.view(x.size(0), -1).to(device).float()

            if is_vae:
                recon, _, _ = model(x)
            else:
                recon = model(x)

            total_loss += nn.MSELoss(reduction='sum')(recon, x).item()
            n_total += x.size(0)

            if n_total >= n_samples:
                break

    return total_loss / n_total


def run_sweep(dataset_name: str, train_loader: DataLoader, test_loader: DataLoader,
              input_dim: int, epochs: int = 50) -> dict:
    """ボトルネック次元スイープを実行する。"""
    latent_dims = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]

    results = {
        'dataset': dataset_name,
        'latent_dims': latent_dims,
        'ae': {
            'mse': [], 'au': [], 'id_twonn': [], 'id_mle': [], 'id_pca': [],
            'cka': []
        },
        'vae': {
            'mse': [], 'au': [], 'id_twonn': [], 'id_mle': [], 'id_pca': [],
            'cka': []
        }
    }

    print(f"\n=== E4: {dataset_name} Bottleneck Sweep ===")
    print(f"latent_dims = {latent_dims}")

    # 参照表現（入力そのもの）を取得
    X_ref_list = []
    n_ref = 0
    for batch in test_loader:
        x = batch[0]
        x = x.view(x.size(0), -1)
        X_ref_list.append(x)
        n_ref += x.size(0)
        if n_ref >= 2000:
            break
    X_ref = torch.cat(X_ref_list, dim=0)[:2000].numpy()

    for m in latent_dims:
        print(f"\n--- m={m} ---")

        # AE
        print(f"  Training AE (m={m}, epochs={epochs}) ...")
        ae = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[512, 256])
        train_ae(ae, train_loader, epochs=epochs, lr=1e-3, device=device)

        mse_ae = compute_test_mse(ae, test_loader, n_samples=2000)
        Z_ae, _ = get_test_embeddings(ae, test_loader, n_samples=2000)

        au_ae = active_units(Z_ae)
        id_twonn_ae = twonn_estimate(Z_ae)
        id_mle_ae = mle_estimate(Z_ae, k=10)
        id_pca_ae = pca_estimate(Z_ae)
        cka_ae = centered_kernel_alignment(X_ref[:len(Z_ae)], Z_ae)

        results['ae']['mse'].append(mse_ae)
        results['ae']['au'].append(au_ae)
        results['ae']['id_twonn'].append(id_twonn_ae)
        results['ae']['id_mle'].append(id_mle_ae)
        results['ae']['id_pca'].append(id_pca_ae)
        results['ae']['cka'].append(cka_ae)

        print(f"    AE: MSE={mse_ae:.4f}, AU={au_ae}, ID_TwoNN={id_twonn_ae:.2f}, CKA={cka_ae:.4f}")

        torch.save(ae.state_dict(), f'results/E4_{dataset_name}_AE_m{m}.pth')

        # VAE
        print(f"  Training VAE (m={m}, epochs={epochs}) ...")
        vae = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[512, 256])
        train_vae(vae, train_loader, epochs=epochs, lr=1e-3, beta=1.0, device=device)

        mse_vae = compute_test_mse(vae, test_loader, n_samples=2000, is_vae=True)
        Z_vae, _ = get_test_embeddings(vae, test_loader, n_samples=2000, is_vae=True)

        au_vae = active_units(Z_vae)
        id_twonn_vae = twonn_estimate(Z_vae)
        id_mle_vae = mle_estimate(Z_vae, k=10)
        id_pca_vae = pca_estimate(Z_vae)
        cka_vae = centered_kernel_alignment(X_ref[:len(Z_vae)], Z_vae)

        results['vae']['mse'].append(mse_vae)
        results['vae']['au'].append(au_vae)
        results['vae']['id_twonn'].append(id_twonn_vae)
        results['vae']['id_mle'].append(id_mle_vae)
        results['vae']['id_pca'].append(id_pca_vae)
        results['vae']['cka'].append(cka_vae)

        print(f"    VAE: MSE={mse_vae:.4f}, AU={au_vae}, ID_TwoNN={id_twonn_vae:.2f}, CKA={cka_vae:.4f}")

        torch.save(vae.state_dict(), f'results/E4_{dataset_name}_VAE_m{m}.pth')

    return results


def plot_results(results: dict, dataset_name: str):
    """実験結果をプロットする。"""
    m_vals = results['latent_dims']
    ae = results['ae']
    vae = results['vae']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # MSE vs m
    ax = axes[0, 0]
    ax.semilogy(m_vals, ae['mse'], 'o-', color='steelblue', label='AE', linewidth=2)
    ax.semilogy(m_vals, vae['mse'], 's-', color='coral', label='VAE', linewidth=2)
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('Test MSE (log scale)')
    ax.set_title(f'{dataset_name}: MSE vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AU vs m
    ax = axes[0, 1]
    ax.plot(m_vals, ae['au'], 'o-', color='steelblue', label='AE', linewidth=2)
    ax.plot(m_vals, vae['au'], 's-', color='coral', label='VAE', linewidth=2)
    ax.plot(m_vals, m_vals, 'k--', alpha=0.5, label='AU=m (linear)')
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('Active Units')
    ax.set_title(f'{dataset_name}: Active Units vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ID_TwoNN vs m
    ax = axes[0, 2]
    ax.plot(m_vals, ae['id_twonn'], 'o-', color='steelblue', label='AE (TwoNN)', linewidth=2)
    ax.plot(m_vals, vae['id_twonn'], 's-', color='coral', label='VAE (TwoNN)', linewidth=2)
    ax.plot(m_vals, ae['id_mle'], '^--', color='steelblue', alpha=0.6, label='AE (MLE)', linewidth=1.5)
    ax.plot(m_vals, vae['id_mle'], 'D--', color='coral', alpha=0.6, label='VAE (MLE)', linewidth=1.5)
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('Estimated ID')
    ax.set_title(f'{dataset_name}: ID estimates vs m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # CKA vs m
    ax = axes[1, 0]
    ax.plot(m_vals, ae['cka'], 'o-', color='steelblue', label='AE', linewidth=2)
    ax.plot(m_vals, vae['cka'], 's-', color='coral', label='VAE', linewidth=2)
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('CKA')
    ax.set_title(f'{dataset_name}: CKA vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ID_PCA vs m
    ax = axes[1, 1]
    ax.plot(m_vals, ae['id_pca'], 'o-', color='steelblue', label='AE', linewidth=2)
    ax.plot(m_vals, vae['id_pca'], 's-', color='coral', label='VAE', linewidth=2)
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('ID (PCA)')
    ax.set_title(f'{dataset_name}: PCA-based ID vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE vs AU
    ax = axes[1, 2]
    ax.scatter(ae['au'], ae['mse'], c='steelblue', s=80, label='AE', zorder=5)
    ax.scatter(vae['au'], vae['mse'], c='coral', s=80, marker='s', label='VAE', zorder=5)
    for i, m in enumerate(m_vals):
        ax.annotate(str(m), (ae['au'][i], ae['mse'][i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=7)
    ax.set_xlabel('Active Units')
    ax.set_ylabel('Test MSE')
    ax.set_title(f'{dataset_name}: MSE vs Active Units')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'E4: Bottleneck Sweep — {dataset_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/figures/E4_{dataset_name}_sweep.pdf', bbox_inches='tight')
    plt.savefig(f'results/figures/E4_{dataset_name}_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/figures/E4_{dataset_name}_sweep.pdf, .png")


def main():
    parser = argparse.ArgumentParser(description='E4: Bottleneck sweep experiment')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_train', type=int, default=10000)
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}, epochs={args.epochs}, n_train={args.n_train}")

    # データセット読み込み
    if args.dataset == 'mnist':
        train_loader_orig, test_loader, input_dim = get_mnist(batch_size=128)
    else:
        train_loader_orig, test_loader, input_dim = get_fashion_mnist(batch_size=128)

    # サブセット取得
    train_loader = get_subset_loader(train_loader_orig, n_samples=args.n_train, batch_size=128)

    # テストローダーをフラット化
    test_data = []
    for batch in test_loader:
        x = batch[0]
        test_data.append(x)
    X_test = torch.cat(test_data, dim=0)
    X_test_flat = X_test.view(X_test.size(0), -1)
    test_loader_flat = DataLoader(TensorDataset(X_test_flat), batch_size=128, shuffle=False)

    # スイープ実行
    results = run_sweep(args.dataset, train_loader, test_loader_flat, input_dim, epochs=args.epochs)

    # JSON 保存
    with open(f'results/tables/E4_results_{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/tables/E4_results_{args.dataset}.json")

    # CSV 保存
    df_ae = pd.DataFrame({
        'latent_dim': results['latent_dims'],
        'mse': results['ae']['mse'],
        'au': results['ae']['au'],
        'id_twonn': results['ae']['id_twonn'],
        'id_mle': results['ae']['id_mle'],
        'id_pca': results['ae']['id_pca'],
        'cka': results['ae']['cka'],
        'model': 'AE'
    })
    df_vae = pd.DataFrame({
        'latent_dim': results['latent_dims'],
        'mse': results['vae']['mse'],
        'au': results['vae']['au'],
        'id_twonn': results['vae']['id_twonn'],
        'id_mle': results['vae']['id_mle'],
        'id_pca': results['vae']['id_pca'],
        'cka': results['vae']['cka'],
        'model': 'VAE'
    })
    df = pd.concat([df_ae, df_vae], ignore_index=True)
    df.to_csv(f'results/tables/E4_results_{args.dataset}.csv', index=False)
    print(f"Saved: results/tables/E4_results_{args.dataset}.csv")

    # プロット
    plot_results(results, args.dataset)

    print("\n=== E4 Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"{'m':>4} | {'AE_MSE':>10} | {'AE_AU':>6} | {'VAE_MSE':>10} | {'VAE_AU':>6}")
    print("-" * 50)
    for i, m in enumerate(results['latent_dims']):
        print(f"{m:>4} | {results['ae']['mse'][i]:>10.4f} | {results['ae']['au'][i]:>6} | "
              f"{results['vae']['mse'][i]:>10.4f} | {results['vae']['au'][i]:>6}")


if __name__ == '__main__':
    main()
