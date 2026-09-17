"""
実験 Exp-Iso-ID: TwoNN ID の幾何学的正確性検証
通常 AE vs IsometricAE のデコーダ等長性が TwoNN 推定に与える影響を定量化する．

仮説:
- 等長性(κ≈1)が改善されると，潜在空間で計算した TwoNN ID が
  真の d_ID (=2 for Swiss Roll) により安定的に収束する
- 通常 AE (κ>>1) では潜在空間の計量が歪み，TwoNN ID の値が不安定・バイアスを持つ可能性がある
- 真の d_ID は入力空間の TwoNN で ground truth として確認済み

実験設計:
- Swiss Roll (d_true=2, 入力空間 TwoNN≈2)
- m ∈ {2, 3, 4, 6, 8} で AE vs IsometricAE を学習
- 潜在表現 z を取り出し，latent TwoNN ID を計算
- κ (条件数) と latent TwoNN ID の関係を比較
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthetic import generate_swiss_roll
from src.models.ae import AutoEncoder, train_ae
from src.models.isometric_ae import IsometricAE, train_isometric_ae
from src.metrics.intrinsic_dim import twonn_estimate
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def make_dataloader(X, batch_size=128):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_latent(model, X, is_iso=False):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if is_iso:
            z = model.encoder(X_t)
        else:
            z = model.encoder(X_t)
    return z.cpu().numpy()


def compute_kappa(model, z_samples, is_iso=False, n_points=50):
    """潜在空間サンプル点でのデコーダ条件数の平均を計算"""
    model.eval()
    kappas = []
    indices = np.random.choice(len(z_samples), min(n_points, len(z_samples)), replace=False)
    for idx in indices:
        z_t = torch.tensor(z_samples[idx:idx+1], dtype=torch.float32).to(device)
        z_t.requires_grad_(True)
        if is_iso:
            x_recon = model.decoder(z_t)
        else:
            x_recon = model.decoder(z_t)
        # Jacobian computation
        m = z_t.shape[1]
        N = x_recon.shape[1]
        J = []
        for i in range(N):
            if z_t.grad is not None:
                z_t.grad.zero_()
            grad = torch.autograd.grad(x_recon[0, i], z_t, retain_graph=True)[0]
            J.append(grad[0].detach().cpu().numpy())
        J = np.array(J)  # (N, m)
        sv = np.linalg.svd(J, compute_uv=False)
        sv = sv[sv > 1e-10]
        if len(sv) > 0:
            kappas.append(sv[0] / sv[-1])
    return np.mean(kappas) if kappas else float('inf')


def run_ae_experiment(X_train, X_test, m, n_input=3, epochs=300):
    """通常 AE の学習と評価"""
    model = AutoEncoder(
        input_dim=n_input,
        latent_dim=m,
        hidden_dims=[64, 32]
    ).to(device)
    loader = make_dataloader(X_train)
    train_ae(model, loader, epochs=epochs, lr=1e-3, device=device)

    # MSE
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = nn.MSELoss()(recon, X_t).item()

    # Latent representations
    z = get_latent(model, X_test)

    # TwoNN in latent space
    latent_id = twonn_estimate(z)

    # κ (condition number)
    try:
        kappa = compute_kappa(model, z, is_iso=False, n_points=30)
    except Exception:
        kappa = float('nan')

    return {'mse': mse, 'latent_id': latent_id, 'kappa': kappa, 'z': z}


def run_iso_experiment(X_train, X_test, m, n_input=3, epochs=300):
    """IsometricAE の学習と評価"""
    model = IsometricAE(
        input_dim=n_input,
        latent_dim=m,
        hidden_dims=[64, 32],
        lambda_iso=0.01
    ).to(device)
    loader = make_dataloader(X_train)
    train_isometric_ae(model, loader, epochs=epochs, lr=1e-3, device=device)

    # MSE
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, _ = model(X_t)
        mse = nn.MSELoss()(recon, X_t).item()

    # Latent representations
    z = get_latent(model, X_test, is_iso=True)

    # TwoNN in latent space
    latent_id = twonn_estimate(z)

    # κ
    try:
        kappa = compute_kappa(model, z, is_iso=True, n_points=30)
    except Exception:
        kappa = float('nan')

    return {'mse': mse, 'latent_id': latent_id, 'kappa': kappa, 'z': z}


def main():
    print("=" * 60)
    print("Exp-Iso-ID: TwoNN ID Geometric Accuracy Validation")
    print("=" * 60)

    # Swiss Roll データ生成
    X_train_full, _, _ = generate_swiss_roll(n_samples=5000, noise=0.05, seed=SEED)
    X_train = X_train_full[:4000]
    X_test  = X_train_full[4000:]
    n_input = X_train.shape[1]  # 3

    # 入力空間 TwoNN (ground truth)
    input_id = twonn_estimate(X_test)
    print(f"\nInput-space TwoNN ID (ground truth): {input_id:.3f} (expected: ~2.0)")

    m_values = [2, 3, 4, 6, 8]

    results = {
        'input_space_id': input_id,
        'd_true': 2,
        'ae': {},
        'iso': {}
    }

    print("\n{:>4s}  {:>14s}  {:>8s}  {:>12s}  |  {:>14s}  {:>8s}  {:>12s}".format(
        'm', 'AE latent-ID', 'AE κ', 'AE MSE', 'Iso latent-ID', 'Iso κ', 'Iso MSE'))
    print("-" * 80)

    for m in m_values:
        print(f"  Training m={m}...")
        ae_res  = run_ae_experiment(X_train, X_test, m, n_input=n_input, epochs=300)
        iso_res = run_iso_experiment(X_train, X_test, m, n_input=n_input, epochs=300)

        results['ae'][str(m)]  = {k: float(v) for k, v in ae_res.items() if k != 'z'}
        results['iso'][str(m)] = {k: float(v) for k, v in iso_res.items() if k != 'z'}

        print(f"  m={m:2d}  AE: latentID={ae_res['latent_id']:.3f}, κ={ae_res['kappa']:.2f}, MSE={ae_res['mse']:.4f}"
              f"  |  Iso: latentID={iso_res['latent_id']:.3f}, κ={iso_res['kappa']:.2f}, MSE={iso_res['mse']:.4f}")

    # Save results
    with open('results/tables/Exp_Iso_ID.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/tables/Exp_Iso_ID.json")

    # --- Plot ---
    ms = m_values
    ae_ids  = [results['ae'][str(m)]['latent_id'] for m in ms]
    iso_ids = [results['iso'][str(m)]['latent_id'] for m in ms]
    ae_ks   = [results['ae'][str(m)]['kappa'] for m in ms]
    iso_ks  = [results['iso'][str(m)]['kappa'] for m in ms]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ms, ae_ids, 'o-', color='tab:blue', label='AE (通常)')
    ax.plot(ms, iso_ids, 's-', color='tab:orange', label='IsometricAE')
    ax.axhline(2.0, color='gray', linestyle='--', label='真の $d_{\\rm ID}=2$')
    ax.axhline(input_id, color='green', linestyle=':', label=f'入力空間 TwoNN={input_id:.2f}')
    ax.set_xlabel('ボトルネック次元 $m$')
    ax.set_ylabel('潜在空間 TwoNN ID')
    ax.set_title('潜在空間 TwoNN ID: AE vs IsometricAE')
    ax.legend(fontsize=9)
    ax.set_xticks(ms)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(ms, ae_ks,  'o-', color='tab:blue', label='AE κ')
    ax2.plot(ms, iso_ks, 's-', color='tab:orange', label='IsometricAE κ')
    ax2.axhline(1.0, color='gray', linestyle='--', label='等長性 κ=1')
    ax2.set_xlabel('ボトルネック次元 $m$')
    ax2.set_ylabel('条件数 $\\kappa$')
    ax2.set_title('デコーダ条件数 κ: AE vs IsometricAE')
    ax2.legend(fontsize=9)
    ax2.set_xticks(ms)
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/Exp_Iso_ID.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('results/figures/Exp_Iso_ID.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved results/figures/Exp_Iso_ID.{pdf,png}")

    print("\n=== Summary ===")
    print(f"Input-space TwoNN ID: {input_id:.3f}")
    for m in ms:
        ae  = results['ae'][str(m)]
        iso = results['iso'][str(m)]
        print(f"m={m}: AE(ID={ae['latent_id']:.2f}, κ={ae['kappa']:.1f}) | "
              f"Iso(ID={iso['latent_id']:.2f}, κ={iso['kappa']:.1f})")

    print("\nDone.")


if __name__ == '__main__':
    main()
