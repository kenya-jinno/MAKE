"""
実験 EQ: MNIST AE — m スイープによる κ・TwoNN・Trustworthiness の同時計測
目的: Stage 1（ID 推定）の幾何学的妥当性を実データで検証する．
     - AE の m を系統的に変化させ，κ（条件数），TwoNN，MLE，Trustworthiness を同時に計測
     - m が d_ID（≈10）を超えるにつれて κ が安定化し，TwoNN が収束する様子を示す
     - Swiss Roll のみでなく実データでも「m >> d_ID で TwoNN は幾何的に信頼しやすくなる」を実証

実験設定:
    データ: MNIST (train=10000, test=2000)
    モデル: AE, hidden=[512, 256], 200 epoch, lr=1e-3
    m: [4, 8, 12, 16, 20, 32, 48, 64, 96, 128]
    κ 計算: n_eval=8 点（各点で 784 backward pass → post-training のみ）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder, train_ae
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import trustworthiness
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def load_mnist(n_train=10000, n_test=2000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
    # サブセット
    torch.manual_seed(SEED)
    idx_tr = torch.randperm(len(train_ds))[:n_train]
    idx_te = torch.randperm(len(test_ds))[:n_test]
    X_train = train_ds.data[idx_tr].float().view(-1, 784) / 255.0
    X_test  = test_ds.data[idx_te].float().view(-1, 784) / 255.0
    return X_train.numpy(), X_test.numpy()


def compute_kappa(decoder, Z_test: np.ndarray, n_eval: int = 8, device='cpu') -> float:
    """デコーダヤコビアンの条件数（κ）をテスト点のサブセットで計算"""
    n_eval = min(n_eval, len(Z_test))
    idx = np.random.choice(len(Z_test), n_eval, replace=False)
    kappas = []
    for i in idx:
        z_t = torch.tensor(Z_test[i], dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(decoder, z_t)
        J_np = J.cpu().numpy()
        _, _, kappa = analyze_singular_values(J_np, threshold_ratio=0.01)
        kappas.append(kappa)
    return float(np.median(kappas))


def encode_data(model, X: np.ndarray, device='cpu') -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model.encoder(X_t)
    return Z.cpu().numpy()


def run_eq():
    print("=== EQ: MNIST AE m-sweep — κ・TwoNN・Trustworthiness ===")
    X_train, X_test = load_mnist(n_train=10000, n_test=2000)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    m_list = [4, 8, 12, 16, 20, 32, 48, 64, 96, 128]
    results = []

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=128, shuffle=True
    )

    for m in m_list:
        print(f"\n  Training AE m={m} ...")
        model = AutoEncoder(
            input_dim=784, latent_dim=m, hidden_dims=[512, 256]
        ).to(device)
        train_ae(model, loader, epochs=200, lr=1e-3, device=device)

        # 潜在表現
        Z_train = encode_data(model, X_train, device)
        Z_test  = encode_data(model, X_test, device)

        # TwoNN, MLE
        twonn = twonn_estimate(Z_test)
        mle   = mle_estimate(Z_test, k=10)

        # Trustworthiness (subsample for speed)
        n_tw = min(1000, len(X_test))
        tw = trustworthiness(X_test[:n_tw], Z_test[:n_tw], k=10)

        # κ (計算が重いのでメッセージを出す)
        print(f"    Computing κ (n_eval=8, 784 backprop each) ...")
        kappa = compute_kappa(model.decoder, Z_test, n_eval=8, device=device)

        rec_loss = float(np.mean((X_test - model.decoder(
            torch.tensor(Z_test, dtype=torch.float32).to(device)
        ).detach().cpu().numpy()) ** 2))

        print(f"    m={m:3d}: κ={kappa:.3f}, TwoNN={twonn:.3f}, MLE={mle:.3f}, "
              f"Trust={tw:.4f}, MSE={rec_loss:.5f}")

        results.append({
            'm': m, 'kappa': kappa,
            'twonn': twonn, 'mle': mle,
            'trustworthiness': tw, 'mse': rec_loss
        })

    # 保存
    out = {'d_id_est': 10.0, 'note': 'MNIST AE m-sweep, seed=42, 200ep', 'results': results}
    with open('results/tables/EQ_geometry_validity.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/tables/EQ_geometry_validity.json")

    # 結果表示
    print("\n=== EQ Summary ===")
    print(f"{'m':>5} {'κ':>10} {'TwoNN':>8} {'MLE':>8} {'Trust':>8}")
    for r in results:
        print(f"{r['m']:>5d} {r['kappa']:>10.3f} {r['twonn']:>8.3f} "
              f"{r['mle']:>8.3f} {r['trustworthiness']:>8.4f}")

    return results


def make_eq_figure(results):
    m_vals    = [r['m']              for r in results]
    kappas    = [r['kappa']          for r in results]
    twonns    = [r['twonn']          for r in results]
    mles      = [r['mle']            for r in results]
    trusts    = [r['trustworthiness'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: κ と TwoNN の m 依存性（双軸）
    ax = axes[0]
    color1, color2 = 'tab:blue', 'tab:orange'
    ax2_r = ax.twinx()
    l1, = ax.semilogy(m_vals, kappas, 'o-', color=color1, lw=2, ms=6, label='条件数 κ')
    l2, = ax2_r.plot(m_vals, twonns, 's--', color=color2, lw=2, ms=6, label='TwoNN ID')
    ax2_r.axhline(10.0, color='red', ls=':', lw=1.5, label='$\\hat{d}_\\mathrm{ID}\\approx10$')
    ax.set_xlabel('ボトルネック次元 $m$')
    ax.set_ylabel('条件数 κ（対数軸）', color=color1)
    ax2_r.set_ylabel('TwoNN ID 推定値', color=color2)
    ax.tick_params(axis='y', labelcolor=color1)
    ax2_r.tick_params(axis='y', labelcolor=color2)
    ax.set_title('EQ: MNIST AE — κ と TwoNN の $m$ 依存性\n'
                 '($d_\\mathrm{ID}\\approx10$，200ep，seed=42)')
    lines = [l1, l2, plt.Line2D([0],[0], color='red', ls=':', lw=1.5)]
    labels = ['条件数 κ', 'TwoNN ID', '$\\hat{d}_{ID}\\approx10$']
    ax.legend(lines, labels, fontsize=8, loc='upper right')
    ax.axvline(10, color='gray', ls='--', lw=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 右: Trustworthiness と MLE の m 依存性
    ax3 = axes[1]
    ax4_r = ax3.twinx()
    l3, = ax3.plot(m_vals, trusts, 'D-', color='tab:green', lw=2, ms=6, label='Trustworthiness')
    l4, = ax4_r.plot(m_vals, mles, '^--', color='tab:purple', lw=2, ms=6, label='MLE ID')
    ax4_r.axhline(10.0, color='red', ls=':', lw=1.5)
    ax3.set_xlabel('ボトルネック次元 $m$')
    ax3.set_ylabel('Trustworthiness', color='tab:green')
    ax4_r.set_ylabel('MLE ID 推定値', color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax4_r.tick_params(axis='y', labelcolor='tab:purple')
    ax3.set_title('EQ: Trustworthiness と MLE ID の $m$ 依存性')
    lines2 = [l3, l4]
    labels2 = ['Trustworthiness', 'MLE ID']
    ax3.legend(lines2, labels2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EQ_geometry_validity.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EQ_geometry_validity.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EQ_geometry_validity.{png,pdf}")


if __name__ == '__main__':
    results = run_eq()
    make_eq_figure(results)
