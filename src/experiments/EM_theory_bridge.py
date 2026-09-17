"""
実験 EM: 理論-実験橋渡し分析
目的: 定理1（線形β-VAEのAU飽和しきい値 λ_j > βτ²）と
     非線形MNIST VAE実験（E4）の定性的対応を定量的に示す
内容:
  (a) MNIST学習データのPCA固有値スペクトル → 線形理論の予測 d*(β=4, τ²) を計算
  (b) 実験 E4（300ep, 3seed）のAU(m)曲線との比較
  (c) 線形理論が継承される側面・崩れる側面を整理した比較表と図の生成
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision
import torchvision.transforms as transforms
import torch

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def load_mnist_numpy(n_train=20000):
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True,
        transform=transforms.ToTensor())
    X = train_ds.data[:n_train].float().view(-1, 784).numpy() / 255.0
    return X


def compute_mknee(results, tau=0.25):
    ms  = [r['m']  for r in results]
    aus = [r['au'] for r in results]
    for i in range(1, len(ms)):
        dm  = ms[i] - ms[i-1]
        dau = aus[i] - aus[i-1]
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def main():
    print("=== EM: Theory-Experiment Bridge Analysis ===\n")

    # ── (a) MNIST PCA 固有値スペクトル ───────────────────────────────────────
    print("Loading MNIST and computing PCA spectrum ...")
    X = load_mnist_numpy(n_train=20000)
    # 中心化
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=100)
    pca.fit(X_centered)
    eigenvalues = pca.explained_variance_  # λ_j (分散値, j=1,...,100)
    ev_ratio    = pca.explained_variance_ratio_

    print(f"Top 20 eigenvalues (PCA variance): {eigenvalues[:20].round(3)}")
    print(f"Cumulative variance ratio at k=10: {ev_ratio[:10].sum():.3f}")
    print(f"Cumulative variance ratio at k=20: {ev_ratio[:20].sum():.3f}")
    print(f"Cumulative variance ratio at k=43: {ev_ratio[:43].sum():.3f}")

    # ── (b) 線形理論の予測：d*(β=4, τ²) ─────────────────────────────────────
    # β-VAE のデコーダノイズ τ² は通常「標準化データの復元誤差スケール」に対応する
    # MNISTは[0,1]正規化，データ分散 ≈ 0.13（ピクセル平均で）
    # 線形 VAE の τ² は hyperparameter で，β-ELBO では τ²=1（入力空間の標準ノイズスケール）が
    # よく使われる default だが，より自然な選択は「再構成誤差スケール」τ² ≈ MSE at optimal m

    # 複数の τ² 設定での d* 予測
    beta = 4.0
    tau2_candidates = {
        'τ²=0.5 (0.5×データ分散目安)': 0.5,
        'τ²=1.0 (標準正規化)': 1.0,
        'τ²=2.0 (緩い設定)': 2.0,
        'τ²=5.0 (高ノイズ)': 5.0,
    }
    print("\n線形理論予測 d*(β=4, τ²) = |{j: λ_j > β·τ²}|:")
    print(f"{'条件':35s} | β·τ² | d*(β,τ²)")
    for label, tau2 in tau2_candidates.items():
        threshold = beta * tau2
        d_star = int(np.sum(eigenvalues > threshold))
        print(f"  {label}: threshold={threshold:.1f}, d*={d_star}")

    # 最も MNIST の観測値（AU≈10〜12）と整合する τ² を特定
    # eigenvalues[9]（10番目）〜 eigenvalues[11]（12番目）の値を確認
    print(f"\nPCA eigenvalue at rank 10: {eigenvalues[9]:.4f}")
    print(f"PCA eigenvalue at rank 12: {eigenvalues[11]:.4f}")
    print(f"PCA eigenvalue at rank 15: {eigenvalues[14]:.4f}")
    print(f"PCA eigenvalue at rank 20: {eigenvalues[19]:.4f}")

    # d*=10 を実現する β·τ²: λ_10 > β·τ² > λ_11 → τ² ∈ (λ_11/β, λ_10/β)
    tau2_for_d10_low  = eigenvalues[10] / beta
    tau2_for_d10_high = eigenvalues[9]  / beta
    print(f"\nd*(β=4)=10 となる τ² の範囲: ({tau2_for_d10_low:.4f}, {tau2_for_d10_high:.4f})")
    print(f"  → β·τ² ∈ ({eigenvalues[10]:.4f}, {eigenvalues[9]:.4f})")

    # ── (c) E4_300ep_multiseed の AU(m) 曲線を読み込み ────────────────────────
    with open('results/tables/E4_300ep_multiseed.json') as f:
        e4_data = json.load(f)
    seeds = [42, 123, 777]

    # 各シードの (m, AU) 取得
    m_sparse = [4, 8, 12, 16, 20, 32, 64]
    au_by_seed = {}
    for s in seeds:
        au_by_seed[s] = [e4_data[str(s)][i]['au'] for i in range(len(m_sparse))]
    au_mean_sparse = np.mean([au_by_seed[s] for s in seeds], axis=0)

    print("\n疎グリッド E4 (300ep, 3seed) AU(m):")
    print(f"{'m':>5} | AU seed42 | AU seed123 | AU seed777 | mean")
    for i, m in enumerate(m_sparse):
        aus = [au_by_seed[s][i] for s in seeds]
        print(f"{m:5d} | {aus[0]:9d} | {aus[1]:10d} | {aus[2]:10d} | {np.mean(aus):.1f}")

    # ── 保存データ ────────────────────────────────────────────────────────────
    bridge_data = {
        'pca_eigenvalues_top100': eigenvalues.tolist(),
        'pca_ev_ratio_top100': ev_ratio.tolist(),
        'linear_theory': {
            'beta': beta,
            'd_star_predictions': {
                f'tau2={tau2}': {
                    'threshold': float(beta * tau2),
                    'd_star': int(np.sum(eigenvalues > beta * tau2))
                }
                for tau2 in [0.5, 1.0, 2.0, 5.0]
            },
            'eigenvalue_rank10': float(eigenvalues[9]),
            'eigenvalue_rank11': float(eigenvalues[10]),
            'eigenvalue_rank12': float(eigenvalues[11]),
            'tau2_for_d_star_10': {
                'lower': float(tau2_for_d10_low),
                'upper': float(tau2_for_d10_high),
            }
        }
    }
    with open('results/tables/EM_theory_bridge.json', 'w') as f:
        json.dump(bridge_data, f, indent=2)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PCA eigenvalue spectrum with theory thresholds
    ax = axes[0]
    ks = np.arange(1, 51)
    ax.semilogy(ks, eigenvalues[:50], 'b-o', ms=4, lw=1.5, label='MNIST PCA eigenvalue $\\lambda_j$')
    # β·τ² 水面線（d*≈10と整合する τ² を使用）
    tau2_show = (tau2_for_d10_low + tau2_for_d10_high) / 2
    water_level = beta * tau2_show
    ax.axhline(water_level, color='red', ls='--', lw=1.5,
               label=f'Water level $\\beta\\tau^2={water_level:.3f}$ ($d^*\\approx10$)')
    ax.axvline(10, color='orange', ls='-.', lw=1.5, alpha=0.8,
               label=f'$d^*(\\beta=4)=10$ (linear theory)')
    ax.fill_betweenx([eigenvalues[49], eigenvalues[0]*1.5],
                     0, 10, alpha=0.1, color='orange')
    ax.set_xlabel('PCA rank $j$')
    ax.set_ylabel('Eigenvalue $\\lambda_j$ (log scale)')
    ax.set_title('MNIST PCA spectrum: 線形理論の水充填予測')
    ax.set_xlim(0, 51)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    # Right: Linear theory prediction vs. nonlinear E4 observation
    ax2 = axes[1]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for s, col in zip(seeds, colors):
        aus = au_by_seed[s]
        ax2.plot(m_sparse, aus, 'o--', color=col, ms=5, alpha=0.7, label=f'Observed AU (seed={s})')
    ax2.plot(m_sparse, au_mean_sparse, 'ks-', ms=6, lw=2, label='Observed AU mean (3 seeds)')

    # 線形理論：m ≤ d* では AU = min(m, d*)，m ≥ d* では AU = d*
    d_star = 10
    m_theory = np.arange(1, 70)
    au_theory = np.minimum(m_theory, d_star).astype(float)
    ax2.plot(m_theory, au_theory, 'r-', lw=2, alpha=0.6, label=f'Linear theory (sharp, $d^*={d_star}$)')
    ax2.axvline(d_star, color='red', ls='-.', lw=1.5, alpha=0.7,
                label=f'$d^*=10$ (linear theory threshold)')

    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('Active Units (AU)')
    ax2.set_title('線形理論 vs. 非線形 MNIST VAE (E4, 300ep)')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 70)
    ax2.set_ylim(0, 30)

    plt.tight_layout()
    plt.savefig('results/figures/EM_theory_bridge.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EM_theory_bridge.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/EM_theory_bridge.{png,pdf}")
    print("Saved: results/tables/EM_theory_bridge.json")


if __name__ == '__main__':
    main()
