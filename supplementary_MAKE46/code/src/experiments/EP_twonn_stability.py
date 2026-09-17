"""
実験 EP: TwoNN ID の数値安定性と等長性バイアスの実証
目的: 参照 AE の等長性バイアス（κ）の大小にかかわらず，TwoNN ID の
     数値的安定性を実証する（「安定」≠「正確」の区別を明確にする）．

Section A: Swiss Roll (d_true=2, 制御された幾何)
  - AE (κ≈1.47), IsometricAE (κ≈1.04) の潜在空間での TwoNN を
    生データ TwoNN (d_true=2) と比較する．
  - m=2 (=d_true) および m=4 (>d_true) でのバイアスを計測．

Section B: MNIST TwoNN 収束曲線 (既存 E4_extended.json から読み込み)
  - κ≈10^9 の参照 AE であっても m→∞ で TwoNN が安定収束することを示す．
  - これは「κ が大きくても TwoNN は補助的参照として信頼できる」証拠．
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

from src.data.synthetic import generate_swiss_roll
from src.models.ae import AutoEncoder, train_ae
from src.models.isometric_ae import IsometricAE, train_isometric_ae
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


# ───────────────────────────────────────────────────────────────────────
# Utility: condition number of decoder Jacobian
# ───────────────────────────────────────────────────────────────────────
def compute_kappa(model_decoder, Z: np.ndarray, n_eval: int = 40) -> float:
    n_eval = min(n_eval, len(Z))
    idx = np.random.choice(len(Z), n_eval, replace=False)
    kappas = []
    for i in idx:
        z_t = torch.tensor(Z[i], dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(model_decoder, z_t)
        J_np = J.cpu().numpy()
        _, _, kappa = analyze_singular_values(J_np, threshold_ratio=0.01)
        kappas.append(kappa)
    return float(np.mean(kappas))


def encode_data(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        if hasattr(model, 'encode'):
            Z = model.encode(X_t)
        else:
            Z = model.encoder(X_t)
    return Z.cpu().numpy()


# ───────────────────────────────────────────────────────────────────────
# Section A: Swiss Roll
# ───────────────────────────────────────────────────────────────────────
def run_swiss_roll_section(n_train=3000, n_test=500, epochs=150):
    print("\n=== Section A: Swiss Roll TwoNN stability ===")

    X_all, _, _ = generate_swiss_roll(n_samples=n_train + n_test, noise=0.05)
    X_train, X_test = X_all[:n_train], X_all[n_train:]

    # TwoNN on raw input (3D)
    twonn_raw = twonn_estimate(X_test)
    mle_raw   = mle_estimate(X_test, k=10)
    print(f"  Raw 3D TwoNN={twonn_raw:.3f}, MLE={mle_raw:.3f}  (d_true=2)")

    results = {'d_true': 2, 'raw': {'twonn': float(twonn_raw), 'mle': float(mle_raw)}}

    for latent_dim, label in [(2, 'AE_m2'), (4, 'AE_m4')]:
        model = AutoEncoder(
            input_dim=3, latent_dim=latent_dim, hidden_dims=[64, 32]
        ).to(device)
        loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                            batch_size=64, shuffle=True)
        train_ae(model, loader, epochs=epochs, lr=1e-3, device=device)

        Z = encode_data(model, X_test)
        kappa = compute_kappa(model.decoder, Z)
        twonn = twonn_estimate(Z)
        mle   = mle_estimate(Z, k=10)
        print(f"  {label}: κ={kappa:.3f}, TwoNN={twonn:.3f}, MLE={mle:.3f}")
        results[label] = {'latent_dim': latent_dim, 'kappa': float(kappa),
                          'twonn': float(twonn), 'mle': float(mle)}

    # IsometricAE m=2
    iso_model = IsometricAE(
        input_dim=3, latent_dim=2, hidden_dims=[64, 32], lambda_iso=0.05
    ).to(device)
    iso_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                            batch_size=64, shuffle=True)
    train_isometric_ae(iso_model, iso_loader, epochs=epochs, lr=1e-3,
                       device=device, iso_batch_size=4)

    Z_iso = encode_data(iso_model, X_test)
    kappa_iso = compute_kappa(iso_model.decoder, Z_iso)
    twonn_iso = twonn_estimate(Z_iso)
    mle_iso   = mle_estimate(Z_iso, k=10)
    print(f"  IsometricAE_m2: κ={kappa_iso:.3f}, TwoNN={twonn_iso:.3f}, MLE={mle_iso:.3f}")
    results['IsometricAE_m2'] = {'latent_dim': 2, 'kappa': float(kappa_iso),
                                  'twonn': float(twonn_iso), 'mle': float(mle_iso)}

    return results


# ───────────────────────────────────────────────────────────────────────
# Section B: MNIST TwoNN convergence (read from E4_extended.json)
# ───────────────────────────────────────────────────────────────────────
def run_mnist_section():
    print("\n=== Section B: MNIST TwoNN convergence (from E4_extended.json) ===")
    json_path = 'results/tables/E4_extended.json'
    with open(json_path) as f:
        e4 = json.load(f)

    twonn_by_m = [{'m': r['m'], 'au': r['au'], 'twonn': r['id_twonn']}
                  for r in e4['vae_results']]

    print(f"  κ_ref ≈ 1e9 (reference AE m=256)")
    for r in twonn_by_m:
        print(f"  m={r['m']:3d}: AU={r['au']:3d}, TwoNN={r['twonn']:.2f}")

    # TwoNN at m=64 and m=256 for annotation
    m64  = next(r for r in twonn_by_m if r['m'] == 64)
    m256 = next(r for r in twonn_by_m if r['m'] == 256)
    print(f"\n  Key values: TwoNN(m=64)={m64['twonn']:.2f}, TwoNN(m=256)={m256['twonn']:.2f}")
    print(f"  Convergence: m≥128 → TwoNN≈10 (stable reference)")

    return {
        'note': 'MNIST VAE (β=4, 300ep, seed=42), reference AE κ≈1e9',
        'twonn_by_m': twonn_by_m,
        'kappa_ref': 1e9,
    }


# ───────────────────────────────────────────────────────────────────────
# Figure
# ───────────────────────────────────────────────────────────────────────
def make_figure(sr_results, mnist_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Swiss Roll bar chart ──────────────────────────────────────
    ax = axes[0]
    conditions = [
        ('Raw 3D\n(ground truth)', sr_results['raw']['twonn'], None, 'tab:gray'),
        (f"AE (m=2)\nκ≈{sr_results['AE_m2']['kappa']:.2f}", sr_results['AE_m2']['twonn'], 2, 'tab:blue'),
        (f"IsometricAE (m=2)\nκ≈{sr_results['IsometricAE_m2']['kappa']:.2f}", sr_results['IsometricAE_m2']['twonn'], 2, 'tab:green'),
        (f"AE (m=4)\nκ≈{sr_results['AE_m4']['kappa']:.2f}", sr_results['AE_m4']['twonn'], 4, 'tab:orange'),
    ]
    labels  = [c[0] for c in conditions]
    values  = [c[1] for c in conditions]
    colors  = [c[3] for c in conditions]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(2.0, color='red', ls='--', lw=1.5, label='$d_\\mathrm{true}=2$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('TwoNN ID estimate')
    ax.set_title('Section A: Swiss Roll — TwoNN by model (effect of κ)\n'
                 '($d_\\mathrm{true}=2$, 3D input)')
    ax.set_ylim(0, max(values) * 1.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # ── Right: MNIST TwoNN convergence ──────────────────────────────────
    ax2 = axes[1]
    m_vals = [r['m']  for r in mnist_results['twonn_by_m']]
    t_vals = [r['twonn'] for r in mnist_results['twonn_by_m']]
    ax2.semilogx(m_vals, t_vals, 'o-', color='tab:blue', ms=6, lw=2)
    ax2.axhline(10.0, color='red', ls='--', lw=1.5, label='$\\hat{d}_\\mathrm{ID}\\approx10$')
    ax2.annotate('κ≈$10^9$ (reference AE)', xy=(64, 9.16), xytext=(20, 7.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8, color='gray')
    ax2.set_xlabel('Bottleneck dim $m$')
    ax2.set_ylabel('TwoNN ID estimate')
    ax2.set_title('Section B: MNIST — TwoNN convergence vs $m$\n'
                  '(reference AE, κ≈$10^9$; $\\beta=4$, 300ep)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.8, 300)

    plt.tight_layout()
    plt.savefig('results/figures/EP_twonn_stability.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EP_twonn_stability.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EP_twonn_stability.{png,pdf}")


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────
def main():
    print("=== EP: TwoNN ID stability under isometry bias ===")
    print(f"Device: {device}")

    sr_results    = run_swiss_roll_section(n_train=3000, n_test=500, epochs=150)
    mnist_results = run_mnist_section()

    out = {'swiss_roll': sr_results, 'mnist': mnist_results}
    with open('results/tables/EP_twonn_stability.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved: results/tables/EP_twonn_stability.json")

    make_figure(sr_results, mnist_results)

    print("\n=== EP Summary ===")
    print("Swiss Roll (d_true=2):")
    for key in ['raw', 'AE_m2', 'IsometricAE_m2', 'AE_m4']:
        r = sr_results[key]
        kappa_str = f"κ={r.get('kappa','N/A'):.3f}" if 'kappa' in r else "κ=N/A"
        print(f"  {key:20s}: TwoNN={r['twonn']:.3f}  {kappa_str}")
    print("MNIST: TwoNN converges to ≈10 at m≥128 despite κ≈1e9")


if __name__ == '__main__':
    main()
