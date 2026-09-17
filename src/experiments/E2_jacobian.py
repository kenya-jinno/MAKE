"""
実験 E2: ヤコビアン特異値スペクトル解析
対応定理: 定理2（定数階数・接空間）, 定理3（プルバック計量）
検証仮説: m=d_trueのAEのデコーダヤコビアンはd_true個の大きな特異値を持つ
"""

import sys
import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder, train_ae
from src.metrics.jacobian import (
    compute_decoder_jacobian,
    analyze_singular_values,
    pullback_metric,
    tangent_space_approximation_error
)

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


def load_or_train_model(manifold_name: str, m: int, input_dim: int,
                        X_train: np.ndarray, epochs: int = 100) -> AutoEncoder:
    """E1 の学習済みモデルを読み込む。なければ再訓練する。"""
    save_path = f'results/E1_{manifold_name}_m{m}.pth'
    model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[256, 128])

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"  Loaded model from {save_path}")
    else:
        print(f"  Model not found at {save_path}, retraining ...")
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        train_ae(model, train_loader, epochs=epochs, lr=1e-3, device=device)
        torch.save(model.state_dict(), save_path)

    return model.to(device)


def analyze_jacobians(model: AutoEncoder, X_test: np.ndarray,
                      n_points: int = 50) -> dict:
    """
    テスト点でのデコーダヤコビアンを解析する。

    Returns:
        dict: 特異値スペクトル、有意特異値数、条件数の辞書
    """
    model.eval()
    idx = np.random.choice(len(X_test), min(n_points, len(X_test)), replace=False)

    all_svs = []
    all_n_sig = []
    all_cond = []

    for i in idx:
        x = torch.tensor(X_test[i], dtype=torch.float32).to(device)
        with torch.no_grad():
            z = model.encode(x.unsqueeze(0)).squeeze(0)

        J = compute_decoder_jacobian(model.decoder, z)
        J_np = J.cpu().numpy()
        sv, n_sig, cond = analyze_singular_values(J_np, threshold_ratio=0.1)
        all_svs.append(sv)
        all_n_sig.append(n_sig)
        all_cond.append(cond)

    return {
        'singular_values': all_svs,
        'n_significant': all_n_sig,
        'condition_number': all_cond,
        'mean_n_significant': float(np.mean(all_n_sig)),
        'mean_condition_number': float(np.mean(all_cond))
    }


def plot_sv_spectra(manifold_name: str, d_true: int,
                   results_by_m: dict, m_values: list):
    """特異値スペクトルを比較プロットする。"""
    n_m = len(m_values)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5))
    if n_m == 1:
        axes = [axes]

    for ax, m in zip(axes, m_values):
        sv_list = results_by_m[m]['singular_values']
        # 平均特異値スペクトルを計算
        max_len = max(len(sv) for sv in sv_list)
        sv_padded = np.zeros((len(sv_list), max_len))
        for i, sv in enumerate(sv_list):
            sv_padded[i, :len(sv)] = sv
        mean_sv = np.mean(sv_padded, axis=0)
        std_sv = np.std(sv_padded, axis=0)

        x_idx = np.arange(1, max_len + 1)
        ax.bar(x_idx, mean_sv, color='steelblue', alpha=0.7, label='Mean SV')
        ax.errorbar(x_idx, mean_sv, yerr=std_sv, fmt='none', color='navy', capsize=3)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
        ax.axvline(x=d_true + 0.5, color='green', linestyle='--', alpha=0.7,
                   label=f'd_true={d_true}')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Normalized Singular Value')
        ax.set_title(f'{manifold_name}: m={m}\n'
                     f'Mean n_sig={results_by_m[m]["mean_n_significant"]:.1f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'E2_{manifold_name}_sv_spectra'
    plt.savefig(f'results/figures/{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'results/figures/{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/figures/{fname}.pdf, .png")


def plot_tsa_errors(manifold_name: str, d_true: int,
                    m_values: list, tsa_errors: list):
    """TSA 誤差を m ごとにプロットする。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m_values, tsa_errors, 'o-', color='steelblue', linewidth=2)
    ax.axvline(x=d_true, color='red', linestyle='--', label=f'd_true={d_true}')
    ax.set_xlabel('Bottleneck dim m')
    ax.set_ylabel('Mean TSA Error [degrees]')
    ax.set_title(f'{manifold_name}: Tangent Space Approximation Error vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(m_values)
    plt.tight_layout()
    fname = f'E2_{manifold_name}_tsa_error'
    plt.savefig(f'results/figures/{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'results/figures/{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/figures/{fname}.pdf, .png")


def run_jacobian_analysis(manifold_name: str, X_train: np.ndarray,
                          X_test: np.ndarray, d_true: int) -> dict:
    """ヤコビアン解析を実行する。"""
    input_dim = X_train.shape[1]
    m_values = list(range(1, 2 * d_true + 1))

    print(f"\n=== E2: Jacobian Analysis — {manifold_name} (d_true={d_true}) ===")

    results = {
        'manifold': manifold_name,
        'd_true': d_true,
        'm_values': m_values,
        'jacobian_analysis': {},
        'tsa_errors': []
    }

    tsa_errors = []

    for m in m_values:
        print(f"\n  m={m}:")
        model = load_or_train_model(manifold_name, m, input_dim, X_train, epochs=100)

        # ヤコビアン解析
        jac_results = analyze_jacobians(model, X_test, n_points=50)
        results['jacobian_analysis'][str(m)] = {
            'mean_n_significant': jac_results['mean_n_significant'],
            'mean_condition_number': jac_results['mean_condition_number']
        }
        print(f"    Mean n_significant: {jac_results['mean_n_significant']:.2f}")
        print(f"    Mean condition number: {jac_results['mean_condition_number']:.4f}")

        # TSA 誤差
        print(f"    Computing TSA error ...")
        try:
            tsa_err = tangent_space_approximation_error(
                model.decoder, model.encoder, X_test, d_true, k=20, device=device
            )
            tsa_errors.append(float(tsa_err) if not np.isnan(tsa_err) else -1.0)
            print(f"    TSA error: {tsa_err:.2f} degrees")
        except Exception as e:
            print(f"    TSA error computation failed: {e}")
            tsa_errors.append(-1.0)

    results['tsa_errors'] = tsa_errors

    # 特異値スペクトルをロードして可視化
    jac_by_m = {}
    for m in m_values:
        model = load_or_train_model(manifold_name, m, input_dim, X_train, epochs=100)
        jac_results = analyze_jacobians(model, X_test, n_points=50)
        jac_by_m[m] = jac_results

    plot_sv_spectra(manifold_name, d_true, jac_by_m, m_values)

    valid_tsa = [(m, e) for m, e in zip(m_values, tsa_errors) if e >= 0]
    if valid_tsa:
        m_vals_valid = [x[0] for x in valid_tsa]
        tsa_vals_valid = [x[1] for x in valid_tsa]
        plot_tsa_errors(manifold_name, d_true, m_vals_valid, tsa_vals_valid)

    return results


def main():
    """実験 E2 のメイン関数。"""
    n_train = 3000
    n_test = 1000
    all_results = []

    # Swiss Roll
    print("Generating Swiss Roll data ...")
    X_train_sr, d_true_sr, _ = generate_swiss_roll(n_train, noise=0.01, seed=SEED)
    X_test_sr, _, T_basis_sr = generate_swiss_roll(n_test, noise=0.01, seed=SEED + 1)
    res_sr = run_jacobian_analysis('SwissRoll', X_train_sr, X_test_sr, d_true_sr)
    all_results.append(res_sr)

    # Torus
    print("\nGenerating Torus data ...")
    X_train_tor, d_true_tor, _ = generate_torus(n_train, R=3.0, r=1.0, noise=0.01, seed=SEED)
    X_test_tor, _, T_basis_tor = generate_torus(n_test, R=3.0, r=1.0, noise=0.01, seed=SEED + 1)
    res_tor = run_jacobian_analysis('Torus', X_train_tor, X_test_tor, d_true_tor)
    all_results.append(res_tor)

    # 結果保存
    save_data = []
    for res in all_results:
        save_data.append(res)

    with open('results/tables/E2_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved: results/tables/E2_results.json")

    # サマリー
    print("\n=== E2 Summary ===")
    for res in all_results:
        print(f"\n{res['manifold']} (d_true={res['d_true']}):")
        for m in res['m_values']:
            jac = res['jacobian_analysis'][str(m)]
            print(f"  m={m}: n_sig={jac['mean_n_significant']:.2f}, "
                  f"cond={jac['mean_condition_number']:.4f}")
        print(f"  TSA errors: {res['tsa_errors']}")


if __name__ == '__main__':
    main()
