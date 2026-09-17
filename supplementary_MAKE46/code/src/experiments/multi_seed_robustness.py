"""
複数シードによるロバスト性検証実験
E1, EA, EC の主要指標を3シードで実行し，mean±std を計算する
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.ae import AutoEncoder, train_ae
from src.models.vae import VAE, train_vae
from src.models.cae import ContractiveAE, train_cae
from src.models.dae import DenoisingAE, train_dae
from src.models.isometric_ae import IsometricAE, train_isometric_ae
from src.metrics.intrinsic_dim import twonn_estimate
from src.metrics.structure import active_units, trustworthiness
from src.metrics.jacobian import compute_decoder_jacobian, analyze_singular_values

SEEDS = [42, 123, 456]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/tables', exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dataloader(X, batch_size=64, train=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def evaluate_mse(model, X_test):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        if isinstance(model, IsometricAE):
            recon, _ = model(X_tensor)
        elif isinstance(model, VAE):
            recon, _, _ = model(X_tensor)
        else:
            recon = model(X_tensor)
        mse = nn.MSELoss()(recon, X_tensor).item()
    return mse


def get_latent_mu(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if isinstance(model, VAE):
            mu, _ = model.encode(X_tensor)
        else:
            mu = model.encode(X_tensor)
    return mu.cpu().numpy()


# =====================================================================
# E1: MSE肘点確認（Swiss Roll と Torus，m=1〜4）
# =====================================================================
def run_e1_single(manifold_name, X_train, X_test, m_values, epochs=200):
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64)
    results = {'mse': [], 'au': [], 'id_twonn': []}
    for m in m_values:
        model = AutoEncoder(input_dim=input_dim, latent_dim=m, hidden_dims=[64, 32]).to(device)
        train_ae(model, train_loader, epochs=epochs, lr=1e-3, device=device)
        mse = evaluate_mse(model, X_test)
        Z = get_latent_mu(model, X_test)
        au = int(active_units(Z))
        id_val = float(twonn_estimate(Z))
        results['mse'].append(mse)
        results['au'].append(au)
        results['id_twonn'].append(id_val)
        print(f"  m={m}: MSE={mse:.4f}, AU={au}, TwoNN={id_val:.2f}")
    return results


def run_e1_multiseed():
    print("\n" + "="*60)
    print("E1: MSE肘点確認 (複数シード)")
    print("="*60)
    m_values = [1, 2, 3, 4]
    manifolds = [
        ('SwissRoll', generate_swiss_roll, 4000, 1000),
        ('Torus', generate_torus, 4000, 1000),
    ]
    all_results = {}
    for name, gen_func, n_train, n_test in manifolds:
        seed_results = {m: {'mse': [], 'au': [], 'id_twonn': []} for m in m_values}
        for seed in SEEDS:
            print(f"\n  Seed={seed}, {name}")
            set_seed(seed)
            X_all, _, _ = gen_func(n_train + n_test, seed=seed)
            X_train, X_test = X_all[:n_train], X_all[n_train:]
            r = run_e1_single(name, X_train, X_test, m_values, epochs=200)
            for i, m in enumerate(m_values):
                seed_results[m]['mse'].append(r['mse'][i])
                seed_results[m]['au'].append(r['au'][i])
                seed_results[m]['id_twonn'].append(r['id_twonn'][i])
        # Compute stats
        stats = {}
        for m in m_values:
            stats[m] = {
                'mse_mean': float(np.mean(seed_results[m]['mse'])),
                'mse_std': float(np.std(seed_results[m]['mse'])),
                'au_mean': float(np.mean(seed_results[m]['au'])),
                'au_std': float(np.std(seed_results[m]['au'])),
                'id_mean': float(np.mean(seed_results[m]['id_twonn'])),
                'id_std': float(np.std(seed_results[m]['id_twonn'])),
            }
        all_results[name] = stats
        print(f"\n  {name} 結果まとめ:")
        print(f"  {'m':>4} {'MSE(mean±std)':>20} {'AU':>10} {'TwoNN ID':>16}")
        for m in m_values:
            s = stats[m]
            print(f"  {m:>4} {s['mse_mean']:.4f}±{s['mse_std']:.4f}  "
                  f"{s['au_mean']:.1f}±{s['au_std']:.2f}  "
                  f"{s['id_mean']:.2f}±{s['id_std']:.2f}")
    with open('results/tables/E1_multiseed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nE1 結果を results/tables/E1_multiseed.json に保存しました")
    return all_results


# =====================================================================
# EA: AU飽和確認（VAE, Swiss Roll と Torus，m=1,2,3,4,6,8,10,12,16）
# =====================================================================
def run_ea_single(X_train, X_test, m_values, beta=4.0, epochs=200):
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64)
    vae_aus = []
    vae_mses = []
    for m in m_values:
        model = VAE(input_dim=input_dim, latent_dim=m, hidden_dims=[64, 32]).to(device)
        train_vae(model, train_loader, epochs=epochs, lr=1e-3, beta=beta, device=device)
        mse = evaluate_mse(model, X_test)
        mu = get_latent_mu(model, X_test)
        au = int(active_units(mu))
        vae_aus.append(au)
        vae_mses.append(mse)
        print(f"    m={m}: VAE AU={au}, MSE={mse:.4f}")
    return {'au': vae_aus, 'mse': vae_mses}


def run_ea_multiseed():
    print("\n" + "="*60)
    print("EA: AU飽和確認 (複数シード)")
    print("="*60)
    m_values = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    manifolds = [
        ('SwissRoll', generate_swiss_roll, 4000, 1000),
        ('Torus', generate_torus, 4000, 1000),
    ]
    all_results = {}
    for name, gen_func, n_train, n_test in manifolds:
        seed_results = {m: {'au': [], 'mse': []} for m in m_values}
        for seed in SEEDS:
            print(f"\n  Seed={seed}, {name}")
            set_seed(seed)
            X_all, _, _ = gen_func(n_train + n_test, seed=seed)
            X_train, X_test = X_all[:n_train], X_all[n_train:]
            r = run_ea_single(X_train, X_test, m_values, epochs=200)
            for i, m in enumerate(m_values):
                seed_results[m]['au'].append(r['au'][i])
                seed_results[m]['mse'].append(r['mse'][i])
        stats = {}
        for m in m_values:
            stats[m] = {
                'au_mean': float(np.mean(seed_results[m]['au'])),
                'au_std': float(np.std(seed_results[m]['au'])),
                'mse_mean': float(np.mean(seed_results[m]['mse'])),
                'mse_std': float(np.std(seed_results[m]['mse'])),
            }
        all_results[name] = stats
        print(f"\n  {name} 結果まとめ:")
        print(f"  {'m':>4} {'AU(mean±std)':>15} {'MSE(mean±std)':>20}")
        for m in m_values:
            s = stats[m]
            print(f"  {m:>4} {s['au_mean']:.1f}±{s['au_std']:.2f}  "
                  f"{s['mse_mean']:.4f}±{s['mse_std']:.4f}")
    with open('results/tables/EA_multiseed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nEA 結果を results/tables/EA_multiseed.json に保存しました")
    return all_results


# =====================================================================
# EC: IsometricAE 比較（Swiss Roll, m=2, 4モデル）
# =====================================================================
def compute_kappa(model, Z, n_eval=50):
    """平均条件数を計算する（原EC実験スクリプトと同じパターン）"""
    n_eval = min(n_eval, len(Z))
    idx = np.random.choice(len(Z), n_eval, replace=False)
    kappas = []
    model.eval()
    for i in idx:
        z_t = torch.tensor(Z[i], dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(model.decoder, z_t)
        J_np = J.cpu().numpy()
        _, _, kappa = analyze_singular_values(J_np, threshold_ratio=0.01)
        kappas.append(kappa)
    return float(np.mean(kappas))


def run_ec_single(X_train, X_test, epochs=200):
    input_dim = X_train.shape[1]
    train_loader = make_dataloader(X_train, batch_size=64)
    results = {}

    # AE
    ae = AutoEncoder(input_dim=input_dim, latent_dim=2, hidden_dims=[64, 32]).to(device)
    train_ae(ae, train_loader, epochs=epochs, lr=1e-3, device=device)
    mse_ae = evaluate_mse(ae, X_test)
    Z_ae = get_latent_mu(ae, X_test)
    kappa_ae = compute_kappa(ae, Z_ae)
    trust_ae = float(trustworthiness(X_test, Z_ae, k=10))
    results['AE'] = {'mse': mse_ae, 'kappa': kappa_ae, 'trust': trust_ae}
    print(f"    AE: MSE={mse_ae:.4f}, κ={kappa_ae:.3f}")

    # CAE
    cae = ContractiveAE(input_dim=input_dim, latent_dim=2, hidden_dims=[64, 32]).to(device)
    train_cae(cae, train_loader, epochs=epochs, lr=1e-3, device=device)
    mse_cae = evaluate_mse(cae, X_test)
    Z_cae = get_latent_mu(cae, X_test)
    kappa_cae = compute_kappa(cae, Z_cae)
    trust_cae = float(trustworthiness(X_test, Z_cae, k=10))
    results['CAE'] = {'mse': mse_cae, 'kappa': kappa_cae, 'trust': trust_cae}
    print(f"    CAE: MSE={mse_cae:.4f}, κ={kappa_cae:.3f}")

    # IsometricAE
    iso = IsometricAE(input_dim=input_dim, latent_dim=2, hidden_dims=[64, 32], lambda_iso=0.01).to(device)
    train_isometric_ae(iso, train_loader, epochs=epochs, lr=1e-3, device=device)
    mse_iso = evaluate_mse(iso, X_test)
    Z_iso = get_latent_mu(iso, X_test)
    kappa_iso = compute_kappa(iso, Z_iso)
    trust_iso = float(trustworthiness(X_test, Z_iso, k=10))
    results['IsometricAE'] = {'mse': mse_iso, 'kappa': kappa_iso, 'trust': trust_iso}
    print(f"    IsometricAE: MSE={mse_iso:.4f}, κ={kappa_iso:.3f}")

    # DAE
    dae = DenoisingAE(input_dim=input_dim, latent_dim=2, hidden_dims=[64, 32]).to(device)
    train_dae(dae, train_loader, epochs=epochs, lr=1e-3, device=device)
    mse_dae = evaluate_mse(dae, X_test)
    Z_dae = get_latent_mu(dae, X_test)
    kappa_dae = compute_kappa(dae, Z_dae)
    trust_dae = float(trustworthiness(X_test, Z_dae, k=10))
    results['DAE'] = {'mse': mse_dae, 'kappa': kappa_dae, 'trust': trust_dae}
    print(f"    DAE: MSE={mse_dae:.4f}, κ={kappa_dae:.3f}")

    return results


def run_ec_multiseed():
    print("\n" + "="*60)
    print("EC: IsometricAE 比較 (複数シード)")
    print("="*60)
    models = ['AE', 'CAE', 'IsometricAE', 'DAE']
    seed_results = {m: {'mse': [], 'kappa': [], 'trust': []} for m in models}
    for seed in SEEDS:
        print(f"\n  Seed={seed}")
        set_seed(seed)
        X_all, _, _ = generate_swiss_roll(5000, seed=seed)
        X_train, X_test = X_all[:4000], X_all[4000:]
        r = run_ec_single(X_train, X_test, epochs=200)
        for m in models:
            seed_results[m]['mse'].append(r[m]['mse'])
            seed_results[m]['kappa'].append(r[m]['kappa'])
            seed_results[m]['trust'].append(r[m]['trust'])
    stats = {}
    for m in models:
        stats[m] = {
            'mse_mean': float(np.mean(seed_results[m]['mse'])),
            'mse_std': float(np.std(seed_results[m]['mse'])),
            'kappa_mean': float(np.mean(seed_results[m]['kappa'])),
            'kappa_std': float(np.std(seed_results[m]['kappa'])),
            'trust_mean': float(np.mean(seed_results[m]['trust'])),
            'trust_std': float(np.std(seed_results[m]['trust'])),
        }
    print(f"\n  EC 結果まとめ:")
    print(f"  {'Model':>12} {'MSE(mean±std)':>22} {'κ(mean±std)':>18} {'Trust':>10}")
    for m in models:
        s = stats[m]
        print(f"  {m:>12} {s['mse_mean']:.4f}±{s['mse_std']:.4f}  "
              f"{s['kappa_mean']:.3f}±{s['kappa_std']:.3f}  "
              f"{s['trust_mean']:.4f}±{s['trust_std']:.4f}")
    with open('results/tables/EC_multiseed.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nEC 結果を results/tables/EC_multiseed.json に保存しました")
    return stats


# =====================================================================
# メイン
# =====================================================================
if __name__ == '__main__':
    print("複数シードによるロバスト性検証実験を開始します")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {device}")

    e1_results = run_e1_multiseed()
    ea_results = run_ea_multiseed()
    ec_results = run_ec_multiseed()

    print("\n" + "="*60)
    print("全実験完了")
    print("="*60)
    print("結果ファイル:")
    print("  results/tables/E1_multiseed.json")
    print("  results/tables/EA_multiseed.json")
    print("  results/tables/EC_multiseed.json")
