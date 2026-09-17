"""
実験 Exp-Syn-Verify: Torus の境界域 AU 挙動の確定（Review_codex36 Major 1 対応）
Torus（d_ID=2, d_emb=3）の VAE (beta=4) を m ∈ {3,4,5,6,8} × 5 シードで学習し，
- m=3,4 の AU=2（過剰刈り込み）が再現性のある真の現象か
- 遷移点（m=5?）でシード間のばらつき（最適化盆地効果）があるか
- 過剰刈り込みが MSE ペナルティとして検出できるか
を per-seed で記録する．設定は Exp-Syn-AU（multi_seed_robustness.run_ea_multiseed）と同一．
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import numpy as np
import torch

from src.data.synthetic import generate_swiss_roll, generate_torus
from src.models.vae import VAE, train_vae
from src.metrics.structure import active_units
from src.experiments.multi_seed_robustness import (
    make_dataloader, evaluate_mse, get_latent_mu, set_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
os.makedirs('results/tables', exist_ok=True)

SEEDS = [42, 123, 777, 2024, 31415]
M_VALUES = [3, 4, 5, 6, 8]


def main():
    out = {}
    for name, gen_func in [('Torus', generate_torus),
                           ('SwissRoll', generate_swiss_roll)]:
        out[name] = {str(m): {'au': [], 'mse': []} for m in M_VALUES}
        for seed in SEEDS:
            set_seed(seed)
            X_all, _, _ = gen_func(5000, seed=seed)
            X_train, X_test = X_all[:4000], X_all[4000:]
            loader = make_dataloader(X_train, batch_size=64)
            for m in M_VALUES:
                model = VAE(input_dim=X_train.shape[1], latent_dim=m,
                            hidden_dims=[64, 32]).to(device)
                train_vae(model, loader, epochs=200, lr=1e-3, beta=4.0,
                          device=device)
                mse = evaluate_mse(model, X_test)
                au = int(active_units(get_latent_mu(model, X_test)))
                out[name][str(m)]['au'].append(au)
                out[name][str(m)]['mse'].append(float(mse))
                print(f"{name} seed={seed} m={m}: AU={au}, MSE={mse:.4f}",
                      flush=True)

    with open('results/tables/Exp_Syn_Verify.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'm_values': M_VALUES,
                   'beta': 4.0, 'epochs': 200, 'results': out}, f, indent=2)

    print("\n=== Summary (AU per seed | MSE mean) ===")
    for name in out:
        print(name)
        for m in M_VALUES:
            aus = out[name][str(m)]['au']
            mses = out[name][str(m)]['mse']
            print(f"  m={m}: AU={aus}, MSE={np.mean(mses):.4f}±{np.std(mses):.4f}")
    print("Saved: results/tables/Exp_Syn_Verify.json")


if __name__ == '__main__':
    main()
