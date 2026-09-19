"""B7（GECO+L0）と B8（ARD-VAE）を主比較に加える。

**両手法は beta を持たない。** B7 は beta を再構成制約 tau に、
B8 は階層事前に置き換える設計だからである。
ただし B7 の tau は品質目標なので、**beta 梯子の各段の anchor から導いた T_D** を用いる。
これにより各段で「同じ品質目標のもとでの比較」になる（方針 §5.3）。

費用の公平性: 他の方法と同様、**選択後に選択次元での最終学習を一律に課す**。

実行: プロジェクト直下で python3 src/make49/stage7_run_b7b8.py [--betas 1] [--datasets ...]
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, SPLIT_SEED, provenance
from src.make49.datasets import LOADERS
from src.make49.cache import CandidateCache
from src.make49.pruning_methods import run_B7_geco_l0, run_B8_ard

OUT = os.path.join('results', 'make49')
SPECS = {'MNIST': {'kind': 'fc', 'input_dim': 784},
         'FashionMNIST': {'kind': 'fc', 'input_dim': 784},
         'dSprites': {'kind': 'conv', 'in_ch': 1, 'size': 64},
         'CIFAR10': {'kind': 'conv', 'in_ch': 3, 'size': 32}}
KW = {'MNIST': {'flatten': True}, 'FashionMNIST': {'flatten': True},
      'dSprites': {}, 'CIFAR10': {}}
SEEDS = CFG['data_splits']['training_seeds']
REL = CFG['quality_criterion_Q']['epsilon_D']['rel_primary']
MAX_EPOCHS = CFG['budget_and_termination']['max_epochs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='MNIST,FashionMNIST,dSprites,CIFAR10')
    ap.add_argument('--betas', default='1')
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(',')]

    path = os.path.join(OUT, 'stage7_b7b8.json')
    res = json.load(open(path))['results'] if os.path.exists(path) else []
    done = {(r['dataset'], r['method'], r['seed'], r['beta_context']) for r in res}
    prov = provenance()

    for ds in args.datasets.split(','):
        grid = CFG['candidate_grids'][ds]
        lik = CFG['likelihood']['by_dataset'][ds]
        data = LOADERS[ds](**KW[ds])
        rng = np.random.RandomState(SPLIT_SEED + 3)
        n = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
        X_est = data['X_train'][rng.choice(len(data['X_train']), n, replace=False)]
        anchor = grid[-1]

        for beta in betas:
            cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=beta)
            print(f"\n=== {ds}  beta 文脈={beta}  尤度={lik}  初期容量={anchor} ===", flush=True)

            for seed in SEEDS:
                # tau は当該 beta の anchor から導いた品質目標 T_D
                a = cache.get_vae(anchor, seed)
                tau = a['val']['mse'] * (1 + REL)

                for name, fn, kw in [
                        ('B7_geco_l0', run_B7_geco_l0, {'tau': tau}),
                        ('B8_ard_vae', run_B8_ard, {})]:
                    if (ds, name, seed, beta) in done:
                        continue
                    t0 = time.time()
                    r = fn(data, SPECS[ds], lik, m_start=anchor, seed=seed,
                           epochs=MAX_EPOCHS, **kw)
                    sel = r['selected_m']
                    final = None
                    state = 'SUCCESS'
                    if sel is None or sel < 1:
                        state = 'DEGENERATE_SELECTION'
                    else:
                        m_grid = min(grid, key=lambda g: abs(g - sel))
                        fr = cache.get_vae(int(m_grid), seed)  # 最終学習（全方法に一律）
                        final = {'m_on_grid': int(m_grid), 'val': fr['val'],
                                 'test': fr['test'], 'au': fr['readouts']['B']['au']}
                        r['final_train_seconds'] = fr['elapsed_seconds']
                    total = r['elapsed_seconds'] + (r.get('final_train_seconds', 0.0))
                    rec = {'dataset': ds, 'method': name, 'seed': seed,
                           'beta_context': beta, 'tau': kw.get('tau'),
                           'selected_m': sel, 'final_model': final,
                           'termination_state': state,
                           'total_seconds': total, 'training_runs': 2,
                           'auxiliary_runs': 0,
                           'wallclock_of_this_call': time.time() - t0, **r}
                    res.append(rec)
                    q = final['val']['mse'] if final else float('nan')
                    print(f"  {name:12s} s={seed:<6d} m={str(sel):>5s} "
                          f"(格子上 {final['m_on_grid'] if final else '-'}) "
                          f"val_MSE={q:.5f} {total:7.1f}s {state}", flush=True)
                    json.dump({'_provenance': prov,
                               '_spec': {'note': 'B7/B8 は beta を持たない。'
                                                 'beta_context は tau の導出元の段を示す',
                                         'tau_rule': f'anchor の val MSE x (1+{REL})',
                                         'B7_deviation': 'L0-ARM ではなく hard concrete 緩和',
                                         'B8_deviation': 'Student-t ではなく N(0, sigma_hat^2) 近似'},
                               'results': res},
                              open(path, 'w'), ensure_ascii=False)
    print(f"\n完了: {len(res)} 件 -> {path}")


if __name__ == '__main__':
    main()
