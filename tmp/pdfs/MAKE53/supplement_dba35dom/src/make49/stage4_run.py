"""段階 4: E7b と E1、+Q 対照の主比較。

完了条件（方針 §10 段階 4）: 選択結果・品質・費用の比較が揃うこと。

費用の公平性:
  どの方法も「最終評価できる選択済みモデルが得られるまで」に揃える。
  選択後に**選択次元での最終学習（300 epochs 相当）を全方法へ一律に課し**、その費用を計上する。
  FONDUE 系の短時間探索はそのまま尊重し、探索の安さは費用に反映される。

実行: プロジェクト直下で python3 src/make49/stage4_run.py [--dataset MNIST]
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, SPLIT_SEED, provenance
from src.make49.datasets import LOADERS
from src.make49.cache import CandidateCache
from src.make49 import methods as M

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)

SPECS = {'MNIST': {'kind': 'fc', 'input_dim': 784},
         'FashionMNIST': {'kind': 'fc', 'input_dim': 784},
         'dSprites': {'kind': 'conv', 'in_ch': 1, 'size': 64},
         'CIFAR10': {'kind': 'conv', 'in_ch': 3, 'size': 32}}
LOADER_KW = {'MNIST': {'flatten': True}, 'FashionMNIST': {'flatten': True},
             'dSprites': {}, 'CIFAR10': {}}
FONDUE_EPOCHS_PAPER = {'dSprites': 1}     # 原著 Table 1。MNIST は原著になく手順で決める
SEEDS = CFG['data_splits']['training_seeds']
GRIDS = CFG['candidate_grids']


def build_methods(d0, m_refs, fondue_ep):
    """設定表 §5.3 の比較法と +Q 対照。"""
    base_kw = dict(d0=d0, m_refs=m_refs, fondue_epochs=fondue_ep)
    return [
        ('B1_full_mse', M.B1_full_grid_mse, {}),
        ('B2_full_elbo', M.B2_full_grid_elbo, {}),
        ('B3_full_downstream', M.B3_full_grid_downstream, {}),
        ('B4_coarse', M.B4_coarse, {}),
        ('B5_fixed32', M.B5_fixed, {'fixed': 32}),
        ('B6a_fondue', M.B6a_fondue, base_kw),
        ('B6b_fondue_var', M.B6b_fondue_var, dict(base_kw, keep_mixed=True)),
        ('B6b_fondue_var_keepmixed_false', M.B6b_fondue_var, dict(base_kw, keep_mixed=False)),
        ('proposed', M.proposed, base_kw),
        ('proposed_minus_AU', M.proposed, dict(base_kw, record_au=False)),
        ('ascending_scan_Q', M.ascending_scan_Q, {}),
        ('B4_plus_Q', M.with_Q(M.B4_coarse, 'B4'), {}),
        ('B6a_plus_Q', M.with_Q(M.B6a_fondue, 'B6a'), base_kw),
        ('B6b_plus_Q', M.with_Q(M.B6b_fondue_var, 'B6b'), dict(base_kw, keep_mixed=True)),
    ]


def determine_fondue_epochs(cache, grid, d0, seed, limit, max_ep=5):
    """原著の手順: 予測が変わらなくなるまで epoch を増やす（Table 1 の脚注 1）。

    この探索の費用は B6a 本体とは別に記録する（原著は 1 回分の時間のみ報告している）。
    """
    preds, sec = {}, 0.0
    for e in range(1, max_ep + 1):
        led = M.Ledger(cache, seed, limit)
        r = M.B6a_fondue(led, grid, d0=d0, fondue_epochs=e)
        preds[e] = r['selected_m']; sec += led.seconds
        if e >= 2 and preds[e] == preds[e - 1]:
            return {'epochs': e - 1, 'predictions': preds, 'search_seconds': sec,
                    'stable': True}
    return {'epochs': max_ep, 'predictions': preds, 'search_seconds': sec, 'stable': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--betas', default=None, help='カンマ区切り。既定は主設定のみ')
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else ['MNIST', 'dSprites']
    betas = ([float(b) for b in args.betas.split(',')] if args.betas
             else [CFG['beta']['primary']])

    iid = json.load(open(os.path.join(OUT, 'input_space_id.json')))
    prov = provenance()
    out_path = os.path.join(OUT, 'stage4_methods.json')
    allres = json.load(open(out_path))['results'] if os.path.exists(out_path) else []
    for r in allres:
        r.setdefault('beta', CFG['beta']['primary'])
    done = {(r['dataset'], r['method'], r['seed'], r['beta']) for r in allres}

    for beta in betas:
     for ds in datasets:
        grid = GRIDS[ds]
        d0 = iid[ds]['twonn']
        m_refs = [int(round(k * d0)) for k in (4, 8, 16)]
        lik = CFG['likelihood']['by_dataset'][ds]
        limit = CFG['budget_and_termination']['max_wallclock_seconds_per_dataset_method_seed'][ds]
        data = LOADERS[ds](**LOADER_KW[ds])
        rng = np.random.RandomState(SPLIT_SEED + 3)
        n_est = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
        X_est = data['X_train'][rng.choice(len(data['X_train']), n_est, replace=False)]
        cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=beta)

        print(f"\n{'='*78}\n=== {ds}  beta={beta}  尤度={lik}  d̂0={d0:.2f}  "
              f"m_ref={m_refs}  上限={limit}s ===\n{'='*78}", flush=True)

        if ds in FONDUE_EPOCHS_PAPER:
            fe = {'epochs': FONDUE_EPOCHS_PAPER[ds], 'source': '原著 Table 1',
                  'search_seconds': 0.0, 'stable': True}
        else:
            print("  FONDUE の epoch 数を原著の手順で決める（予測が安定するまで増やす）", flush=True)
            fe = determine_fondue_epochs(cache, grid, d0, SEEDS[0], limit)
            fe['source'] = '原著の手順を MNIST に適用（Table 1 に MNIST は無い）'
            print(f"    -> epochs={fe['epochs']} 予測={fe['predictions']} "
                  f"安定={fe['stable']} 探索費用={fe['search_seconds']:.1f}s", flush=True)

        for name, fn, kw in build_methods(d0, m_refs, fe['epochs']):
            for seed in SEEDS:
                if (ds, name, seed, beta) in done:
                    continue
                t0 = time.time()
                led = M.Ledger(cache, seed, limit)
                r = fn(led, grid, X_val=data['X_val'], **kw)
                # 全方法へ一律: 選択次元での最終学習を課して費用に入れる
                sel = r['selected_m']
                final = None
                # 選択次元が 0 以下だと学習できない。Algorithm 3 は keep_mixed=False のとき
                # active 数 0 を返しうる（原著どおりの挙動）。落とさず退化として記録する。
                if sel is not None and int(sel) < 1:
                    r['termination_state'] = 'DEGENERATE_SELECTION'
                    r['degenerate_reason'] = f'選択次元 {sel} は学習可能な次元ではない'
                    sel = None
                if sel is not None:
                    final_run = led.vae(int(sel))
                    final = {'val': final_run['val'], 'test': final_run['test'],
                             'au': final_run['readouts']['B']['au'],
                             'epochs_run': final_run['epochs_run'],
                             'plateau': final_run['plateau']}
                    r['total_seconds'] = led.seconds
                    r['training_runs'] = led.runs
                    r['budget_exhausted'] = led.exhausted
                    if led.exhausted:
                        r['termination_state'] = 'BUDGET_EXHAUSTED'
                rec = {'dataset': ds, 'method': name, 'seed': seed, 'beta': beta,
                       'fondue_epochs_info': fe if name.startswith('B6') else None,
                       'final_model': final, 'wallclock_of_this_call': time.time() - t0, **r}
                allres.append(rec)
                q = final['val']['mse'] if final else float('nan')
                print(f"  {name:32s} s={seed:<6d} m={str(sel):>6s} "
                      f"val_MSE={q:.5f} runs={r['training_runs']:2d} "
                      f"{r['total_seconds']:7.1f}s {r['termination_state']}", flush=True)
                with open(out_path, 'w') as f:
                    json.dump({'_provenance': prov, 'results': allres}, f, ensure_ascii=False)
    print(f"\n完了: {len(allres)} 件 -> {out_path}")


if __name__ == '__main__':
    main()
