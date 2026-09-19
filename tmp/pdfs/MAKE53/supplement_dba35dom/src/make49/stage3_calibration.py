"""MAKE49 段階 3 校正実行。

`MAKE49_実験設定表.md` §8 の 3 条件で実時間を実測し、
未確定だった「GPU 実時間上限」を決めるための素データを出す。

  1. MNIST FC-VAE,      m=16,  300 epochs, 1 seed
  2. dSprites Conv-VAE,  m=16,  300 epochs, 1 seed
  3. CIFAR-10 Conv-VAE,  m=64,  300 epochs, 1 seed

設定は config/experiment_config_MAKE49.json に従う（beta=1, 損失規約 sum_recon_sum_kl_v1,
early stopping patience 30, plateau 判定 直近50ep/1%）。
実行: プロジェクト直下で python3 src/make49/stage3_calibration.py
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, DEVICE, provenance, set_seed, train_vae, elbo_terms
from src.make49.datasets import LOADERS
from src.make49.models import build
from src.metrics.structure import active_units
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate

OUT_DIR = os.path.join('results', 'make49')
os.makedirs(OUT_DIR, exist_ok=True)

BETA = CFG['beta']['primary']
MAX_EPOCHS = CFG['budget_and_termination']['max_epochs']
PATIENCE = CFG['budget_and_termination']['early_stopping_patience_epochs']
DELTA = CFG['au']['delta_primary']
GRIDS = CFG['candidate_grids']

CONFIGS = [
    {'name': 'MNIST_FC',      'dataset': 'MNIST',    'm': 16,
     'spec': {'kind': 'fc', 'input_dim': 784},           'loader_kw': {'flatten': True}},
    {'name': 'dSprites_Conv', 'dataset': 'dSprites', 'm': 16,
     'spec': {'kind': 'conv', 'in_ch': 1, 'size': 64},   'loader_kw': {}},
    {'name': 'CIFAR10_Conv',  'dataset': 'CIFAR10',  'm': 64,
     'spec': {'kind': 'conv', 'in_ch': 3, 'size': 32},   'loader_kw': {}},
]


def evaluate(model, X, delta):
    """最終評価。test は選択に一切使っていない checkpoint に対してのみ適用する。"""
    model.eval()
    outs = {}
    with torch.no_grad():
        recs, mus = [], []
        for i in range(0, len(X), 512):
            xb = torch.as_tensor(X[i:i + 512]).to(DEVICE)
            xh, mu, lv = model(xb)
            recs.append(((xh - xb) ** 2).mean(dim=tuple(range(1, xb.dim()))).cpu())
            mus.append(mu.cpu())
        outs['mse_per_pixel_mean'] = float(torch.cat(recs).mean())
        Z = torch.cat(mus).numpy()
    outs['au'] = int(active_units(Z, delta))
    outs['twonn'] = float(twonn_estimate(Z))
    outs['mle'] = float(mle_estimate(Z, k=10))
    return outs


def main():
    prov = provenance()
    print('== provenance ==')
    for k, v in prov.items():
        print(f'  {k}: {v}')
    print(f'\n== 設定 ==\n  beta={BETA}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  '
          f'delta={DELTA}  loss={CFG["loss_convention"]["id"]}\n')

    results = []
    for c in CONFIGS:
        print(f"=== {c['name']}  (m={c['m']}) ===", flush=True)
        t_load = time.time()
        data = LOADERS[c['dataset']](**c['loader_kw'])
        load_s = time.time() - t_load
        print(f"  データ {data['X_train'].shape} / val {data['X_val'].shape} "
              f"({load_s:.1f}s, hash={data['split']['index_hash']})", flush=True)

        set_seed(CFG['data_splits']['training_seeds'][0])
        model = build(c['spec'], c['m'])
        n_params = sum(p.numel() for p in model.parameters())

        r = train_vae(model, data['X_train'], data['X_val'], beta=BETA,
                      max_epochs=MAX_EPOCHS, patience=PATIENCE, progress=25)

        ev = evaluate(r['model'], data['X_test'], DELTA)
        per_epoch = r['elapsed_seconds'] / r['epochs_run']
        grid = GRIDS[c['dataset']]
        rec = {
            'name': c['name'], 'dataset': c['dataset'], 'm': c['m'],
            'n_params': n_params,
            'n_train': int(data['X_train'].shape[0]), 'n_val': int(data['X_val'].shape[0]),
            'n_test': int(data['X_test'].shape[0]),
            'index_hash': data['split']['index_hash'],
            'beta': BETA,
            'epochs_run': r['epochs_run'], 'best_epoch': r['best_epoch'],
            'stopped_early': r['stopped_early'], 'plateau': r['plateau'],
            'best_val_elbo': r['best_val_elbo'],
            'elapsed_seconds': r['elapsed_seconds'],
            'update_steps': r['update_steps'],
            'seconds_per_epoch': per_epoch,
            'data_load_seconds': load_s,
            'projected_300ep_seconds': per_epoch * MAX_EPOCHS,
            'grid_size': len(grid),
            'grid': grid,
            'test_eval': ev,
            # 校正ランは選択を伴わないので、設定表 §1.6 の終了状態は適用されない
            'run_type': 'CALIBRATION_RUN',
            'termination_state': None,
            'provenance': prov,
            'history': r['history'],
        }
        results.append(rec)
        print(f"  実測 {r['elapsed_seconds']:.1f}s / {r['epochs_run']}ep "
              f"= {per_epoch:.3f}s/ep  (300ep 換算 {per_epoch*MAX_EPOCHS/60:.1f}分)")
        print(f"  best_epoch={r['best_epoch']}  early_stop={r['stopped_early']}  "
              f"plateau={r['plateau']}")
        print(f"  test: MSE={ev['mse_per_pixel_mean']:.6f}  AU={ev['au']}/{c['m']}  "
              f"TwoNN={ev['twonn']:.2f}  MLE={ev['mle']:.2f}\n", flush=True)

        with open(os.path.join(OUT_DIR, 'stage3_calibration.json'), 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    print('=== 予算の導出（グリッド全点 × 1 seed、早期停止なしの最悪値）===')
    print(f"{'条件':16s} {'s/ep':>8} {'300ep':>9} {'グリッド':>7} {'全点300ep':>11} {'推奨上限':>11}")
    budget = {}
    for r in results:
        full = r['projected_300ep_seconds'] * r['grid_size']
        rec_limit = full * 1.5          # 参照AE・anchor・再探索・probe の余裕として 1.5 倍
        budget[r['dataset']] = {
            'seconds_per_epoch': r['seconds_per_epoch'],
            'projected_300ep_seconds': r['projected_300ep_seconds'],
            'grid_size': r['grid_size'],
            'full_grid_300ep_seconds': full,
            'recommended_wallclock_limit_seconds': rec_limit,
            'basis': 'full grid x 1 seed at 300 epochs, x1.5 margin for reference AE / anchor / re-search / probe',
        }
        print(f"{r['name']:16s} {r['seconds_per_epoch']:8.3f} "
              f"{r['projected_300ep_seconds']/60:8.1f}分 {r['grid_size']:7d} "
              f"{full/3600:10.2f}h {rec_limit/3600:10.2f}h")
    with open(os.path.join(OUT_DIR, 'stage3_budget.json'), 'w') as f:
        json.dump(budget, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT_DIR}/stage3_calibration.json, {OUT_DIR}/stage3_budget.json")


if __name__ == '__main__':
    main()
