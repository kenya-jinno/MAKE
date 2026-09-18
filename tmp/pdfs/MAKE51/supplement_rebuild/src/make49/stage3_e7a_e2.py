"""段階 3: E7a（読み取りの安定性）と E2（学習曲線）の代表試行。

E7a と E2 は同じ学習条件を共有するので、同一のランから両方の材料を取る。

E7a（設定表 §1.2）
  - early stopping なし。必ず 300 epochs まで回す。
  - checkpoint 1/2/5/50/300 で読み取り A/B/C を計算。主要比較は 5 vs 300。
  - 独立学習 5 seeds。seed 間の安定性と学習中の安定性を分けて記録する。

E2（方針 §5.5）
  - train / validation の MSE を epoch に対して記録。
  - **train MSE も評価モードで計算**し、validation と同じ再構成方式を使う。
  - VAE は KL と通常の ELBO も別系列で記録。
  - checkpoint は validation のみで選択。test は選択済み checkpoint でのみ評価。

容量条件（results/make49/e7a_capacities.json、事前指定の規則から導出）
  小さい容量 = d̂_ID を下回る最大のグリッド点
  初期候補   = 2·d̂_ID 以上の最初のグリッド点
  余裕ある容量 = 初期候補の 2 倍以上の最初のグリッド点

実行: プロジェクト直下で python3 src/make49/stage3_e7a_e2.py
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import (CFG, DEVICE, SPLIT_SEED, provenance, set_seed,
                               reconstruction_term)
from src.make49.datasets import LOADERS
from src.make49.models import build
from src.make49.readouts import all_readouts

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)
RESULT_PATH = os.path.join(OUT, 'stage3_e7a_e2.json')

BETA = CFG['beta']['primary']
MAX_EPOCHS = CFG['budget_and_termination']['max_epochs']
SNAPSHOTS = CFG['E7a']['recorded_epochs']          # [1, 2, 5, 50, 300]
PRIMARY = CFG['E7a']['primary_comparison_epochs']  # [5, 300]
SEEDS = CFG['data_splits']['training_seeds']       # 5 seeds
EST_N = CFG['reference_and_id']['estimation_sample']['n']

SPECS = {'MNIST': {'kind': 'fc', 'input_dim': 784},
         'dSprites': {'kind': 'conv', 'in_ch': 1, 'size': 64}}
LOADER_KW = {'MNIST': {'flatten': True}, 'dSprites': {}}


def eval_pass(model, X, likelihood, batch=512):
    """評価モードでの MSE・再構成項・KL・通常 ELBO。VAE は決定論的再構成 g(mu(x))。"""
    model.eval()
    tot_mse = tot_rec = tot_kl = 0.0
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
            B = xb.size(0)
            mu, lv = model.encode(xb)
            xh = model.decode(mu)          # 決定論的再構成 g(mu(x))
            tot_mse += F.mse_loss(xh, xb, reduction='sum').item() / xb[0].numel()
            tot_rec += reconstruction_term(xh, xb, likelihood).item() * B
            tot_kl += (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()
    return {'mse_per_pixel_mean': tot_mse / n, 'rec': tot_rec / n,
            'kl': tot_kl / n, 'elbo': -(tot_rec / n + tot_kl / n)}


def run_one(dataset, m, seed, data, X_est, likelihood, progress=100):
    set_seed(seed)
    model = build(SPECS[dataset], m).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(data['X_train'])
    Xva = data['X_val']
    # E2 用の train 側評価部分集合（評価モードで計算する固定サブセット）
    rng = np.random.RandomState(SPLIT_SEED + 7)
    tr_eval_idx = rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)
    Xtr_eval = data['X_train'][tr_eval_idx]

    n = len(Xtr)
    curves = {k: [] for k in ['epoch', 'val_mse', 'val_kl', 'val_elbo', 'train_mse']}
    readouts, best = {}, {'elbo': -float('inf'), 'epoch': -1, 'state': None}
    t0 = time.time(); steps = 0

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xtr[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            rec = reconstruction_term(xh, xb, likelihood)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
            (rec + BETA * kl).backward(); opt.step(); steps += 1

        v = eval_pass(model, Xva, likelihood)
        t = eval_pass(model, Xtr_eval, likelihood)      # E2: 評価モードの train MSE
        curves['epoch'].append(ep)
        curves['val_mse'].append(v['mse_per_pixel_mean'])
        curves['val_kl'].append(v['kl'])
        curves['val_elbo'].append(v['elbo'])
        curves['train_mse'].append(t['mse_per_pixel_mean'])

        if v['elbo'] > best['elbo']:
            best = {'elbo': v['elbo'], 'epoch': ep,
                    'state': {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}}

        if ep in SNAPSHOTS:
            r = all_readouts(model, X_est, seed)
            r['val'] = v
            readouts[str(ep)] = r

        if progress and ep % progress == 0:
            print(f"      ep {ep:4d}  val_ELBO={v['elbo']:11.3f}  val_MSE={v['mse_per_pixel_mean']:.5f}"
                  f"  {time.time()-t0:6.1f}s", flush=True)

    # plateau は記録のみ（E7a では停止に使わない）
    w = curves['val_elbo'][-50:]
    plateau = (max(w) - min(w)) / max(abs(np.mean(w)), 1e-12) < 0.01

    model.load_state_dict(best['state'])
    test_eval = eval_pass(model, data['X_test'], likelihood)   # 選択済み checkpoint のみ
    return {'dataset': dataset, 'm': m, 'seed': seed, 'beta': BETA,
            'likelihood': likelihood, 'epochs_run': MAX_EPOCHS,
            'early_stopping_enabled': False, 'best_epoch': best['epoch'],
            'plateau': bool(plateau), 'elapsed_seconds': time.time() - t0,
            'update_steps': steps, 'curves': curves, 'readouts': readouts,
            'test_eval_at_best_checkpoint': test_eval}


def main():
    caps = json.load(open(os.path.join(OUT, 'e7a_capacities.json')))
    prov = provenance()
    print(f"E7a / E2 代表試行  beta={BETA}  {MAX_EPOCHS} epochs  early stopping なし")
    print(f"snapshots={SNAPSHOTS}  主要比較={PRIMARY}  seeds={SEEDS}\n")

    results = json.load(open(RESULT_PATH))['runs'] if os.path.exists(RESULT_PATH) else []
    done = {(r['dataset'], r['m'], r['seed']) for r in results}

    for ds in ['MNIST', 'dSprites']:
        c = caps[ds]
        data = LOADERS[ds](**LOADER_KW[ds])
        rng = np.random.RandomState(SPLIT_SEED + 3)
        idx = rng.choice(len(data['X_train']), min(EST_N, len(data['X_train'])), replace=False)
        X_est = data['X_train'][idx]          # 全読み取りで共通の推定用標本
        print(f"=== {ds} (尤度 {c['likelihood']}, d̂_ID={c['d_hat']:.2f}) ===")
        for label in ['small', 'initial', 'ample']:
            m = c[label]
            for seed in SEEDS:
                if (ds, m, seed) in done:
                    print(f"  [skip] {label} m={m} seed={seed}")
                    continue
                print(f"  {label:8s} m={m:3d} seed={seed}", flush=True)
                r = run_one(ds, m, seed, data, X_est, c['likelihood'])
                r['capacity_label'] = label
                r['d_hat'] = c['d_hat']
                results.append(r)
                a = r['readouts'][str(MAX_EPOCHS)]
                print(f"    -> 300ep: A(act+mix)={a['A']['active_plus_mixed']}/{m} "
                      f"(a={a['A']['active']},mx={a['A']['mixed']},p={a['A']['passive']})  "
                      f"B(AU)={a['B']['au']}/{m}  C(gap)={a['C']['gap']:+.3f}  "
                      f"MSE={a['val']['mse_per_pixel_mean']:.5f}  {r['elapsed_seconds']:.0f}s",
                      flush=True)
                with open(RESULT_PATH, 'w') as f:
                    json.dump({'_provenance': prov,
                               '_spec': {'beta': BETA, 'max_epochs': MAX_EPOCHS,
                                         'snapshots': SNAPSHOTS, 'primary_comparison': PRIMARY,
                                         'seeds': SEEDS, 'early_stopping': False,
                                         'estimation_sample_n': int(len(X_est))},
                               'capacities': caps, 'runs': results}, f, ensure_ascii=False)
    print(f"\n完了: {len(results)} ラン -> {RESULT_PATH}")


if __name__ == '__main__':
    main()
