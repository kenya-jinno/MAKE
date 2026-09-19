"""真の ELBO による容量選択を複数 seed で比較する（MAKE53_revision_review.md §5.1）。

指摘: 現在の 1 seed は実 ELBO を計算する手順の確認としては有効だが、
「stronger baselines based on ELBO」への比較結果としては弱い。

§5.1 が挙げた最小限の補強をそのまま実施する:
  1. MNIST、beta=1、同じ 12 候補、同じ train/validation/test 分割を維持する
  2. 学習 seed を主比較と同じ 5 つに揃える
  3. 各 seed の同じ候補モデル群から MSE・proxy・真の MC ELBO でそれぞれ容量を選択する
  4. seed ごとの選択幅、validation の選択指標、選択モデルの test MSE と test MC ELBO、
     選択の一致・不一致を示す
  5. 学習 seed 間の SD と Monte Carlo 評価の SE を**別々に**報告する

**環境について**: 既存 seed 42 は CPU 環境で実行されている。
§5.1 の注意に従い、それと混ぜて「同一環境の 5-seed 反復」とは扱わない。
**seed 42 を含む 5 seeds すべてを本 run の GPU 環境で新規に実行する。**

**実装の事前確認**（§5.1 の列挙）:
  - posterior サンプルごとの対数尤度を平均し、対象モデルの prior に対する KL を使う
  - Gaussian、観測分散 1/2 のとき
    MC ELBO = -D * (標本ごとの画素平均 MSE) - KL - (D/2) log pi
    と一致することを小規模に照合する（`--selfcheck`）
  - 既存の raw な `elbo` 欄は再利用しない。本 run で新規に計算する

**限定**: checkpoint は決定論的 proxy で選ぶ。したがって本比較は
「proxy で選んだ同一 checkpoint 群から容量だけを選ぶ」設計であり、
epoch 選択まで ELBO で行う pipeline との比較ではない。

実行: プロジェクト直下で python3 src/make53/elbo_multiseed.py [--selfcheck]
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.make49.common import CFG, DEVICE, set_seed
from src.make49.datasets import LOADERS
from src.make49.models import FCVAE
from src.make52.geco_arm import provenance
from src.make53 import artifacts as AR

OUT = ROOT / 'results/make53'; OUT.mkdir(exist_ok=True)
GRID = CFG['candidate_grids']['MNIST']
SEEDS = CFG['data_splits']['training_seeds']          # 主比較と同じ 5 seeds
D = 784
SPEC = dict(dataset='MNIST', beta=1.0, grid=GRID, seeds=SEEDS,
            max_epochs=300, patience=30, batch_size=128, lr=1e-3,
            checkpoint='best deterministic ELBO proxy',
            mc_samples=32, mc_seed=20260919, gaussian_variance=0.5,
            device=str(DEVICE), note='seed 42 を含め全 seed を本 run の環境で新規学習')


@torch.no_grad()
def deterministic_eval(model, X):
    """決定論的 g(mu) による MSE と proxy（Gaussian 定数と事後期待を含まない）。"""
    model.eval(); rec = kl = 0.0
    for xb in X.split(512):
        mu, lv = model.encode(xb); xh = model.decode(mu)
        rec += float((xh - xb).square().sum())
        kl += float(.5 * (mu.square() + lv.exp() - 1 - lv).sum())
    rec /= len(X); kl /= len(X)
    return dict(mse=rec / D, rec=rec, kl=kl, proxy=-(rec + kl))


@torch.no_grad()
def mc_elbo(model, X, K=32, seed=20260919):
    """真の ELBO（Gaussian、観測分散 1/2）。

    posterior サンプルごとに対数尤度を評価して平均し、prior に対する KL を引く。
    共通乱数を用い、容量の異なるモデル間で対応のある比較ができるようにする。
    """
    model.eval(); mus, lvs = [], []
    for xb in X.split(512):
        mu, lv = model.encode(xb); mus.append(mu); lvs.append(lv)
    mu, lv = torch.cat(mus), torch.cat(lvs)
    kl = float(.5 * (mu.square() + lv.exp() - 1 - lv).sum() / len(X))
    gen = torch.Generator(device='cpu').manual_seed(seed)
    vals, sampled_mse = [], []
    for _ in range(K):
        eps = torch.randn(len(X), max(GRID), generator=gen)[:, :mu.shape[1]].to(DEVICE)
        z = mu + eps * torch.exp(.5 * lv); rec = 0.0
        for st in range(0, len(X), 512):
            rec += float((model.decode(z[st:st + 512]) - X[st:st + 512]).square().sum())
        rec /= len(X)
        sampled_mse.append(rec / D)
        vals.append(-(rec + kl + (D / 2) * math.log(math.pi)))
    return dict(elbo=float(np.mean(vals)),
                mc_se=float(np.std(vals, ddof=1) / np.sqrt(K)),
                kl=kl, K=K, evaluation_seed=seed,
                mean_sampled_mse=float(np.mean(sampled_mse)))


def selfcheck(model, X):
    """MC ELBO = -D * sampled per-pixel MSE - KL - (D/2) log pi の照合。"""
    r = mc_elbo(model, X, K=8)
    lhs = r['elbo']
    rhs = -D * r['mean_sampled_mse'] - r['kl'] - (D / 2) * math.log(math.pi)
    return {'name': 'MC ELBO identity (Gaussian, variance 1/2)',
            'mc_elbo': lhs, 'from_sampled_mse': rhs,
            'abs_diff': abs(lhs - rhs), 'pass': abs(lhs - rhs) < 1e-6 * max(1, abs(lhs))}


def train_one(Xtr, Xv, Xcurve, m, seed):
    set_seed(seed)
    model = FCVAE(D, m).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=SPEC['lr'])
    best = None; curves = []; t0 = time.perf_counter()
    for ep in range(1, SPEC['max_epochs'] + 1):
        model.train(); perm = torch.randperm(len(Xtr))
        for ids in perm.split(SPEC['batch_size']):
            xb = Xtr[ids]; opt.zero_grad()
            xh, mu, lv = model(xb)
            ((xh - xb).square().sum() + .5 * (mu.square() + lv.exp() - 1 - lv).sum()).div(
                len(xb)).backward()
            opt.step()
        v = deterministic_eval(model, Xv); t = deterministic_eval(model, Xcurve)
        curves.append(dict(epoch=ep, train_mse=t['mse'], **{k: v[k] for k in ('mse', 'proxy')}))
        if best is None or v['proxy'] > best['proxy']:
            best = dict(epoch=ep, proxy=v['proxy'],
                        state={k: x.detach().clone() for k, x in model.state_dict().items()})
        if ep - best['epoch'] >= SPEC['patience']:
            break
    model.load_state_dict(best['state'])
    return model, best['epoch'], len(curves), curves, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()
    data = LOADERS['MNIST'](flatten=True)
    Xtr = torch.as_tensor(data['X_train']).to(DEVICE)
    Xv = torch.as_tensor(data['X_val']).to(DEVICE)
    Xte = torch.as_tensor(data['X_test']).to(DEVICE)
    Xcurve = Xtr[:2000]
    prov = provenance(); prov['code_fingerprint'] = AR.code_fingerprint(
        ['src/make53/elbo_multiseed.py', 'src/make53/artifacts.py'])

    print(f"真の ELBO による容量選択（MNIST, beta=1, {len(GRID)} 候補 x {len(SEEDS)} seeds）")
    print(f"  環境 {prov['gpu']} / {prov['torch']} / 分割ハッシュ {data['split']['index_hash']}")
    print(f"  **seed 42 を含む全 seed を本 run の環境で新規学習する**\n")

    results, checks = [], []
    for s in SEEDS:
        per = {}
        for m in GRID:
            model, best_ep, ran, curves, sec = train_one(Xtr, Xv, Xcurve, m, s)
            val = deterministic_eval(model, Xv); tst = deterministic_eval(model, Xte)
            mv = mc_elbo(model, Xv, SPEC['mc_samples'], SPEC['mc_seed'])
            mt = mc_elbo(model, Xte, SPEC['mc_samples'], SPEC['mc_seed'])
            if args.selfcheck and not checks:
                checks.append(selfcheck(model, Xv))
            per[m] = dict(m=m, best_epoch=best_ep, epochs_run=ran, seconds=sec,
                          val=val, test=tst, val_mc=mv, test_mc=mt,
                          curve_tail=curves[-5:])
            print(f"  seed {s:<6d} m={m:3d} ep={best_ep:3d}/{ran:3d} "
                  f"val MSE {val['mse']:.5f} proxy {val['proxy']:9.2f} "
                  f"MC ELBO {mv['elbo']:9.2f}±{mv['mc_se']:.2f} {sec:6.0f}s", flush=True)
        choices = {'MSE': min(per, key=lambda m: per[m]['val']['mse']),
                   'ELBO proxy': max(per, key=lambda m: per[m]['val']['proxy']),
                   'MC ELBO': max(per, key=lambda m: per[m]['val_mc']['elbo'])}
        results.append(dict(seed=s, per_width=per, choices=choices))
        print(f"  -> seed {s} の選択: MSE {choices['MSE']}, proxy {choices['ELBO proxy']}, "
              f"MC ELBO {choices['MC ELBO']}\n", flush=True)
        json.dump({'_spec': SPEC, '_provenance': prov, '_checks': checks,
                   'results': results}, open(OUT / 'elbo_multiseed.json', 'w'),
                  ensure_ascii=False)
    if checks:
        AR.report(checks, OUT / 'elbo_selfcheck.json')
    print(f"書き出し: {OUT}/elbo_multiseed.json")


if __name__ == '__main__':
    main()
