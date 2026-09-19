"""段階 5 / E4: AU の増分価値の最終判定（設定表 §1.1 の方式 2）。

事前登録した予測課題:
  評価単位 : 1 checkpoint = (データ, アーキテクチャ, m, seed, 最良 epoch)
  予測対象 Y: 候補グリッド上で **1 つ小さい点 m- で学習したモデルが Q を満たすか**（二値）
  非循環性 : Y は別の m- で学習した別モデルの品質であり、現 checkpoint の AU からは決まらない
  S_base   : D_val, KL_val, ELBO_val, m, m/d_ID, 最終 20 epochs の D_val の傾き
  S_AU     : S_base ∪ {AU, AU/m, AU/d_ID}
  予測器   : ロジスティック回帰(L2)。正則化は**開発条件の validation のみ**で選び以後固定
  開発条件 : MNIST FC-VAE
  未使用条件: dSprites, Fashion-MNIST, CIFAR-10
  主要量   : ΔAUROC = AUROC(S_AU) − AUROC(S_base)
  判定規則 : 未使用条件全体の平均 ΔAUROC について seed 単位ブートストラップ 2000 反復の
             95% CI を求め、**下限が 0 を上回る場合に限り**「AU に増分価値あり」と判定する。
             区間が 0 を含む、または上限が 0 を下回る場合は AU を補助分析へ移す。

再学習は不要。段階 5 のキャッシュから構成する。

実行: プロジェクト直下で python3 src/make49/stage5_e4_au_value.py
"""
import json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
CACHE = os.path.join(OUT, 'stage4_cache')
BOOT = 2000
RNG = np.random.default_rng(20260913)
EPS = CFG['quality_criterion_Q']['epsilon_D']
DEV = ['MNIST']
HELD_OUT = ['dSprites', 'FashionMNIST', 'CIFAR10']
BASE_FEATS = ['D_val', 'KL_val', 'ELBO_val', 'm', 'm_over_d', 'slope20']
AU_FEATS = ['AU', 'AU_over_m', 'AU_over_d']


def load_runs(ds):
    p = os.path.join(CACHE, f'{ds}.json')
    if not os.path.exists(p):
        return None
    store = json.load(open(p))
    runs = {}
    for k, v in store.items():
        if not k.startswith('vae|') or '|e300|' not in k:
            continue
        if 'curve_val_mse' not in v:
            continue
        runs[(v['m'], v['seed'])] = v
    return runs


def build_table(ds, d_hat):
    """各 checkpoint について特徴量と予測対象 Y を作る。"""
    runs = load_runs(ds)
    if not runs:
        return None
    grid = sorted({m for m, _ in runs})
    seeds = sorted({s for _, s in runs})
    rows = []
    for s in seeds:
        # Q は seed ごとに anchor（グリッド最大点）から決める
        anchor = grid[-1]
        if (anchor, s) not in runs:
            continue
        Da = runs[(anchor, s)]['val']['mse']
        T_D = Da + max(EPS['rel_primary'] * Da, 0.0)
        for i, m in enumerate(grid):
            if i == 0 or (m, s) not in runs:
                continue
            m_minus = grid[i - 1]
            if (m_minus, s) not in runs:
                continue
            r = runs[(m, s)]
            c = r['curve_val_mse']
            w = c[-20:] if len(c) >= 20 else c
            slope = float(np.polyfit(np.arange(len(w)), w, 1)[0]) if len(w) > 1 else 0.0
            au = r['readouts']['B']['au']
            rows.append({
                'dataset': ds, 'm': m, 'seed': s, 'm_minus': m_minus,
                'D_val': r['val']['mse'], 'KL_val': r['val']['kl'],
                'ELBO_val': r['val']['elbo'],
                'm_over_d': m / max(d_hat, 1e-9), 'slope20': slope,
                'AU': au, 'AU_over_m': au / m, 'AU_over_d': au / max(d_hat, 1e-9),
                'Y': int(runs[(m_minus, s)]['val']['mse'] <= T_D),
                'T_D': T_D,
            })
    return rows


def fit_eval(train_rows, test_rows, feats, C):
    Xtr = np.array([[r[f] for f in feats] for r in train_rows], float)
    ytr = np.array([r['Y'] for r in train_rows])
    Xte = np.array([[r[f] for f in feats] for r in test_rows], float)
    yte = np.array([r['Y'] for r in test_rows])
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return {'auroc': float(roc_auc_score(yte, p)),
            'balanced_acc': float(balanced_accuracy_score(yte, (p >= 0.5).astype(int))),
            'n_test': len(yte), 'pos_rate_test': float(yte.mean())}


def main():
    iid = json.load(open(os.path.join(OUT, 'input_space_id.json')))
    tables = {}
    for ds in DEV + HELD_OUT:
        if ds not in iid:
            print(f"  [skip] {ds}: 入力空間 ID が未計算")
            continue
        t = build_table(ds, iid[ds]['twonn'])
        if t:
            tables[ds] = t
            print(f"  {ds:12s} {len(t):4d} checkpoint  Y=1 の割合 "
                  f"{np.mean([r['Y'] for r in t]):.2f}")
        else:
            print(f"  [skip] {ds}: キャッシュに拡張統計が無い")
    if not set(DEV) <= set(tables):
        print("開発条件のデータが無いため中止")
        return

    dev = [r for ds in DEV for r in tables[ds]]
    # 正則化強度は開発条件の validation のみで選ぶ（以後固定）
    dev_seeds = sorted({r['seed'] for r in dev})
    inner_tr = [r for r in dev if r['seed'] in dev_seeds[:3]]
    inner_va = [r for r in dev if r['seed'] in dev_seeds[3:]]
    best_C, best = None, -1
    for C in (0.01, 0.1, 1.0, 10.0):
        e = fit_eval(inner_tr, inner_va, BASE_FEATS, C)
        if e and e['auroc'] > best:
            best, best_C = e['auroc'], C
    print(f"\n正則化 C={best_C}（開発条件の validation のみで選択、以後固定）\n")

    print(f"{'未使用条件':12s} {'n':>5s} {'AUROC(S_base)':>14s} {'AUROC(S_AU)':>12s} "
          f"{'ΔAUROC':>9s} {'bAcc base':>10s} {'bAcc AU':>9s}")
    per_ds, deltas_by_seed = {}, []
    for ds in HELD_OUT:
        if ds not in tables:
            continue
        te = tables[ds]
        b = fit_eval(dev, te, BASE_FEATS, best_C)
        a = fit_eval(dev, te, BASE_FEATS + AU_FEATS, best_C)
        if not (b and a):
            print(f"{ds:12s} 片方のクラスしか無く評価不能")
            continue
        d = a['auroc'] - b['auroc']
        per_ds[ds] = {'base': b, 'au': a, 'delta_auroc': d}
        print(f"{ds:12s} {b['n_test']:5d} {b['auroc']:14.3f} {a['auroc']:12.3f} "
              f"{d:+9.3f} {b['balanced_acc']:10.3f} {a['balanced_acc']:9.3f}")
        # seed 単位のブートストラップ用に seed ごとの ΔAUROC も出す
        for s in sorted({r['seed'] for r in te}):
            sub = [r for r in te if r['seed'] == s]
            bb = fit_eval(dev, sub, BASE_FEATS, best_C)
            aa = fit_eval(dev, sub, BASE_FEATS + AU_FEATS, best_C)
            if bb and aa:
                deltas_by_seed.append({'dataset': ds, 'seed': s,
                                       'delta': aa['auroc'] - bb['auroc']})

    if not deltas_by_seed:
        print("\nseed 単位の ΔAUROC を構成できず、事前登録の判定は不能")
        return
    vals = np.array([d['delta'] for d in deltas_by_seed])
    boots = [np.mean(vals[RNG.integers(0, len(vals), len(vals))]) for _ in range(BOOT)]
    lo, med, hi = np.percentile(boots, [2.5, 50, 97.5])
    if lo > 0:
        verdict = 'AU に増分価値あり'
    elif hi < 0:
        verdict = 'AU を加えると悪化する'
    else:
        verdict = '増分価値は示されない（AU を補助分析へ移す）'
    print(f"\n=== 事前登録の判定（seed 単位ブートストラップ {BOOT} 反復）===")
    print(f"  平均 ΔAUROC = {vals.mean():+.3f}")
    print(f"  95% CI = [{lo:+.3f}, {hi:+.3f}]   中央値 {med:+.3f}")
    print(f"  判定規則: 下限 > 0 のときのみ増分価値あり")
    print(f"  → 判定: **{verdict}**")

    res = {'_spec': {'bootstrap': BOOT, 'C': best_C, 'dev': DEV, 'held_out': HELD_OUT,
                     'base_features': BASE_FEATS, 'au_features': AU_FEATS,
                     'rule': 'AU has value only if the lower bound of the CI exceeds 0'},
           'per_dataset': per_ds, 'per_seed_delta': deltas_by_seed,
           'mean_delta_auroc': float(vals.mean()),
           'ci': {'lo': float(lo), 'median': float(med), 'hi': float(hi)},
           'verdict': verdict}
    with open(os.path.join(OUT, 'stage5_e4_au_value.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage5_e4_au_value.json")


if __name__ == '__main__':
    main()
