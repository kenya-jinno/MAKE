"""段階 3: 入力空間の ID 推定。初期候補窓 W と m_ref の基準 d̂_0 を決める。

設定表 §5.3 に従い、**train** から固定乱数で 10,000 点を取り、TwoNN（主）と MLE（補助）で推定する。
推定器一致の基準 |TwoNN-MLE|/TwoNN <= 0.25 を判定する。

実行: プロジェクト直下で python3 src/make49/stage3_input_id.py
"""
import json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, SPLIT_SEED, provenance
from src.make49.datasets import LOADERS
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)

ES = CFG['reference_and_id']['estimation_sample']
AGREE = CFG['reference_and_id']['estimator_agreement_criterion']['max_relative_diff']
GRIDS = CFG['candidate_grids']
C = CFG['initial_window']['c_primary']

DATASETS = [('MNIST', {'flatten': True}), ('FashionMNIST', {'flatten': True}),
            ('dSprites', {}), ('CIFAR10', {})]


def first_ge(grid, x):
    for g in grid:
        if g >= x:
            return g
    return None


def main():
    print(f"入力空間の ID 推定（{ES['source']} から固定 {ES['n']} 点、設定表 §5.3）")
    print(f"推定器一致の基準: |TwoNN-MLE|/TwoNN <= {AGREE}\n")
    print(f"{'データ':10s} {'n':>6} {'TwoNN':>8} {'MLE':>8} {'rel差':>7} {'一致':>5} "
          f"{'AE窓[d,2d]':>14} {'VAE初期点':>9} {'m_ref(4/8/16 d̂0)':>20}")
    out = {'_provenance': provenance(), '_spec': {'source': ES['source'], 'n': ES['n'],
                                                  'agreement_criterion': AGREE, 'c_primary': C}}
    for name, kw in DATASETS:
        d = LOADERS[name](**kw)
        X = d['X_train'].reshape(len(d['X_train']), -1)
        k = min(ES['n'], len(X))
        rng = np.random.RandomState(SPLIT_SEED + 3)
        Xs = X[rng.choice(len(X), k, replace=False)]
        t = time.time()
        tn = float(twonn_estimate(Xs)); ml = float(mle_estimate(Xs, k=10))
        rel = abs(tn - ml) / tn
        grid = GRIDS[name]
        ae_win = [g for g in grid if tn <= g <= 2 * tn]
        vae_pt = first_ge(grid, C * tn)
        d0 = first_ge(grid, tn) or grid[-1]
        m_refs = [first_ge(grid, mult * tn) or grid[-1] for mult in (4, 8, 16)]
        out[name] = {
            'n_used': k, 'twonn': tn, 'mle': ml, 'rel_diff': rel,
            'estimator_agreement_pass': bool(rel <= AGREE),
            'ae_initial_window': ae_win,
            'vae_initial_diagnostic_point': vae_pt,
            'd0_grid': d0,
            'm_ref_candidates': m_refs,
            'grid': grid,
            'seconds': time.time() - t,
        }
        print(f"{name:10s} {k:6d} {tn:8.3f} {ml:8.3f} {rel:7.3f} "
              f"{str(rel <= AGREE):>5s} {str(ae_win):>14s} {str(vae_pt):>9s} {str(m_refs):>20s}")

    with open(os.path.join(OUT, 'input_space_id.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/input_space_id.json")
    print("注: m_ref は候補グリッド上に丸めた値。グリッド上限を超える倍率は上限で頭打ちになる。")


if __name__ == '__main__':
    main()
