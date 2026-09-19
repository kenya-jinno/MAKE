"""段階 5 / E3: 参照表現への依存性（方針 §5.5 E3、R3-3 直結）。

問い: 参照 AE が変わっても、ID だけでなく最終選択と品質が安定するか。

比較する表現: 入力空間 / PCA 前処理空間 / FC-AE（複数 $m_{ref}$・複数 seed）。
推定用サンプル数も変える。Conv-AE 参照は未実施（計算予算の都合）。

**独立した VAE seed で推定値が近いことは、参照 AE 自体の seed 頑健性の代わりにならない。**

実行: プロジェクト直下で python3 src/make49/stage5_e3_reference.py
"""
import glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG, SPLIT_SEED
from src.make49.datasets import LOADERS
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate

OUT = os.path.join('results', 'make49')
KW = {'MNIST': {'flatten': True}, 'FashionMNIST': {'flatten': True},
      'dSprites': {}, 'CIFAR10': {}}
SAMPLE_SIZES = [1000, 2000, 5000, 10000]
PCA_DIMS = [16, 32, 64, 128]
CRIT = CFG['reference_and_id']


def collect_ae_from_caches(ds):
    """段階 4 のキャッシュ（退避済みを含む）から参照 AE の推定値を集める。"""
    rows = []
    for p in glob.glob(os.path.join(OUT, 'stage4_cache*', f'{ds}.json')):
        for k, v in json.load(open(p)).items():
            if k.startswith('ae|'):
                rows.append(v)
    uniq = {}
    for r in rows:
        uniq[(r['m_ref'], r['seed'])] = r
    return list(uniq.values())


def main():
    datasets = [d for d in ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']]
    out = {}
    for ds in datasets:
        data = LOADERS[ds](**KW[ds])
        X = data['X_train'].reshape(len(data['X_train']), -1)
        rng = np.random.RandomState(SPLIT_SEED + 3)
        print(f"\n{'='*74}\n=== {ds} ===\n{'='*74}")

        # 1. 推定用サンプル数への依存
        print(f"\n--- 1. 推定用サンプル数への依存（入力空間）---")
        print(f"  {'n':>7s} {'TwoNN':>8s} {'MLE':>8s} {'相対差':>8s}")
        ss = []
        for n in SAMPLE_SIZES:
            Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)]
            tw = float(twonn_estimate(Xs)); ml = float(mle_estimate(Xs, k=10))
            ss.append({'n': n, 'twonn': tw, 'mle': ml,
                       'rel_diff': abs(tw - ml) / max(tw, 1e-9)})
            print(f"  {n:7d} {tw:8.3f} {ml:8.3f} {abs(tw-ml)/max(tw,1e-9):8.3f}")
        tws = [r['twonn'] for r in ss]
        print(f"  → TwoNN の n 依存の振れ幅 {min(tws):.2f}–{max(tws):.2f} "
              f"（相対 {(max(tws)-min(tws))/np.median(tws):.2f}）")

        # 2. PCA 前処理空間への依存
        print(f"\n--- 2. PCA 前処理空間での推定 ---")
        from sklearn.decomposition import PCA
        Xs = X[rng.choice(len(X), min(5000, len(X)), replace=False)]
        keep = Xs.var(0) > 1e-8
        Xk = Xs[:, keep]
        print(f"  {'PCA 次元':>9s} {'TwoNN':>8s} {'MLE':>8s}")
        pc = []
        for k in PCA_DIMS:
            if k >= Xk.shape[1]:
                continue
            Z = PCA(n_components=k, random_state=0).fit_transform(Xk)
            tw = float(twonn_estimate(Z)); ml = float(mle_estimate(Z, k=10))
            pc.append({'pca_dim': k, 'twonn': tw, 'mle': ml})
            print(f"  {k:9d} {tw:8.3f} {ml:8.3f}")
        if pc:
            t = [r['twonn'] for r in pc]
            print(f"  → PCA 次元による振れ幅 {min(t):.2f}–{max(t):.2f}")

        # 3. 参照 AE（段階 4 のキャッシュから）
        print(f"\n--- 3. 参照 AE の容量と seed への依存 ---")
        ae = collect_ae_from_caches(ds)
        if not ae:
            print("  （段階 4 で参照 AE を学習していない）")
            ae_summary = None
        else:
            byref = {}
            for r in ae:
                byref.setdefault(r['m_ref'], []).append(r)
            print(f"  {'m_ref':>7s} {'seeds':>6s} {'TwoNN mean±SD':>18s} {'MLE mean':>10s}")
            means = []
            for mr in sorted(byref):
                tw = [r['twonn'] for r in byref[mr]]
                ml = [r['mle'] for r in byref[mr]]
                means.append(float(np.mean(tw)))
                sd = np.std(tw, ddof=1) if len(tw) > 1 else 0.0
                print(f"  {mr:7d} {len(tw):6d} {np.mean(tw):11.3f}±{sd:<6.3f} {np.mean(ml):10.3f}")
            rel_range = (max(means) - min(means)) / max(np.median(means), 1e-9)
            stable = rel_range <= CRIT['id_stability_criterion']['max_relative_range']
            print(f"  → 容量間の相対範囲 {rel_range:.3f} "
                  f"（基準 ≤{CRIT['id_stability_criterion']['max_relative_range']}）→ "
                  f"{'安定' if stable else '**不安定**'}")
            ae_summary = {'by_m_ref': {str(k): [r['twonn'] for r in v] for k, v in byref.items()},
                          'relative_range': rel_range, 'stable': bool(stable)}

        out[ds] = {'sample_size': ss, 'pca_space': pc, 'reference_ae': ae_summary}

    print(f"\n\n{'='*74}\n=== まとめ: 表現ごとの $\\hat d$ の食い違い ===\n{'='*74}")
    print(f"{'データ':13s} {'入力(n=10k)':>12s} {'PCA 最小':>10s} {'PCA 最大':>10s} {'参照 AE':>18s}")
    for ds, v in out.items():
        inp = [r['twonn'] for r in v['sample_size'] if r['n'] == 10000]
        pcs = [r['twonn'] for r in v['pca_space']]
        ae = v['reference_ae']
        aestr = '—'
        if ae:
            allv = [x for lst in ae['by_m_ref'].values() for x in lst]
            aestr = f"{min(allv):.2f}–{max(allv):.2f}"
        print(f"{ds:13s} {(inp[0] if inp else float('nan')):12.2f} "
              f"{(min(pcs) if pcs else float('nan')):10.2f} "
              f"{(max(pcs) if pcs else float('nan')):10.2f} {aestr:>18s}")

    with open(os.path.join(OUT, 'stage5_e3_reference.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage5_e3_reference.json")


if __name__ == '__main__':
    main()
