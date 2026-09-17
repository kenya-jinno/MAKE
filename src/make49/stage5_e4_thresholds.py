"""段階 5 / E4: 閾値・構成要素の検証（方針 §5.5 E4、R3-1 直結）。

問い: 係数 2 や AU 条件に依存せず、品質・費用に基づく結論が維持されるか。

保存した潜在統計（次元別 μ 分散）から**再学習なし**で再評価する:
  - AU の δ 感度（0.001 / 0.01 / 0.1）
  - 絶対閾値 AU と**スケール正規化 AU** の比較（段階 3 の C6 への対応）
  - V1 の活性割合閾値、V2 の上下係数の再評価
  - 品質許容誤差 ε_D の水準別の選択 m
  - 初期範囲の係数 c（**事後的な探索シミュレーション**として明記する）

実行: プロジェクト直下で python3 src/make49/stage5_e4_thresholds.py
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
CACHE = os.path.join(OUT, 'stage4_cache')
DELTAS = CFG['au']['delta_sensitivity']
C_VALUES = CFG['initial_window']['c_sensitivity']
EPS_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50]
V1_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
V2_LOWER = [0.5, 0.75, 1.0]
V2_UPPER = [2.0, 3.0, 4.0]


def load(ds):
    p = os.path.join(CACHE, f'{ds}.json')
    if not os.path.exists(p):
        return None
    st = json.load(open(p))
    runs = {}
    for k, v in st.items():
        if k.startswith('vae|') and '|e300|' in k and \
           'mu_var_per_dim' in v.get('readouts', {}).get('B', {}):
            runs[(v['m'], v['seed'])] = v
    return runs or None


def au_abs(var, delta):
    return int((np.asarray(var) > delta).sum())


def au_rel(var, delta_rel):
    """スケール正規化 AU: 最大分散に対する相対閾値。

    段階 3 の C6（β を上げると μ のスケールが縮み、絶対閾値の判定が逆行する）への対照。
    """
    v = np.asarray(var)
    if v.max() <= 0:
        return 0
    return int((v > delta_rel * v.max()).sum())


def main():
    iid = json.load(open(os.path.join(OUT, 'input_space_id.json')))
    datasets = [d for d in ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']
                if load(d) and d in iid]
    if not datasets:
        print('拡張統計つきキャッシュが無い')
        return
    out = {}

    # ── 1. AU の δ 感度と、絶対 / 正規化の比較 ─────────────────────────
    print("=== 1. AU の δ 感度（再学習なし、5 seeds 平均）===")
    print("絶対閾値 Var(μ_j) > δ と、スケール正規化 Var(μ_j) > δ_rel·max Var を比較する")
    for ds in datasets:
        runs = load(ds)
        grid = sorted({m for m, _ in runs})
        print(f"\n  --- {ds}（d̂_ID={iid[ds]['twonn']:.2f}）---")
        hdr = '  ' + f"{'m':>5s}" + ''.join(f"{'δ='+str(d):>10s}" for d in DELTAS) \
              + ''.join(f"{'rel='+str(r):>11s}" for r in (0.01, 0.05))
        print(hdr)
        rows = []
        for m in grid:
            vs = [runs[(m, s)]['readouts']['B']['mu_var_per_dim']
                  for s in sorted({s for mm, s in runs if mm == m})]
            if not vs:
                continue
            a = [np.mean([au_abs(v, d) for v in vs]) for d in DELTAS]
            r = [np.mean([au_rel(v, t) for v in vs]) for t in (0.01, 0.05)]
            rows.append({'m': m, 'au_abs': a, 'au_rel': r})
            print('  ' + f"{m:5d}" + ''.join(f"{x:10.1f}" for x in a)
                  + ''.join(f"{x:11.1f}" for x in r))
        out.setdefault(ds, {})['delta_sensitivity'] = rows
        # δ を変えたときに AU が m に依存するようになるか
        for j, d in enumerate(DELTAS):
            seq = [row['au_abs'][j] for row in rows]
            print(f"    δ={d}: AU の範囲 {min(seq):.1f}–{max(seq):.1f}  "
                  f"m 依存の幅 {max(seq)-min(seq):.1f}")

    # ── 2. V1 / V2 の再評価 ───────────────────────────────────────────
    print("\n\n=== 2. V1 / V2 の再評価（δ=0.01、5 seeds 平均）===")
    print("V1: 活性割合 AU/m が閾値以上か   V2: AU/d̂_ID が下側・上側係数の間か")
    for ds in datasets:
        runs = load(ds)
        d_hat = iid[ds]['twonn']
        grid = sorted({m for m, _ in runs})
        print(f"\n  --- {ds}（d̂_ID={d_hat:.2f}）---")
        print(f"  {'m':>5s} {'AU':>6s} {'AU/m':>7s} {'AU/d̂':>7s}  "
              + ' '.join(f"V1≥{t}" for t in V1_THRESHOLDS))
        rows = []
        for m in grid:
            aus = [runs[(m, s)]['readouts']['B']['au']
                   for s in sorted({s for mm, s in runs if mm == m})]
            au = float(np.mean(aus))
            v1 = [au / m >= t for t in V1_THRESHOLDS]
            rows.append({'m': m, 'au': au, 'au_over_m': au / m,
                         'au_over_d': au / d_hat,
                         'V1_pass': v1,
                         'V2_lower_pass': [au / d_hat >= t for t in V2_LOWER],
                         'V2_upper_pass': [m / d_hat <= t for t in V2_UPPER]})
            print(f"  {m:5d} {au:6.1f} {au/m:7.2f} {au/d_hat:7.2f}  "
                  + ' '.join('  ○ ' if x else '  × ' for x in v1))
        out.setdefault(ds, {})['v1_v2'] = rows
        aud = [r['au_over_d'] for r in rows]
        print(f"    AU/d̂ の範囲: {min(aud):.2f}–{max(aud):.2f}  "
              f"（原稿の V2 下側不変域は (0.70, 1.0]）")

    # ── 3. 品質許容誤差 ε_D の水準別 ─────────────────────────────────
    print("\n\n=== 3. 品質許容誤差 ε_D の水準と選択 m（anchor はグリッド最大点）===")
    for ds in datasets:
        runs = load(ds)
        grid = sorted({m for m, _ in runs})
        seeds = sorted({s for _, s in runs})
        print(f"\n  --- {ds} ---")
        print(f"  {'ε_D(相対)':>10s} {'選択 m（seed 別）':>28s} {'平均':>7s}")
        rows = []
        for eps in EPS_LEVELS:
            sel = []
            for s in seeds:
                Da = runs[(grid[-1], s)]['val']['mse']
                T = Da * (1 + eps)
                ok = [m for m in grid if runs[(m, s)]['val']['mse'] <= T]
                sel.append(min(ok) if ok else grid[-1])
            rows.append({'eps': eps, 'selected': sel, 'mean': float(np.mean(sel))})
            print(f"  {eps:10.2f} {str(sel):>28s} {np.mean(sel):7.1f}")
        out.setdefault(ds, {})['epsilon_sensitivity'] = rows

    # ── 4. 初期範囲の係数 c（事後的な探索シミュレーション）───────────
    print("\n\n=== 4. 初期範囲の係数 c の感度 ===")
    print("**これは固定グリッドのログ上での事後的な探索シミュレーションである。**")
    print("実時間の計測を伴う実行例は段階 4 の主比較（提案法）を参照すること。")
    for ds in datasets:
        runs = load(ds)
        d_hat = iid[ds]['twonn']
        grid = sorted({m for m, _ in runs})
        seeds = sorted({s for _, s in runs})
        print(f"\n  --- {ds}（d̂_ID={d_hat:.2f}）---")
        print(f"  {'c':>5s} {'初期点':>7s} {'窓の点数':>9s} {'Q 達成率':>9s} {'選択 m 平均':>11s}")
        rows = []
        for c in C_VALUES:
            init = next((g for g in grid if g >= c * d_hat), grid[-1])
            win = [g for g in grid if g <= init]
            hit, sel = 0, []
            for s in seeds:
                Da = runs[(grid[-1], s)]['val']['mse']
                T = Da * 1.10
                ok = [m for m in win if runs[(m, s)]['val']['mse'] <= T]
                if ok:
                    hit += 1; sel.append(min(ok))
                else:
                    sel.append(grid[-1])
            rows.append({'c': c, 'init': init, 'window_size': len(win),
                         'q_hit_rate': hit / len(seeds), 'mean_selected': float(np.mean(sel))})
            print(f"  {c:5.1f} {init:7d} {len(win):9d} {hit/len(seeds):9.2f} {np.mean(sel):11.1f}")
        out.setdefault(ds, {})['c_sensitivity'] = rows

    with open(os.path.join(OUT, 'stage5_e4_thresholds.json'), 'w') as f:
        json.dump({'_spec': {'deltas': DELTAS, 'c_values': C_VALUES,
                             'eps_levels': EPS_LEVELS, 'V1': V1_THRESHOLDS,
                             'V2_lower': V2_LOWER, 'V2_upper': V2_UPPER,
                             'note': 'c の比較は事後的な探索シミュレーション'},
                   'by_dataset': out}, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage5_e4_thresholds.json")


if __name__ == '__main__':
    main()
