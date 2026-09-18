"""段階 7: beta 梯子全体での再解析（B7/B8 を含む）。

段階 4・5 の解析を beta を因子として回し直し、B7（GECO+L0）と B8（ARD-VAE）を統合する。
**再学習は行わない。** すべて既存キャッシュから計算する。

実行: プロジェクト直下で python3 src/make49/stage7_analysis.py
"""
import json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
CACHE = os.path.join(OUT, 'stage4_cache')
BETAS = CFG['beta']['ladder']
DELTAS = CFG['au']['delta_sensitivity']
REL = CFG['quality_criterion_Q']['epsilon_D']['rel_primary']
DS = ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']


def load(ds, beta):
    p = os.path.join(CACHE, f'{ds}.json')
    if not os.path.exists(p):
        return {}
    st = json.load(open(p))
    out = {}
    grid = set(CFG['candidate_grids'][ds])
    for k, v in st.items():
        if not (k.startswith('vae|') and '|e300|' in k and k.endswith(f'|b{beta:g}')):
            continue
        if 'mu_var_per_dim' not in v.get('readouts', {}).get('B', {}):
            continue
        # FONDUE 等は格子外の次元も最終学習するので、解析は候補グリッドに限定する
        if v['m'] not in grid:
            continue
        out[(v['m'], v['seed'])] = v
    return out


def sec1_au_vs_beta(res):
    print("=" * 96)
    print("1. AU は beta で決まるか（delta=0.01、5 seeds 平均、容量ごと）")
    print("=" * 96)
    for ds in DS:
        grid = sorted({m for b in BETAS for m, _ in load(ds, b)})
        if not grid:
            continue
        print(f"\n--- {ds} ---")
        print(f"  {'m':>5s} " + ''.join(f"{'b='+str(b):>9s}" for b in BETAS))
        rows = []
        for m in grid:
            cells = []
            for b in BETAS:
                r = load(ds, b)
                aus = [v['readouts']['B']['au'] for (mm, s), v in r.items() if mm == m]
                cells.append(np.mean(aus) if aus else float('nan'))
            rows.append({'m': m, 'au_by_beta': cells})
            print(f"  {m:5d} " + ''.join(f"{c:9.1f}" for c in cells))
        # m 方向・beta 方向の変動幅を比較
        arr = np.array([r['au_by_beta'] for r in rows], float)
        sp_m = np.nanmax(arr, 0) - np.nanmin(arr, 0)       # 各 beta での m 依存の幅
        sp_b = np.nanmax(arr, 1) - np.nanmin(arr, 1)       # 各 m での beta 依存の幅
        print(f"  → m 依存の幅（beta 別）平均 {np.nanmean(sp_m):6.1f} / "
              f"beta 依存の幅（m 別）平均 {np.nanmean(sp_b):6.1f}")
        res.setdefault(ds, {})['au_vs_beta'] = rows


def sec2_delta_across_beta(res):
    print("\n" + "=" * 96)
    print("2. delta 感度は beta 全域で成り立つか（AU が m と一致する割合＝遮断が全く起きない割合）")
    print("=" * 96)
    print(f"  {'データ':13s} {'delta':>7s} " + ''.join(f"{'b='+str(b):>9s}" for b in BETAS))
    for ds in DS:
        for d in DELTAS:
            cells = []
            for b in BETAS:
                r = load(ds, b)
                if not r:
                    cells.append(float('nan')); continue
                eq = [int((np.asarray(v['readouts']['B']['mu_var_per_dim']) > d).sum()) == v['m']
                      for v in r.values()]
                cells.append(100.0 * np.mean(eq))
            print(f"  {ds:13s} {d:7.3f} " + ''.join(f"{c:8.0f}%" for c in cells))
            res.setdefault(ds, {}).setdefault('delta_across_beta', []).append(
                {'delta': d, 'pct_au_eq_m_by_beta': cells})


def sec3_selection_across_beta(res):
    print("\n" + "=" * 96)
    print("3. 品質条件 Q による選択 m（anchor=グリッド最大、eps=10% 相対）— beta 別")
    print("=" * 96)
    print(f"  {'データ':13s} " + ''.join(f"{'b='+str(b):>11s}" for b in BETAS))
    for ds in DS:
        cells = []
        for b in BETAS:
            r = load(ds, b)
            if not r:
                cells.append('—'); continue
            grid = sorted({m for m, _ in r}); seeds = sorted({s for _, s in r})
            sel = []
            for s in seeds:
                if (grid[-1], s) not in r:
                    continue
                Da = r[(grid[-1], s)]['val']['mse']; T = Da * (1 + REL)
                ok = [m for m in grid if (m, s) in r and r[(m, s)]['val']['mse'] <= T]
                sel.append(min(ok) if ok else grid[-1])
            if not sel:
                cells.append('—'); continue
            cells.append(f"{np.mean(sel):5.1f}±{np.std(sel, ddof=1):<4.1f}")
            res.setdefault(ds, {}).setdefault('q_selection_by_beta', []).append(
                {'beta': b, 'selected': sel})
        print(f"  {ds:13s} " + ''.join(f"{c:>11s}" for c in cells))


def sec4_b7_is_Q(res):
    print("\n" + "=" * 96)
    print("4. B7(GECO) の選択は品質目標 tau で説明できるか — Q との同一性")
    print("=" * 96)
    p = os.path.join(OUT, 'stage7_b7b8.json')
    if not os.path.exists(p):
        print("  （B7/B8 の結果が無い）"); return
    d = json.load(open(p))['results']
    print(f"  {'データ':13s} {'手法':11s} " + ''.join(f"{'b='+str(b):>12s}" for b in BETAS)
          + f"{'beta 幅':>9s}")
    for ds in DS:
        for meth in ['B7_geco_l0', 'B8_ard_vae']:
            cells, means = [], []
            for b in BETAS:
                rs = [r for r in d if r['dataset'] == ds and r['method'] == meth
                      and r['beta_context'] == b]
                if rs:
                    ms = [r['selected_m'] for r in rs]
                    cells.append(f"{np.mean(ms):5.1f}±{np.std(ms, ddof=1):<4.1f}")
                    means.append(np.mean(ms))
                else:
                    cells.append('—')
            sp = (max(means) - min(means)) if means else float('nan')
            print(f"  {ds:13s} {meth:11s} " + ''.join(f"{c:>12s}" for c in cells)
                  + f"{sp:9.1f}")
            res.setdefault(ds, {}).setdefault('b7b8_by_beta', []).append(
                {'method': meth, 'means': means, 'beta_span': sp})
    print("\n  B7 は tau（= 各段の anchor から導いた品質目標）で動き、B8 は beta 文脈に不変。")
    print("  B7 の beta 幅が大きいほど、その選択が品質目標に支配されていることを示す。")

    print("\n  --- B7 の制約充足率（5 seeds 中）---")
    print(f"  {'データ':13s} " + ''.join(f"{'b='+str(b):>8s}" for b in BETAS))
    for ds in DS:
        cells = []
        for b in BETAS:
            rs = [r for r in d if r['dataset'] == ds and r['method'] == 'B7_geco_l0'
                  and r['beta_context'] == b]
            cells.append(f"{sum(1 for r in rs if r.get('final_constraint_ma', 1) <= 0)}/5"
                         if rs else '—')
        print(f"  {ds:13s} " + ''.join(f"{c:>8s}" for c in cells))


def main():
    res = {}
    sec1_au_vs_beta(res)
    sec2_delta_across_beta(res)
    sec3_selection_across_beta(res)
    sec4_b7_is_Q(res)
    with open(os.path.join(OUT, 'stage7_analysis.json'), 'w') as f:
        json.dump({'_spec': {'betas': BETAS, 'deltas': DELTAS, 'eps_rel': REL},
                   'by_dataset': res}, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage7_analysis.json")


if __name__ == '__main__':
    main()
