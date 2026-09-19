"""開始幅のアブレーション対照（MAKE51 残作業 §2 項目 2）。

問題: 現行の Midpoint local は、ID の信頼性判定も fallback も使わない別方策である。
そのため ID local と比べると「開始幅の効果」と「判定・fallback の効果」が混ざる。

本対照 'ID gated midpoint' は、**信頼性判定と fallback を ID local と同一に保ち、
ID を採用したときの開始幅だけを midpoint へ置き換える**。
ID local との差は開始幅のみになるので、開始幅の寄与を分離できる。

参照 bank の費用は、判定に必要なので両者に等しく計上する。
これは「開始幅のアブレーション」であり、ID 情報をまったく使わない実用手順
（Midpoint local, Ascending Q）とは区別する。

実行: プロジェクト直下で python3 src/make52/start_width_ablation.py
"""
import importlib.util, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# replay_search.py を副作用（CSV 書き出し）なしで読み込むため、関数だけ取り出す
spec = importlib.util.spec_from_file_location('rs', ROOT / 'src/make51/replay_search.py')
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

OUT = ROOT / 'results/make52'
OUT.mkdir(exist_ok=True)


def replay_gated_midpoint(ds, s, b, c=2, R=.15, A=.25):
    """ID local と同一の判定・fallback を持ち、開始幅のみ midpoint に置き換える。"""
    grid = rs.CFG['candidate_grids'][ds]; anchor = grid[-1]
    threshold = 1.1 * rs.candidate(ds, anchor, s, b)['val']['mse']
    vals = {m: rs.candidate(ds, m, s, b)['val']['mse'] for m in grid}
    feasible = {m: vals[m] <= threshold for m in grid}
    optimum = next(m for m in grid if feasible[m]); asked = [anchor]
    passed = rs.ID[ds]['relative_range'] <= R and rs.ID[ds]['estimator_rel_diff'] <= A
    fallback = not passed          # ID local と同一の判定

    def request(m):
        if m not in asked: asked.append(m)
        return feasible[m]

    def ascending():
        for m in grid:
            if request(m): return m

    def local(start):
        ix = grid.index(start)
        if request(start):
            selected = start
            for m in reversed(grid[:ix]):
                if not request(m): break
                selected = m
            return selected
        for m in grid[ix + 1:]:
            if request(m): return m
        raise AssertionError('anchor must satisfy Q')

    if fallback:
        chosen = ascending()       # ID local と同一の fallback
    else:
        chosen = local(grid[(len(grid) - 1) // 2])   # 開始幅だけが違う

    candidate_cost = sum(rs.candidate(ds, m, s, b)['elapsed_seconds'] for m in asked)
    bank = rs.BANK[ds]             # 判定に必要なので計上する
    return dict(dataset=ds, seed=s, beta=b, policy='ID gated midpoint',
                id_accepted=bool(passed), fallback=bool(fallback),
                selected_m=chosen, full_grid_min=optimum, exact=chosen == optimum,
                quality_met=feasible[chosen], requested_count=len(asked),
                candidate_seconds=candidate_cost, reference_seconds=bank,
                cold_seconds=candidate_cost + bank)


def main():
    rows = [replay_gated_midpoint(d, s, b)
            for d in rs.DS for b in rs.BETAS for s in rs.SEEDS]
    idl = {(r['dataset'], r['seed'], r['beta']): r
           for r in rs.ROWS if r['policy'] == 'ID local'}

    print("=" * 92)
    print("表 S4  開始幅のアブレーション（判定と fallback を固定し、開始幅のみ変更）")
    print("=" * 92)
    print("ID local と ID gated midpoint の差は **ID 採用時の開始幅のみ**。")
    print("両者とも参照 bank の費用を計上する。\n")
    acc = [r for r in rows if r['id_accepted']]
    rej = [r for r in rows if not r['id_accepted']]
    print(f"{'層':26s} {'条件数':>6s} {'ID local 一致':>14s} {'gated midpoint 一致':>20s}")
    for name, sub in [('ID 採用', acc), ('ID 不採用（fallback）', rej), ('全体', rows)]:
        a = sum(1 for r in sub if idl[(r['dataset'], r['seed'], r['beta'])]['exact'])
        g = sum(1 for r in sub if r['exact'])
        print(f"{name:26s} {len(sub):6d} {f'{a}/{len(sub)}':>14s} {f'{g}/{len(sub)}':>20s}")

    print(f"\n{'層':26s} {'方策':22s} {'候補数':>8s} {'cold 秒':>10s}")
    for name, sub in [('ID 採用', acc), ('ID 不採用（fallback）', rej), ('全体', rows)]:
        ks = [(r['dataset'], r['seed'], r['beta']) for r in sub]
        a = [idl[k] for k in ks]
        print(f"{name:26s} {'ID local':22s} "
              f"{np.mean([x['requested_count'] for x in a]):8.2f} "
              f"{np.mean([x['cold_seconds'] for x in a]):10.1f}")
        print(f"{'':26s} {'ID gated midpoint':22s} "
              f"{np.mean([x['requested_count'] for x in sub]):8.2f} "
              f"{np.mean([x['cold_seconds'] for x in sub]):10.1f}")

    diff = [r for r in rows
            if r['selected_m'] != idl[(r['dataset'], r['seed'], r['beta'])]['selected_m']]
    print(f"\n選択幅が異なった条件: {len(diff)} / {len(rows)}")
    for r in diff:
        o = idl[(r['dataset'], r['seed'], r['beta'])]
        print(f"  {r['dataset']}, seed {r['seed']}, β={r['beta']:g}: "
              f"ID local {o['selected_m']} → gated midpoint {r['selected_m']} "
              f"(full-grid 最小 {r['full_grid_min']})")

    json.dump({'_spec': {'purpose': 'start-width ablation; reliability check and fallback '
                                    'held identical to ID local, only the accepted-case '
                                    'start width replaced by the grid midpoint',
                         'bank_charged_to_both': True},
               'rows': rows}, open(OUT / 'start_width_ablation.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/start_width_ablation.json")


if __name__ == '__main__':
    main()
