"""E7a の解析。設定表 §1.2 の事前登録した判定規則をそのまま適用する。

H1: 学習時間を延ばすと読み取りの seed 間ばらつきが小さくなる
    -> seed 単位ブートストラップ 2000 反復で SD(300)/SD(5) の 95% CI を求め、
       上限が 1 を下回る場合のみ支持。
H2: 平均表現の分散に基づく AU（読み取り B）は、分散表現に基づく変数型分類
    （読み取り A の active+mixed）より安定である
    -> 同じく SD 比の CI で判定する。

守る事項:
  - seed 間の安定性と学習中の安定性を分けて報告する。
  - AU=0 / AU=m の退化条件は SD が小さくなるので H1 の支持証拠に数えない。
  - A/B/C の生の SD を一列に並べて最小を最良と順位づけしない（C は連続量）。

実行: プロジェクト直下で python3 src/make49/stage3_e7a_analysis.py
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
BOOT = 2000
RNG = np.random.default_rng(20260912)
PRIMARY = CFG['E7a']['primary_comparison_epochs']    # [5, 300]
SNAPS = CFG['E7a']['recorded_epochs']


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')


def boot_sd_ratio(a, b, n=BOOT):
    """SD(a)/SD(b) の seed 単位ブートストラップ CI。a,b は同じ seed 順の配列。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = len(a)
    out = []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        sa, sb = np.std(a[idx], ddof=1), np.std(b[idx], ddof=1)
        if sb > 0:
            out.append(sa / sb)
    if len(out) < n * 0.5:
        return {'lo': None, 'hi': None, 'median': None,
                'note': f'分母の SD が 0 になる再標本が多い（有効 {len(out)}/{n}）'}
    q = np.percentile(out, [2.5, 50, 97.5])
    return {'lo': float(q[0]), 'median': float(q[1]), 'hi': float(q[2]),
            'n_effective': len(out)}


def main():
    d = json.load(open(os.path.join(OUT, 'stage3_e7a_e2.json')))
    runs = d['runs']
    print(f"E7a 解析  ラン数={len(runs)}  主要比較={PRIMARY[0]} vs {PRIMARY[1]} epochs  "
          f"ブートストラップ {BOOT} 反復\n")

    # 条件ごとに seed をまとめる
    conds = {}
    for r in runs:
        conds.setdefault((r['dataset'], r['capacity_label'], r['m']), []).append(r)

    rows, h1_rows, h2_rows = [], [], []
    print("=== 読み取り値（seed 平均 ± SD、退化は * 印）===")
    print(f"{'データ':9s} {'容量':8s} {'m':>4s} {'ep':>4s} "
          f"{'A act+mix':>14s} {'B AU':>13s} {'C gap':>14s} {'退化':>5s}")
    for key in sorted(conds):
        ds, lab, m = key
        rs = sorted(conds[key], key=lambda r: r['seed'])
        for ep in SNAPS:
            A = [r['readouts'][str(ep)]['A']['active_plus_mixed'] for r in rs]
            B = [r['readouts'][str(ep)]['B']['au'] for r in rs]
            C = [r['readouts'][str(ep)]['C']['gap'] for r in rs]
            degen = all(b == 0 for b in B) or all(b == m for b in B)
            rows.append({'dataset': ds, 'capacity': lab, 'm': m, 'epoch': ep,
                         'A_active_plus_mixed': A, 'B_au': B, 'C_gap': C,
                         'A_mean': float(np.mean(A)), 'A_sd': sd(A),
                         'B_mean': float(np.mean(B)), 'B_sd': sd(B),
                         'C_mean': float(np.mean(C)), 'C_sd': sd(C),
                         'degenerate': bool(degen)})
            print(f"{ds:9s} {lab:8s} {m:4d} {ep:4d} "
                  f"{np.mean(A):7.1f}±{sd(A):<6.2f} {np.mean(B):6.1f}±{sd(B):<6.2f} "
                  f"{np.mean(C):7.2f}±{sd(C):<6.2f} {'*' if degen else '':>5s}")

    e5, e300 = str(PRIMARY[0]), str(PRIMARY[1])
    print(f"\n=== H1: SD({PRIMARY[1]}) / SD({PRIMARY[0]}) 　上限 < 1 で支持 ===")
    print(f"{'データ':9s} {'容量':8s} {'m':>4s} {'読み取り':>9s} "
          f"{'SD@5':>7s} {'SD@300':>8s} {'比の95%CI':>22s} {'判定':>10s}")
    for key in sorted(conds):
        ds, lab, m = key
        rs = sorted(conds[key], key=lambda r: r['seed'])
        for name, path in [('A act+mix', lambda r, e: r['readouts'][e]['A']['active_plus_mixed']),
                           ('B AU', lambda r, e: r['readouts'][e]['B']['au']),
                           ('C gap', lambda r, e: r['readouts'][e]['C']['gap'])]:
            v5 = [path(r, e5) for r in rs]; v300 = [path(r, e300) for r in rs]
            degen = all(x == 0 for x in v300) or all(x == m for x in v300)
            ci = boot_sd_ratio(v300, v5)
            if ci['lo'] is None:
                verdict = '判定不能'
            elif degen:
                verdict = '退化・除外'
            elif ci['hi'] < 1:
                verdict = '支持'
            elif ci['lo'] > 1:
                verdict = '逆向き'
            else:
                verdict = '不明'
            h1_rows.append({'dataset': ds, 'capacity': lab, 'm': m, 'readout': name,
                            'sd_5': sd(v5), 'sd_300': sd(v300), 'ci': ci,
                            'degenerate': bool(degen), 'verdict': verdict})
            cis = (f"[{ci['lo']:.2f}, {ci['hi']:.2f}]" if ci['lo'] is not None else ci['note'][:20])
            print(f"{ds:9s} {lab:8s} {m:4d} {name:>9s} {sd(v5):7.2f} {sd(v300):8.2f} "
                  f"{cis:>22s} {verdict:>10s}")

    print(f"\n=== H2: SD(A act+mix) / SD(B AU) at {PRIMARY[1]} epochs　上限 < 1 なら A が安定 ===")
    print("（A も B も m 分の個数だが定義が異なる。C は連続量なので同じ列に並べない）")
    print(f"{'データ':9s} {'容量':8s} {'m':>4s} {'SD(A)':>7s} {'SD(B)':>7s} "
          f"{'比の95%CI':>22s} {'判定':>12s}")
    for key in sorted(conds):
        ds, lab, m = key
        rs = sorted(conds[key], key=lambda r: r['seed'])
        A = [r['readouts'][e300]['A']['active_plus_mixed'] for r in rs]
        B = [r['readouts'][e300]['B']['au'] for r in rs]
        ci = boot_sd_ratio(A, B)
        degen = all(x == 0 for x in B) or all(x == m for x in B) or \
                all(x == 0 for x in A) or all(x == m for x in A)
        if ci['lo'] is None:
            verdict = '判定不能'
        elif degen:
            verdict = '退化・除外'
        elif ci['hi'] < 1:
            verdict = 'A が安定'
        elif ci['lo'] > 1:
            verdict = 'B が安定'
        else:
            verdict = '差は不明'
        h2_rows.append({'dataset': ds, 'capacity': lab, 'm': m, 'sd_A': sd(A),
                        'sd_B': sd(B), 'ci': ci, 'degenerate': bool(degen), 'verdict': verdict})
        cis = (f"[{ci['lo']:.2f}, {ci['hi']:.2f}]" if ci['lo'] is not None else ci['note'][:20])
        print(f"{ds:9s} {lab:8s} {m:4d} {sd(A):7.2f} {sd(B):7.2f} {cis:>22s} {verdict:>12s}")

    print("\n=== 学習中の安定性（同一 seed 内、50ep と 300ep の値の差の絶対値）===")
    print("seed 間の安定性とは別の量である。両者を混ぜない。")
    print(f"{'データ':9s} {'容量':8s} {'m':>4s} {'|A50-A300|':>11s} {'|B50-B300|':>11s} {'|C50-C300|':>11s}")
    within = []
    for key in sorted(conds):
        ds, lab, m = key
        rs = sorted(conds[key], key=lambda r: r['seed'])
        dA = [abs(r['readouts']['300']['A']['active_plus_mixed'] - r['readouts']['50']['A']['active_plus_mixed']) for r in rs]
        dB = [abs(r['readouts']['300']['B']['au'] - r['readouts']['50']['B']['au']) for r in rs]
        dC = [abs(r['readouts']['300']['C']['gap'] - r['readouts']['50']['C']['gap']) for r in rs]
        within.append({'dataset': ds, 'capacity': lab, 'm': m,
                       'dA_mean': float(np.mean(dA)), 'dB_mean': float(np.mean(dB)),
                       'dC_mean': float(np.mean(dC))})
        print(f"{ds:9s} {lab:8s} {m:4d} {np.mean(dA):11.2f} {np.mean(dB):11.2f} {np.mean(dC):11.3f}")

    res = {'readout_table': rows, 'H1': h1_rows, 'H2': h2_rows, 'within_training': within,
           '_spec': {'bootstrap': BOOT, 'primary_comparison': PRIMARY,
                     'rule_H1': 'supported only if the upper bound of the SD ratio CI is below 1',
                     'rule_H2': 'A is more stable only if the upper bound of SD(A)/SD(B) is below 1',
                     'degenerate_rule': 'conditions with AU==0 or AU==m at 300 epochs are excluded from support'}}
    with open(os.path.join(OUT, 'stage3_e7a_analysis.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage3_e7a_analysis.json")


if __name__ == '__main__':
    main()
