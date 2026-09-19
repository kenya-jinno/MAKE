"""段階 4 の解析: 選択結果・品質・費用の比較（方針 §10 段階 4 の完了条件）。

守る事項:
  - 予算超過（BUDGET_EXHAUSTED）で終わった試行を平均から黙って除外しない。終了状態の内訳を必ず出す。
  - mean±SD だけでなく seed ごとの選択 m の分布も示す（方針 §5.1）。
  - 目的関数や事前分布が異なる方法間で学習損失の値を直接順位づけしない。
    共通の再構成品質（画素平均 MSE）・下流性能・費用で比較する。
  - AU が減っただけではモデルサイズや推論費用が減ったことにならない。

実行: プロジェクト直下で python3 src/make49/stage4_analysis.py
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')

ORDER = ['B1_full_mse', 'B2_full_elbo', 'B3_full_downstream', 'B4_coarse', 'B5_fixed32',
         'B6a_fondue', 'B6b_fondue_var', 'B6b_fondue_var_keepmixed_false',
         'proposed', 'proposed_minus_AU', 'ascending_scan_Q',
         'B4_plus_Q', 'B6a_plus_Q', 'B6b_plus_Q']
JP = {'B1_full_mse': 'B1 全探索(val MSE)', 'B2_full_elbo': 'B2 全探索(val ELBO)',
      'B3_full_downstream': 'B3 全探索(下流)', 'B4_coarse': 'B4 粗い探索',
      'B5_fixed32': 'B5 固定 m=32', 'B6a_fondue': 'B6a FONDUE',
      'B6b_fondue_var': 'B6b FONDUE-VAR(keep_mixed=T)',
      'B6b_fondue_var_keepmixed_false': 'B6b FONDUE-VAR(keep_mixed=F)',
      'proposed': '提案法(ID+Q)', 'proposed_minus_AU': '提案法 −AU',
      'ascending_scan_Q': '昇順走査+Q（ID なし）',
      'B4_plus_Q': 'B4+Q', 'B6a_plus_Q': 'B6a+Q', 'B6b_plus_Q': 'B6b+Q'}


def main():
    d = json.load(open(os.path.join(OUT, 'stage4_methods.json')))
    res = d['results']
    datasets = sorted({r['dataset'] for r in res})

    summary = []
    for ds in datasets:
        sub = [r for r in res if r['dataset'] == ds]
        print(f"\n{'='*100}\n=== {ds} ===\n{'='*100}")
        print(f"{'方法':32s} {'選択 m':>14s} {'val MSE':>16s} {'test MSE':>10s} "
              f"{'総時間(s)':>11s} {'VAE':>4s} {'補助':>5s} 終了状態の内訳")
        for name in ORDER:
            rs = [r for r in sub if r['method'] == name]
            if not rs:
                continue
            ms = [r['selected_m'] for r in rs if r['selected_m'] is not None]
            states = Counter(r['termination_state'] for r in rs)
            fin = [r['final_model'] for r in rs if r.get('final_model')]
            vm = [f['val']['mse'] for f in fin]
            tm = [f['test']['mse'] for f in fin]
            secs = [r['total_seconds'] for r in rs]
            runs = [r['training_runs'] for r in rs]
            aux = [r.get('auxiliary_runs', 0) for r in rs]
            mstr = (f"{np.mean(ms):.1f}±{np.std(ms, ddof=1):.1f}" if len(ms) > 1
                    else (str(ms[0]) if ms else '—'))
            row = {'dataset': ds, 'method': name,
                   'selected_m': ms, 'selected_m_mean': float(np.mean(ms)) if ms else None,
                   'selected_m_sd': float(np.std(ms, ddof=1)) if len(ms) > 1 else None,
                   'val_mse_mean': float(np.mean(vm)) if vm else None,
                   'val_mse_sd': float(np.std(vm, ddof=1)) if len(vm) > 1 else None,
                   'test_mse_mean': float(np.mean(tm)) if tm else None,
                   'total_seconds_mean': float(np.mean(secs)),
                   'training_runs_mean': float(np.mean(runs)),
                   'auxiliary_runs_mean': float(np.mean(aux)),
                   'termination_states': dict(states), 'n_seeds': len(rs)}
            summary.append(row)
            vs = (f"{np.mean(vm):.5f}±{np.std(vm, ddof=1):.5f}" if len(vm) > 1
                  else (f"{vm[0]:.5f}" if vm else '—'))
            ts = f"{np.mean(tm):.5f}" if tm else '—'
            st = ', '.join(f'{k}×{v}' for k, v in states.items())
            print(f"{JP[name]:32s} {mstr:>14s} {vs:>16s} {ts:>10s} "
                  f"{np.mean(secs):11.1f} {np.mean(runs):4.1f} {np.mean(aux):5.1f} {st}")

        print(f"\n--- seed ごとの選択 m（mean±SD だけで判断しない）---")
        for name in ORDER:
            rs = sorted([r for r in sub if r['method'] == name], key=lambda r: r['seed'])
            if rs:
                print(f"  {JP[name]:32s} {[r['selected_m'] for r in rs]}")

        # +Q の増分
        print(f"\n--- +Q 対照の増分（Q の効果と ID/AU の効果を分離する）---")
        base = {r['method']: r for r in summary if r['dataset'] == ds}
        for pair, label in [(('B4_coarse', 'B4_plus_Q'), 'B4'),
                            (('B6a_fondue', 'B6a_plus_Q'), 'B6a'),
                            (('B6b_fondue_var', 'B6b_plus_Q'), 'B6b')]:
            a, b = base.get(pair[0]), base.get(pair[1])
            if a and b and a['val_mse_mean'] and b['val_mse_mean']:
                print(f"  {label:5s}: m {a['selected_m_mean']:.1f} -> {b['selected_m_mean']:.1f}  "
                      f"val MSE {a['val_mse_mean']:.5f} -> {b['val_mse_mean']:.5f}  "
                      f"時間 {a['total_seconds_mean']:.0f}s -> {b['total_seconds_mean']:.0f}s")
        p, pm = base.get('proposed'), base.get('proposed_minus_AU')
        if p and pm:
            same_m = p['selected_m'] == pm['selected_m']
            print(f"  提案法 −AU 対照: 選択 m が一致={same_m}  "
                  f"時間 {p['total_seconds_mean']:.0f}s -> {pm['total_seconds_mean']:.0f}s")
            print(f"    （AU は選択の分岐に使われないので m は変わらないはず。"
                  f"変われば実装が仕様と食い違っている）")

    with open(os.path.join(OUT, 'stage4_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage4_summary.json")


if __name__ == '__main__':
    main()
