"""段階 3: 校正実行の実測値から GPU 実時間上限を決める。

設定表 §1.6 の「学習ラン数の上限 = 12 ラン」と整合させる。
1 ラン = 1 候補を 300 epochs（早期停止なし）学習すること。
上限 = 12 ラン分の学習時間 x 余裕係数。

余裕係数は ID 推定・probe 学習・評価・データ読み込みの分であり、
学習ラン以外の固定費に相当する。実測値から算出する。

実行: プロジェクト直下で python3 src/make49/stage3_budget.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
MAX_RUNS = CFG['budget_and_termination']['max_training_runs_per_dataset_method_seed']
MAX_EPOCHS = CFG['budget_and_termination']['max_epochs']
OVERHEAD = 1.25   # ID 推定・probe・評価・読み込みの固定費

with open(os.path.join(OUT, 'stage3_calibration.json')) as f:
    cal = json.load(f)

print(f"基準: {MAX_RUNS} ラン x {MAX_EPOCHS} epochs（早期停止なしの最悪値） x 余裕 {OVERHEAD}")
print(f"（設定表 §1.6 の学習ラン数上限と整合させる。グリッド点数ではなくラン数で決める）\n")
print(f"{'条件':16s} {'s/ep':>7} {'1ラン300ep':>11} {'12ラン':>9} {'推奨上限':>9} {'グリッド':>7}")

budget = {}
for r in cal:
    spe = r['seconds_per_epoch']
    one_run = spe * MAX_EPOCHS
    twelve = one_run * MAX_RUNS
    limit = twelve * OVERHEAD
    budget[r['dataset']] = {
        'calibration_config': {'m': r['m'], 'n_train': r['n_train'], 'n_params': r['n_params']},
        'measured_seconds_per_epoch': spe,
        'measured_elapsed_seconds': r['elapsed_seconds'],
        'measured_epochs_run': r['epochs_run'],
        'stopped_early': r['stopped_early'],
        'plateau': r['plateau'],
        'one_run_300ep_seconds': one_run,
        'max_runs': MAX_RUNS,
        'twelve_runs_seconds': twelve,
        'overhead_factor': OVERHEAD,
        'max_wallclock_seconds_per_dataset_method_seed': round(limit),
        'max_wallclock_hours': round(limit / 3600, 2),
        'grid_size': r['grid_size'],
        'basis': f'{MAX_RUNS} training runs x {MAX_EPOCHS} epochs (no early stop) x {OVERHEAD} overhead',
        'note': 'per-epoch time is nearly independent of m, so one calibration m is used per dataset',
    }
    print(f"{r['name']:16s} {spe:7.3f} {one_run/60:10.1f}分 {twelve/3600:8.2f}h "
          f"{limit/3600:8.2f}h {r['grid_size']:7d}")

total = sum(b['max_wallclock_seconds_per_dataset_method_seed'] for b in budget.values())
print(f"\n1 方法 x 1 seed の 3 データ合計上限: {total/3600:.2f} h")
print(f"参考: 5 seeds x 主要比較法 8 種なら {total*5*8/3600:.0f} h（全データ・全方法の粗い上限）")

with open(os.path.join(OUT, 'stage3_budget.json'), 'w') as f:
    json.dump(budget, f, ensure_ascii=False, indent=1)
print(f"\n書き出し: {OUT}/stage3_budget.json")
