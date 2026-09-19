"""§5.4 の集計: 未マスク 30-epoch 対照と Mask+30ep の対応比較。

出力:
  results/make53/full_control_summary.json
  results/make53/full_control.tex        （本文用の対応比較表）
  results/make53/full_control_seeds.tex  （付録用の seed 別表）
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/make53'
ORDER = ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']
ARMS = [('full', 'Full'), ('mask', 'Mask'),
        ('full_plus30', 'Full + 30 ep'), ('mask_plus30', 'Mask + 30 ep')]


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def load():
    runs = []
    for sc in ('A', 'B'):
        f = OUT / f'full_control_scope{sc}.json'
        if f.exists():
            runs += json.loads(f.read_text())['runs']
    sens = []
    fb = OUT / 'full_control_scopeB.json'
    if fb.exists():
        sens = json.loads(fb.read_text()).get('finetune_rng_sensitivity', [])
    return runs, sens


def summarize(runs, sens):
    out = {'by_scope': {}, 'finetune_rng_sensitivity': {}}
    for sc, variant in (('A', 'zero-centred'), ('B', 'sample-centred')):
        rs = [r for r in runs if r['scope'] == sc]
        if not rs:
            continue
        byds = {}
        for ds in ORDER:
            g = [r for r in rs if r['dataset'] == ds]
            if not g:
                continue
            e = {'n_seeds': len(g), 'selected_m': [r['selected_m'] for r in g],
                 'm_start': g[0]['m_start'],
                 'paired_streams': g[0]['paired_streams']}
            for key, _ in ARMS:
                for suf, lab in (('', 'validation'), ('_test', 'test')):
                    m, s = ms([r[key + suf]['mse'] for r in g])
                    e[f'{key}_{lab}_mean'] = m
                    e[f'{key}_{lab}_sd'] = s
            for nm, fld in (('delta_full_pct', 'delta_full_pct'),
                            ('delta_mask_pct', 'delta_mask_pct'),
                            ('mask_specific_pp', 'mask_specific_pp')):
                m, s = ms([r[fld] for r in g])
                e[nm + '_mean'], e[nm + '_sd'] = m, s
            # マスクによる損失そのもの（追加学習前）
            m, s = ms([100.0 * (r['mask']['mse'] - r['full']['mse']) / r['full']['mse']
                       for r in g])
            e['masking_cost_pct_mean'], e['masking_cost_pct_sd'] = m, s
            # 追加学習後に Mask が Full+30ep に対して残す差
            m, s = ms([100.0 * (r['mask_plus30']['mse'] - r['full_plus30']['mse'])
                       / r['full_plus30']['mse'] for r in g])
            e['residual_gap_after_pct_mean'], e['residual_gap_after_pct_sd'] = m, s
            if sc == 'A':
                # 範囲 A の Mask+30ep 再実行と、§5.2 で記録した Adapt との差。
                # 追加学習の RNG 流だけが違うので、これが雑音下限の実測値になる。
                d = [100.0 * (r['mask_plus30']['mse'] - r['recorded_adapt_mse'])
                     / r['recorded_adapt_mse'] for r in g]
                e['adapt_reproduction_pct'] = d
                e['adapt_reproduction_abs_max_pct'] = float(max(abs(x) for x in d))
            if sc == 'B':
                e['reproduction_all_bitwise_identical'] = all(
                    r['reproduction']['bitwise_identical'] for r in g)
                e['reproduction_max_abs_diff'] = max(
                    r['reproduction']['abs_diff'] for r in g)
            byds[ds] = e
        out['by_scope'][variant] = byds
    if sens:
        allsd = [s['delta_sd_pp'] for s in sens]
        out['finetune_rng_sensitivity'] = {
            'dataset': sens[0]['dataset'], 'finetune_seeds': sens[0]['finetune_seeds'],
            'per_training_seed': sens,
            'max_sd_pp': float(max(allsd)), 'mean_sd_pp': float(np.mean(allsd)),
            'max_range_pp': float(max(s['delta_range_pp'] for s in sens))}
    return out


def fmt(m, s, nd=5):
    return f'{m:.{nd}f} $\\pm$ {s:.{nd}f}'


def tex_main(sm):
    L = [r'\begin{table}[t]', r'\caption{Unmasked 30-epoch control (Section 5.4). '
         r'Validation MSE, mean $\pm$ SD over three training seeds. '
         r'Decoder-only fine-tuning, 18{,}000 examples, identical optimiser and budget '
         r"in both arms. Mask-specific is the difference between the two "
         r'relative changes, in percentage points; a value near zero means the '
         r'improvement is not specific to masking.}',
         r'\label{tab:full-control}', r'\begin{tabularx}{\textwidth}{llXXXXrr}',
         r'\toprule',
         r'Variant & Dataset & Full & Mask & Full + 30 ep & Mask + 30 ep & '
         r'$\Delta$Full (\%) & Mask-specific (pp) \\', r'\midrule']
    for variant, byds in sm['by_scope'].items():
        first = True
        for ds in ORDER:
            e = byds.get(ds)
            if not e:
                continue
            L.append(
                f"{variant if first else ''} & {ds} & "
                f"{fmt(e['full_validation_mean'], e['full_validation_sd'])} & "
                f"{fmt(e['mask_validation_mean'], e['mask_validation_sd'])} & "
                f"{fmt(e['full_plus30_validation_mean'], e['full_plus30_validation_sd'])} & "
                f"{fmt(e['mask_plus30_validation_mean'], e['mask_plus30_validation_sd'])} & "
                f"{e['delta_full_pct_mean']:+.2f} & "
                f"{e['mask_specific_pp_mean']:+.2f} $\\pm$ {e['mask_specific_pp_sd']:.2f} \\\\")
            first = False
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabularx}', r'\end{table}']
    return '\n'.join(L)


def tex_seeds(runs):
    L = [r'\begin{table}[t]', r'\caption{Unmasked 30-epoch control, per seed '
         r'(Section 5.4). Validation MSE.}', r'\label{tab:full-control-seeds}',
         r'\begin{tabularx}{\textwidth}{llcXXXXr}', r'\toprule',
         r'Variant & Dataset & Seed & Full & Mask & Full + 30 ep & Mask + 30 ep & '
         r'Mask-specific (pp) \\', r'\midrule']
    for sc, variant in (('A', 'zero-centred'), ('B', 'sample-centred')):
        for ds in ORDER:
            for r in [x for x in runs if x['scope'] == sc and x['dataset'] == ds]:
                L.append(f"{variant} & {ds} & {r['seed']} & "
                         f"{r['full']['mse']:.5f} & {r['mask']['mse']:.5f} & "
                         f"{r['full_plus30']['mse']:.5f} & {r['mask_plus30']['mse']:.5f} & "
                         f"{r['mask_specific_pp']:+.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    return '\n'.join(L)


def main():
    runs, sens = load()
    sm = summarize(runs, sens)
    sm['n_runs'] = len(runs)
    json.dump(sm, open(OUT / 'full_control_summary.json', 'w'),
              ensure_ascii=False, indent=1)
    (OUT / 'full_control.tex').write_text(tex_main(sm))
    (OUT / 'full_control_seeds.tex').write_text(tex_seeds(runs))

    print(f"§5.4 集計  run 数 {len(runs)}\n")
    hdr = (f"{'変種':<15s}{'データ':<14s}{'Full':>10s}{'Mask':>10s}"
           f"{'Full+30':>10s}{'Mask+30':>10s}{'ΔFull%':>9s}{'ΔMask%':>9s}"
           f"{'固有pp':>9s}")
    print(hdr); print('-' * len(hdr))
    for variant, byds in sm['by_scope'].items():
        for ds in ORDER:
            e = byds.get(ds)
            if not e:
                continue
            print(f"{variant:<15s}{ds:<14s}"
                  f"{e['full_validation_mean']:>10.5f}{e['mask_validation_mean']:>10.5f}"
                  f"{e['full_plus30_validation_mean']:>10.5f}"
                  f"{e['mask_plus30_validation_mean']:>10.5f}"
                  f"{e['delta_full_pct_mean']:>+9.2f}{e['delta_mask_pct_mean']:>+9.2f}"
                  f"{e['mask_specific_pp_mean']:>+9.2f}")
    if sm['finetune_rng_sensitivity']:
        s = sm['finetune_rng_sensitivity']
        print(f"\n微調整 RNG 雑音下限（{s['dataset']}）: "
              f"SD 最大 {s['max_sd_pp']:.3f}pp / 平均 {s['mean_sd_pp']:.3f}pp / "
              f"幅 最大 {s['max_range_pp']:.3f}pp")
    b = sm['by_scope'].get('sample-centred', {})
    if b:
        ok = all(e.get('reproduction_all_bitwise_identical') for e in b.values())
        print(f"決定的再学習の照合: {'全件ビット一致' if ok else '不一致あり'}")
    print(f"\n書き出し: {OUT}/full_control_summary.json, full_control.tex, "
          f"full_control_seeds.tex")


if __name__ == '__main__':
    main()
