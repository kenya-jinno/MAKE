"""段階 6: 査読上の疑問に答える図表を生成する（方針 §7.4）。

figure は白黒で読めるよう線種・マーカーで区別する。
**図中で d̂_ID を真値のように表示しない。**
caption に seed 数・データ分割・評価モードを記す（tex 側で付与）。

実行: プロジェクト直下で python3 src/make49/stage6_figures.py
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join('results', 'make49')
FIG = os.path.join('results', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'lines.linewidth': 1.8})

KEY = ['B1_full_mse', 'B4_coarse', 'B5_fixed32', 'B6a_fondue', 'B6b_fondue_var',
       'proposed', 'ascending_scan_Q', 'B6a_plus_Q', 'B6b_plus_Q']
LBL = {'B1_full_mse': 'B1 full grid', 'B4_coarse': 'B4 coarse', 'B5_fixed32': 'B5 fixed 32',
       'B6a_fondue': 'B6a FONDUE', 'B6b_fondue_var': 'B6b FONDUE-VAR',
       'proposed': 'proposed (ID+Q)', 'ascending_scan_Q': 'ascending scan+Q (no ID)',
       'B6a_plus_Q': 'B6a+Q', 'B6b_plus_Q': 'B6b+Q'}
MK = {'B1_full_mse': 'o', 'B4_coarse': 'v', 'B5_fixed32': 'P', 'B6a_fondue': 's',
      'B6b_fondue_var': 'D', 'proposed': '*', 'ascending_scan_Q': 'X',
      'B6a_plus_Q': '^', 'B6b_plus_Q': '<'}


def fig_B():
    """品質と総時間、品質と m の関係（R2-3 / 実用的利点）。"""
    s = json.load(open(os.path.join(OUT, 'stage4_summary.json')))
    dss = ['MNIST', 'dSprites', 'CIFAR10']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, ds in enumerate(dss):
        rows = [r for r in s if r['dataset'] == ds and r['method'] in KEY
                and r['val_mse_mean'] and r['selected_m_mean']]
        for r in rows:
            m = r['method']
            col = 'k' if m in ('proposed', 'ascending_scan_Q') else '0.45'
            sz = 170 if m in ('proposed', 'ascending_scan_Q') else 70
            axes[0, j].scatter(r['total_seconds_mean'], r['val_mse_mean'],
                               marker=MK[m], s=sz, c=col,
                               label=LBL[m] if j == 0 else None, zorder=3)
            axes[1, j].scatter(r['selected_m_mean'], r['val_mse_mean'],
                               marker=MK[m], s=sz, c=col, zorder=3)
            if r['selected_m_sd']:
                axes[1, j].errorbar(r['selected_m_mean'], r['val_mse_mean'],
                                    xerr=r['selected_m_sd'], fmt='none',
                                    ecolor='0.6', capsize=3, zorder=2)
        axes[0, j].set_xscale('log'); axes[0, j].set_title(f'{ds}')
        axes[0, j].set_xlabel('total wall-clock (s, log)'); axes[0, j].set_ylabel('val MSE')
        axes[1, j].set_xscale('log'); axes[1, j].set_xlabel('selected m (log)')
        axes[1, j].set_ylabel('val MSE')
        for a in (axes[0, j], axes[1, j]):
            a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7.5, loc='best')
    fig.suptitle('Figure B: quality vs cost (top) and quality vs selected dimension (bottom)\n'
                 'beta=1, 5 seeds; error bars are SD of the selected m across seeds', y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG, 'MAKE49_FigB_quality_cost.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('図:', p)


def fig_C():
    """参照表現による ID の変化（R3-3）。"""
    e3 = json.load(open(os.path.join(OUT, 'stage5_e3_reference.json')))
    dss = list(e3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, ds in enumerate(dss):
        v = e3[ds]
        ns = [r['n'] for r in v['sample_size']]
        axes[0].plot(ns, [r['twonn'] for r in v['sample_size']], '-o',
                     color=f'{i*0.22}', label=f'{ds} TwoNN')
        axes[0].plot(ns, [r['mle'] for r in v['sample_size']], '--s',
                     color=f'{i*0.22}', alpha=0.7, label=f'{ds} MLE')
        if v['pca_space']:
            axes[1].plot([r['pca_dim'] for r in v['pca_space']],
                         [r['twonn'] for r in v['pca_space']], '-o',
                         color=f'{i*0.22}', label=ds)
    axes[0].set_xscale('log'); axes[0].set_xlabel('estimation sample size n (log)')
    axes[0].set_ylabel('estimated ID'); axes[0].legend(fontsize=7.5, ncol=2)
    axes[0].set_title('ID estimate vs sample size (input space)')
    axes[1].set_xscale('log'); axes[1].set_xlabel('PCA preprocessing dimension (log)')
    axes[1].set_ylabel('estimated ID (TwoNN)'); axes[1].legend(fontsize=8)
    axes[1].set_title('ID estimate vs preprocessing representation')
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle('Figure C: the ID estimate is not a property of the data alone\n'
                 'it depends on the representation, the sample size and the estimator', y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG, 'MAKE49_FigC_id_dependence.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('図:', p)


def fig_D():
    """閾値感度（R3-1）。"""
    e4 = json.load(open(os.path.join(OUT, 'stage5_e4_thresholds.json')))['by_dataset']
    dss = [d for d in ['MNIST', 'dSprites', 'CIFAR10'] if d in e4]
    fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 4.2))
    if len(dss) == 1:
        axes = [axes]
    styles = [('-', 'o'), ('--', 's'), (':', '^')]
    for j, ds in enumerate(dss):
        rows = e4[ds]['delta_sensitivity']
        ms = [r['m'] for r in rows]
        for k, d in enumerate([0.001, 0.01, 0.1]):
            ls, mk = styles[k]
            axes[j].plot(ms, [r['au_abs'][k] for r in rows], ls, marker=mk,
                         color=f'{k*0.3}', label=f'delta={d}')
        axes[j].plot(ms, ms, '-', color='0.8', linewidth=1, label='AU = m')
        axes[j].set_xscale('log'); axes[j].set_yscale('log')
        axes[j].set_xlabel('capacity m (log)'); axes[j].set_ylabel('AU (log)')
        axes[j].set_title(ds); axes[j].legend(fontsize=8); axes[j].grid(alpha=0.3)
    fig.suptitle('Figure D: the AU verdict is governed by the threshold delta\n'
                 'on CIFAR-10 the conclusion reverses between delta=0.001 and delta=0.1 '
                 '(beta=1, 5 seeds)', y=1.03)
    fig.tight_layout()
    p = os.path.join(FIG, 'MAKE49_FigD_delta_sensitivity.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('図:', p)


def table_D():
    """方法単位の選択 m・品質・費用・終了状態（R2-3）を LaTeX 表で出す。"""
    s = json.load(open(os.path.join(OUT, 'stage4_summary.json')))
    lines = []
    for ds in ['MNIST', 'dSprites', 'CIFAR10']:
        rows = [r for r in s if r['dataset'] == ds and r['method'] in KEY]
        rows.sort(key=lambda r: (r['val_mse_mean'] is None, r['val_mse_mean'] or 0))
        lines.append(f"% --- {ds} ---")
        for r in rows:
            m = (f"${r['selected_m_mean']:.1f} \\pm {r['selected_m_sd']:.1f}$"
                 if r['selected_m_sd'] is not None else
                 (f"${r['selected_m_mean']:.1f}$" if r['selected_m_mean'] is not None else '---'))
            q = f"{r['val_mse_mean']:.5f}" if r['val_mse_mean'] else '---'
            st = ', '.join(f"{k}$\\times${v}" for k, v in r['termination_states'].items())
            lines.append(f"{LBL[r['method']]} & {m} & {q} & "
                         f"{r['total_seconds_mean']:.0f} & {r['training_runs_mean']:.1f} & "
                         f"{st} \\\\")
    p = os.path.join(FIG, 'MAKE49_TableD_methods.tex')
    with open(p, 'w') as f:
        f.write('% Table D: method-level comparison (beta=1, 5 seeds)\n')
        f.write('% columns: method & selected m & val MSE & total s & training runs & termination\n')
        f.write('\n'.join(lines) + '\n')
    print('表:', p)


if __name__ == '__main__':
    fig_B(); fig_C(); fig_D(); table_D()
