"""段階 8: beta を因子とした比較表と図表（B7/B8 を含む最終版）。

生成物:
  Table D2 : 方法別 × beta の選択 m・品質・費用・終了状態（LaTeX）
  Table F  : 閾値を持つ手法と持たない手法の対比（LaTeX）
  Figure G : 選択 m の beta 依存（手法別）
  Figure H : 品質と費用（beta 全段を重ねる）

実行: プロジェクト直下で python3 src/make49/stage8_tables_figures.py
"""
import json, os, sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.make49.common import CFG

OUT = os.path.join('results', 'make49')
FIG = os.path.join('results', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'lines.linewidth': 1.8})
BETAS = CFG['beta']['ladder']
DS = ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']

KEY = ['B1_full_mse', 'B4_coarse', 'B5_fixed32', 'B6a_fondue', 'B6b_fondue_var',
       'proposed', 'ascending_scan_Q', 'B6a_plus_Q', 'B7_geco_l0', 'B8_ard_vae']
LBL = {'B1_full_mse': 'B1 full grid', 'B4_coarse': 'B4 coarse', 'B5_fixed32': 'B5 fixed 32',
       'B6a_fondue': 'B6a FONDUE', 'B6b_fondue_var': 'B6b FONDUE-VAR',
       'proposed': 'proposed (ID+Q)', 'ascending_scan_Q': 'ascending scan+Q',
       'B6a_plus_Q': 'B6a+Q', 'B7_geco_l0': 'B7 GECO+L0', 'B8_ard_vae': 'B8 ARD-VAE'}
MK = {'B1_full_mse': 'o', 'B4_coarse': 'v', 'B5_fixed32': 'P', 'B6a_fondue': 's',
      'B6b_fondue_var': 'D', 'proposed': '*', 'ascending_scan_Q': 'X',
      'B6a_plus_Q': '^', 'B7_geco_l0': 'h', 'B8_ard_vae': '8'}


def load_all():
    """段階 4/8 の方法結果と、段階 7 の B7/B8 を 1 つの表に統合する。"""
    rows = []
    p = os.path.join(OUT, 'stage4_methods.json')
    if os.path.exists(p):
        for r in json.load(open(p))['results']:
            rows.append({'dataset': r['dataset'], 'method': r['method'], 'seed': r['seed'],
                         'beta': r.get('beta', 1.0), 'selected_m': r['selected_m'],
                         'val_mse': (r['final_model']['val']['mse'] if r.get('final_model') else None),
                         'seconds': r['total_seconds'], 'state': r['termination_state']})
    p = os.path.join(OUT, 'stage7_b7b8.json')
    if os.path.exists(p):
        for r in json.load(open(p))['results']:
            rows.append({'dataset': r['dataset'], 'method': r['method'], 'seed': r['seed'],
                         'beta': r['beta_context'], 'selected_m': r['selected_m'],
                         'val_mse': (r['final_model']['val']['mse'] if r.get('final_model') else None),
                         'seconds': r['total_seconds'], 'state': r['termination_state']})
    return rows


def agg(rows, ds, meth, beta):
    rs = [r for r in rows if r['dataset'] == ds and r['method'] == meth and r['beta'] == beta]
    if not rs:
        return None
    ms = [r['selected_m'] for r in rs if r['selected_m'] is not None]
    vs = [r['val_mse'] for r in rs if r['val_mse'] is not None]
    return {'n': len(rs),
            'm_mean': float(np.mean(ms)) if ms else None,
            'm_sd': float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
            'mse': float(np.mean(vs)) if vs else None,
            'sec': float(np.mean([r['seconds'] for r in rs])),
            'states': dict(Counter(r['state'] for r in rs))}


def table_D2(rows):
    lines = ['% Table D2: method x beta comparison (5 seeds each)',
             '% cols: method & beta & selected m & val MSE & total s & termination']
    for ds in DS:
        lines.append(f'% ===== {ds} =====')
        for meth in KEY:
            for b in BETAS:
                a = agg(rows, ds, meth, b)
                if not a:
                    continue
                m = (f"${a['m_mean']:.1f} \\pm {a['m_sd']:.1f}$" if a['m_mean'] is not None else '---')
                q = f"{a['mse']:.5f}" if a['mse'] else '---'
                st = ', '.join(f"{k}$\\times${v}" for k, v in a['states'].items())
                lines.append(f"{LBL[meth]} & {b:g} & {m} & {q} & {a['sec']:.0f} & {st} \\\\")
    p = os.path.join(FIG, 'MAKE49_TableD2_method_beta.tex')
    open(p, 'w').write('\n'.join(lines) + '\n')
    print('表:', p, f'({len(lines)} 行)')


def table_F(rows):
    """閾値を持つ手法と持たない手法の対比。beta 幅で示す。"""
    lines = ['% Table F: threshold-governed vs threshold-free methods',
             '% cols: method & dataset & span of selected m over the beta ladder']
    summary = []
    for meth in KEY:
        spans = []
        for ds in DS:
            ms = [agg(rows, ds, meth, b) for b in BETAS]
            ms = [a['m_mean'] for a in ms if a and a['m_mean'] is not None]
            if len(ms) >= 2:
                spans.append((ds, max(ms) - min(ms)))
        if spans:
            summary.append((meth, spans, float(np.mean([s for _, s in spans]))))
            for ds, sp in spans:
                lines.append(f"{LBL[meth]} & {ds} & {sp:.1f} \\\\")
    p = os.path.join(FIG, 'MAKE49_TableF_threshold_span.tex')
    open(p, 'w').write('\n'.join(lines) + '\n')
    print('表:', p)
    print(f"\n  {'手法':20s} " + ''.join(f"{d:>13s}" for d in DS) + f"{'平均':>8s}")
    for meth, spans, mean in sorted(summary, key=lambda t: t[2]):
        d = dict(spans)
        print(f"  {LBL[meth]:20s} " + ''.join(f"{d.get(x, float('nan')):13.1f}" for x in DS)
              + f"{mean:8.1f}")
    print("\n  → beta 梯子にわたる選択 m の幅。0 に近いほど「閾値に支配されていない」。")


def fig_G(rows):
    fig, axes = plt.subplots(1, len(DS), figsize=(5 * len(DS), 4.2))
    show = ['proposed', 'ascending_scan_Q', 'B6a_fondue', 'B7_geco_l0', 'B8_ard_vae']
    for j, ds in enumerate(DS):
        for i, meth in enumerate(show):
            xs, ys, es = [], [], []
            for b in BETAS:
                a = agg(rows, ds, meth, b)
                if a and a['m_mean'] is not None:
                    xs.append(b); ys.append(a['m_mean']); es.append(a['m_sd'])
            if xs:
                axes[j].errorbar(xs, ys, yerr=es, marker=MK[meth], capsize=3,
                                 color=f'{i * 0.18}', label=LBL[meth] if j == 0 else None)
        axes[j].set_xscale('log'); axes[j].set_yscale('log')
        axes[j].set_xlabel('beta'); axes[j].set_ylabel('selected m')
        axes[j].set_title(ds); axes[j].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Figure G: how the selected dimension moves with beta\n'
                 'ARD-VAE is flat because it has no free threshold; the others do '
                 '(5 seeds, error bars are SD)', y=1.04)
    fig.tight_layout()
    p = os.path.join(FIG, 'MAKE49_FigG_beta_dependence.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('図:', p)


def fig_H(rows):
    fig, axes = plt.subplots(1, len(DS), figsize=(5 * len(DS), 4.2))
    for j, ds in enumerate(DS):
        for i, meth in enumerate(KEY):
            xs, ys = [], []
            for b in BETAS:
                a = agg(rows, ds, meth, b)
                if a and a['mse'] and a['sec']:
                    xs.append(a['sec']); ys.append(a['mse'])
            if xs:
                axes[j].scatter(xs, ys, marker=MK[meth], s=55, c=f'{(i % 5) * 0.16}',
                                label=LBL[meth] if j == 0 else None, alpha=0.85)
        axes[j].set_xscale('log'); axes[j].set_yscale('log')
        axes[j].set_xlabel('total wall-clock (s, log)'); axes[j].set_ylabel('val MSE (log)')
        axes[j].set_title(ds); axes[j].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Figure H: quality against cost, all beta rungs overlaid '
                 '(5 seeds per point)', y=1.03)
    fig.tight_layout()
    p = os.path.join(FIG, 'MAKE49_FigH_quality_cost_beta.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('図:', p)


if __name__ == '__main__':
    rows = load_all()
    print(f"統合レコード {len(rows)} 件\n")
    table_D2(rows); table_F(rows); fig_G(rows); fig_H(rows)
