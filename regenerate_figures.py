"""
Regenerate publication-quality figures from cached JSON results.
- Larger fonts (axis labels ≥14pt, titles 15pt, legends 12pt, tick labels 12pt)
- DPI=200
- EB (Fig.4): 1×4  →  2×2 layout
- EC (Fig.5): 1×4  →  2×2 layout
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'results', 'figures')
TAB_DIR = os.path.join(BASE, 'results', 'tables')


def load(name):
    with open(os.path.join(TAB_DIR, name)) as f:
        return json.load(f)


# ── global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 15,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})
DPI = 200


def save(name):
    path_pdf = os.path.join(FIG_DIR, name + '.pdf')
    path_png = os.path.join(FIG_DIR, name + '.png')
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved {name}.pdf / .png')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.1  E1: MSE vs m  (Swiss Roll + Torus, 2×2)
# ──────────────────────────────────────────────────────────────────────────────
def fig_e1():
    data = load('E1_multiseed.json')
    manifolds = [('SwissRoll', 'Swiss Roll', 2), ('Torus', 'Torus', 2)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('E1: MSE & Metrics vs Bottleneck Dim $m$')

    for col, (key, label, dtrue) in enumerate(manifolds):
        d = data[key]
        m_vals = sorted(int(k) for k in d.keys())
        mse_mean = [d[str(m)]['mse_mean'] for m in m_vals]
        mse_std  = [d[str(m)]['mse_std']  for m in m_vals]
        au_mean  = [d[str(m)]['au_mean']  for m in m_vals]
        au_std   = [d[str(m)]['au_std']   for m in m_vals]
        id_mean  = [d[str(m)]['id_mean']  for m in m_vals]
        id_std   = [d[str(m)]['id_std']   for m in m_vals]

        # top row: MSE
        ax = axes[0, col]
        ax.errorbar(m_vals, mse_mean, yerr=mse_std, fmt='o-', capsize=4, label='MSE')
        ax.axvline(x=dtrue, color='red', linestyle='--', label=f'$d_{{\\mathrm{{true}}}}={dtrue}$')
        ax.set_yscale('log')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{label}: MSE vs $m$')
        ax.set_xticks(m_vals)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # bottom row: AU & ID
        ax = axes[1, col]
        ax.errorbar(m_vals, au_mean, yerr=au_std, fmt='s-', capsize=4,
                    color='steelblue', label='AU')
        ax.errorbar(m_vals, id_mean, yerr=id_std, fmt='^--', capsize=4,
                    color='coral', label='TwoNN ID')
        ax.axvline(x=dtrue, color='red', linestyle='--', label=f'$d_{{\\mathrm{{true}}}}={dtrue}$')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Value')
        ax.set_title(f'{label}: AU & ID vs $m$')
        ax.set_xticks(m_vals)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save('E1_mse_vs_m')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.2  E5: Trustworthiness vs m
# ──────────────────────────────────────────────────────────────────────────────
def fig_e5():
    data = load('E5_trustworthiness.json')
    entries = [('Swiss Roll', 'Swiss Roll', 2), ('Torus', 'Torus', 2)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('E5: Trustworthiness vs Bottleneck Dim $m$')

    for ax, (key, label, dtrue) in zip(axes, entries):
        stats = data[key]['stats']
        m_vals = sorted(int(k) for k in stats.keys())
        mean = [stats[str(m)]['mean'] for m in m_vals]
        std  = [stats[str(m)]['std']  for m in m_vals]

        ax.errorbar(m_vals, mean, yerr=std, fmt='o-', capsize=4, color='steelblue',
                    label='Trustworthiness')
        ax.axvline(x=dtrue, color='red', linestyle='--', label=f'$d_{{\\mathrm{{true}}}}={dtrue}$')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Trustworthiness ($k=10$)')
        ax.set_title(f'{label} (3 seeds mean±std)')
        ax.set_xticks(m_vals)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save('E5_trustworthiness')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.3  EA: AU saturation (AE vs VAE, Swiss Roll + Torus, 2×2)
# ──────────────────────────────────────────────────────────────────────────────
def fig_ea():
    data = load('EA_au_saturation.json')  # list of 2 manifolds

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('EA: AU Saturation — AE vs VAE ($\\beta=4$)')

    for col, res in enumerate(data):
        name   = res['manifold']
        dtrue  = res['d_true']
        m_vals = res['m_values']
        ae_au  = res['ae_au']
        vae_au = res['vae_au']
        ae_mse = res['ae_mse']
        vae_mse = res['vae_mse']

        # top row: AU
        ax = axes[0, col]
        ax.plot(m_vals, m_vals, 'k--', alpha=0.5, linewidth=1.5, label='AU=m (diagonal)')
        ax.plot(m_vals, ae_au,  'o-', color='steelblue', label='AE')
        ax.plot(m_vals, vae_au, 's-', color='coral',     label='VAE ($\\beta=4$)')
        ax.axvline(x=dtrue, color='green', linestyle=':', linewidth=2,
                   label=f'$d_{{\\mathrm{{true}}}}={dtrue}$')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Active Units')
        ax.set_title(f'{name}: AU vs $m$')
        ax.set_xticks(m_vals)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # bottom row: MSE
        ax = axes[1, col]
        ax.plot(m_vals, ae_mse,  'o-', color='steelblue', label='AE')
        ax.plot(m_vals, vae_mse, 's-', color='coral',     label='VAE ($\\beta=4$)')
        ax.axvline(x=dtrue, color='green', linestyle=':', linewidth=2,
                   label=f'$d_{{\\mathrm{{true}}}}={dtrue}$')
        ax.set_yscale('log')
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{name}: MSE vs $m$')
        ax.set_xticks(m_vals)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save('EA_au_saturation')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.4  EB: Topology AU  (4 manifolds → 2×2)
# ──────────────────────────────────────────────────────────────────────────────
def fig_eb():
    data = load('EB_topology.json')  # list of 4 manifolds

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EB: AU Saturation vs Topological Min Embedding Dim\n($\\beta$-VAE, $\\beta=4$)')
    axes_flat = axes.flatten()

    for ax, res in zip(axes_flat, data):
        m_values  = res['m_values']
        au_values = res['au_values']
        expected  = res['expected_min_embed_dim']
        saturation = res['saturation_m']

        ax.plot(m_values, m_values, 'k--', alpha=0.5, linewidth=1.5, label='AU=m (diagonal)')
        ax.plot(m_values, au_values, 'o-', color='steelblue', linewidth=2,
                markersize=8, label='Active Units')
        ax.axvline(x=expected,   color='red',   linestyle='--', linewidth=2,
                   label=f'Expected $d_{{\\mathrm{{emb}}}}={expected}$')
        ax.axvline(x=saturation, color='green', linestyle=':',  linewidth=2,
                   label=f'AU saturation={saturation}')

        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel('Active Units')
        ax.set_title(res['name'])
        ax.set_xticks(m_values)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save('EB_topology_au')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.5  EC: Isometric AE comparison  (4 metrics → 2×2)
# ──────────────────────────────────────────────────────────────────────────────
def fig_ec():
    raw = load('EC_isometric.json')
    results = raw['results']
    model_names  = [r['model'] for r in results]
    metrics      = ['mse', 'condition_number', 'nsr', 'trustworthiness']
    metric_labels = ['MSE', 'Condition Number $\\kappa$', 'NSR', 'Trustworthiness']
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('EC: AE Variant Comparison on Swiss Roll ($m=2$)')
    axes_flat = axes.flatten()

    for ax, metric, label in zip(axes_flat, metrics, metric_labels):
        values = [r[metric] for r in results]
        bars = ax.bar(model_names, values, color=colors[:len(results)],
                      alpha=0.85, edgecolor='black')
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    save('EC_isometric_comparison')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.6  E4: MNIST metric sweep (MSE only)
# ──────────────────────────────────────────────────────────────────────────────
def fig_e4():
    data = load('E4_extended.json')
    m_vals  = data['latent_dims']
    ae_res  = data['ae_results']
    vae_res = data['vae_results']

    ae_mse  = [r['mse']  for r in ae_res]
    vae_mse = [r['mse']  for r in vae_res]
    vae_au  = [r['au']   for r in vae_res]

    # find m_knee (first m where AU growth rate drops)
    au_arr = np.array(vae_au, dtype=float)
    growth = np.diff(au_arr)
    r_max  = growth.max() if growth.max() > 0 else 1.0
    tau    = 0.25
    m_knee = None
    for i, (g, m) in enumerate(zip(growth, m_vals[1:])):
        if g / r_max < tau:
            m_knee = m
            break

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(m_vals, ae_mse,  'o-', color='steelblue', label='AE')
    ax.plot(m_vals, vae_mse, 's-', color='coral',     label='VAE ($\\beta=4$)')
    if m_knee:
        ax.axvline(x=m_knee, color='green', linestyle='--', linewidth=2,
                   label=f'$m^{{\\mathrm{{knee}}}}={m_knee}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Test MSE (log scale)')
    ax.set_title('E4: MNIST Metric Sweeps (300 epochs)')
    ax.set_xticks(m_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save('E4_extended_mse')


# ──────────────────────────────────────────────────────────────────────────────
# Fig.7  E6: MNIST training dynamics
# ──────────────────────────────────────────────────────────────────────────────
def fig_e6():
    data = load('E6_mnist.json')
    history = data['history']
    epochs  = sorted(int(k) for k in history.keys())

    test_mse      = [history[str(e)]['test_mse']       for e in epochs]
    au            = [history[str(e)]['au']              for e in epochs]
    trust         = [history[str(e)]['trustworthiness'] for e in epochs]
    cka           = [history[str(e)]['cka']             for e in epochs]
    id_twonn      = [history[str(e)]['id_twonn']        for e in epochs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('E6: MNIST Training Dynamics ($m=16$, 200 epochs)')

    def _plot(ax, y, ylabel, title, color='steelblue', logy=False):
        ax.plot(epochs, y, 'o-', color=color, markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    _plot(axes[0,0], test_mse,  'MSE (log scale)',   'Reconstruction MSE',            logy=True)
    _plot(axes[0,1], au,        'Active Units',       'Active Units over Training',    color='coral')
    _plot(axes[0,2], trust,     'Trustworthiness',    'Trustworthiness vs Epoch ($k=10$)', color='seagreen')
    _plot(axes[1,0], cka,       'CKA',                'CKA (Input vs Latent)',         color='orchid')
    _plot(axes[1,1], id_twonn,  'TwoNN ID',           'TwoNN ID of Latent Space',      color='saddlebrown')

    # Global vs Local formation panel
    ax = axes[1,2]
    trust_norm = np.array(trust)
    id_norm    = np.array(id_twonn)
    t_max = trust_norm.max(); t_min = trust_norm.min()
    i_max = id_norm.max();    i_min = id_norm.min()
    if t_max > t_min:
        trust_norm = (trust_norm - t_min) / (t_max - t_min)
    if i_max > i_min:
        id_norm    = (id_norm    - i_min) / (i_max - i_min)

    ax.plot(epochs, trust_norm, 'o-', color='seagreen',    label='Trustworthiness (norm)')
    ax.plot(epochs, id_norm,    's--', color='saddlebrown', label='TwoNN ID (norm)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Global vs Local Structure Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save('E6_mnist_dynamics')


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.ticker
    os.chdir(BASE)
    print('Regenerating figures...')
    fig_e1()
    fig_e5()
    fig_ea()
    fig_eb()
    fig_ec()
    fig_e4()
    fig_e6()
    print('Done.')
