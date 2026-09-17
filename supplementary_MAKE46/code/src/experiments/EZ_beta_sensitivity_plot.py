"""
EZ_beta_sensitivity_plot: β-sensitivity figure regeneration (plot only).

Loads the saved results from results/tables/EZ_beta_sensitivity.json
(MNIST VAE, 50 epochs, n=5,000 subset; β ∈ {1, 2, 4, 8}) and regenerates
results/figures/EZ_beta_sensitivity.{png,pdf} with English labels.
No training is performed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/figures', exist_ok=True)

MKNEE_BETA4 = 12  # m_knee for β=4 (AU slowdown point)


def main():
    with open('results/tables/EZ_beta_sensitivity.json') as f:
        data = json.load(f)

    beta_list = data['beta_list']          # [1.0, 2.0, 4.0, 8.0]
    m_list    = data['m_list']             # [4, 8, 12, 16, 20, 32]
    results   = data['results']

    colors = {'1.0': 'tab:gray', '2.0': 'tab:blue',
              '4.0': 'tab:orange', '8.0': 'tab:red'}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for beta in beta_list:
        key = str(beta)
        res = results[key]
        ms  = [r['m']  for r in res]
        aus = [r['au'] for r in res]
        label = f'β={beta}  (AU≈m)' if beta == 1.0 else f'β={beta}'
        ax.plot(ms, aus, 'o-', color=colors[key], lw=2, ms=8, label=label)

    # AU = m diagonal (dotted)
    ax.plot(m_list, m_list, ls=':', color='gray', lw=1.5, alpha=0.8,
            label='AU = m (diagonal)')

    # m_knee (β=4) vertical line + annotation
    ax.axvline(MKNEE_BETA4, color='tab:orange', ls=':', lw=1.5, alpha=0.8)
    au_at_knee = next(r['au'] for r in results['4.0'] if r['m'] == MKNEE_BETA4)
    ax.annotate('$m_{\\mathrm{knee}}$\n(β=4)',
                xy=(MKNEE_BETA4, au_at_knee), xytext=(13.5, 9.0),
                arrowprops=dict(arrowstyle='->', color='tab:orange'),
                fontsize=9, color='tab:orange')

    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('β-sensitivity: MNIST VAE AU($m$) (50ep, n=5,000 subset)')
    ax.set_xticks(m_list)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EZ_beta_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EZ_beta_sensitivity.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EZ_beta_sensitivity.{png,pdf}")


if __name__ == '__main__':
    main()
