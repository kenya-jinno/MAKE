"""
E4_300ep_multiseed 図の再生成
- Review_codex35 Minor 6, 10 対応：mknee の per-seed ラベル・縦線を削除し
  Step-1 参照値 dhat=10 と候補範囲 [dhat, 2dhat] を表示
- Peer_Review_Report41 Minor 対応：白黒印刷・2段組縮小に耐えるよう
  シードごとに線種・マーカーを変え，線幅・フォントを拡大
学習は行わず results/tables/E4_300ep_multiseed.json から再描画する．
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

with open('results/tables/E4_300ep_multiseed.json') as f:
    all_results = json.load(f)

seeds = ['42', '123', '777']
m_list = [r['m'] for r in all_results['42']]
DHAT = 10

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
styles = [dict(color='tab:blue', ls='-', marker='o'),
          dict(color='tab:orange', ls='--', marker='s'),
          dict(color='tab:green', ls='-.', marker='^')]

ax = axes[0]
for seed, st in zip(seeds, styles):
    res = all_results[seed]
    ms = [r['m'] for r in res]
    aus = [r['au'] for r in res]
    ax.plot(ms, aus, lw=2, ms=6, label=f'seed={seed}', **st)
ax.plot(m_list, m_list, 'k:', lw=1.5, alpha=0.6, label='AU $=m$')
ax.axvspan(DHAT, 2 * DHAT, color='gray', alpha=0.15,
           label=r'$[\hat{d}_{\mathrm{ID}},\ 2\hat{d}_{\mathrm{ID}}]$')
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('Active Units (AU)')
ax.set_title('VAE AU vs. $m$ (3 seeds, 300 ep)')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2r = ax2.twinx()
all_aus = np.array([[r['au'] for r in all_results[s]] for s in seeds])
all_twons = np.array([[r['twonn'] for r in all_results[s]] for s in seeds])
mu_au = all_aus.mean(0); sd_au = all_aus.std(0)
mu_id = all_twons.mean(0); sd_id = all_twons.std(0)
ax2.fill_between(m_list, mu_au - sd_au, mu_au + sd_au, alpha=0.2, color='tab:blue')
ax2.plot(m_list, mu_au, color='tab:blue', ls='-', marker='o', lw=2, ms=6,
         label='AU mean$\\pm$std')
ax2r.fill_between(m_list, mu_id - sd_id, mu_id + sd_id, alpha=0.2, color='tab:red')
ax2r.plot(m_list, mu_id, color='tab:red', ls='--', marker='s', lw=2, ms=6,
          label='TwoNN ID mean$\\pm$std')
ax2.axhline(DHAT, color='purple', ls=':', lw=2,
            label=r'$\hat{d}_{\mathrm{ID}} \approx 10$ (Step 1)')
ax2.set_xlabel('Bottleneck dimension $m$')
ax2.set_ylabel('AU', color='tab:blue')
ax2r.set_ylabel('TwoNN ID', color='tab:red')
ax2.set_title('Mean$\\pm$std over 3 seeds (VAE, MNIST, 300 ep)')
ax2.set_xscale('log')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2r.tick_params(axis='y', labelcolor='tab:red')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/E4_300ep_multiseed.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/E4_300ep_multiseed.pdf', bbox_inches='tight')
print('Saved: results/figures/E4_300ep_multiseed.{png,pdf}')
