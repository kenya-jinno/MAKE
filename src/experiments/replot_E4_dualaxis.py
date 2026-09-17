"""
E4_mnist_dualaxis 図の再生成（Review_codex35 Minor 6, 10 対応）
- 右パネルの m^knee 縦線・ラベルを削除（mknee は付録の探索的解析のみ）
- 代わりに Step-1 参照値 dhat=10 の水平参照線（AU 軸）を表示
- フォントを拡大して列幅での可読性を改善
学習は行わず results/tables/E4_extended.json から再描画する．
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

with open('results/tables/E4_extended.json') as f:
    d = json.load(f)

ae = d['ae_results']
vae = d['vae_results']
ms_ae = [r['m'] for r in ae]
ms_vae = [r['m'] for r in vae]
DHAT = 10

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: AE ---
ax = axes[0]
axr = ax.twinx()
ax.plot(ms_ae, [r['mse'] for r in ae], 'b-o', lw=2, label='MSE (AE)')
axr.plot(ms_ae, [r['id_twonn'] for r in ae], 'r-s', lw=2, label='TwoNN ID (AE)')
axr.axhline(DHAT, color='green', ls='--', lw=1.2, alpha=0.6)
axr.text(1.2, DHAT + 0.25, r'$\hat{d}_{\mathrm{ID}} \approx 10$',
         color='green', fontsize=11)
for x, lab in [(43, 'PCA\n80%'), (86, 'PCA\n90%'), (152, 'PCA\n95%')]:
    ax.axvline(x, color='gray', ls=':', lw=1)
    ax.text(x * 1.05, 8.5e-3, lab, color='gray', fontsize=9)
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('MSE (log scale)', color='b')
axr.set_ylabel('TwoNN ID', color='r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='y', labelcolor='b')
axr.tick_params(axis='y', labelcolor='r')
axr.set_ylim(0, 12)
ax.set_title(r'AE: MSE \& TwoNN ID vs. $m$' if False else 'AE: MSE & TwoNN ID vs. $m$')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=10, loc='center right')
ax.grid(True, alpha=0.3)

# --- Right: VAE ---
ax2 = axes[1]
ax2r = ax2.twinx()
ax2.plot(ms_vae, [r['mse'] for r in vae], 'b-o', lw=2, label='MSE (VAE)')
ax2r.plot(ms_vae, [r['au'] for r in vae], color='orange', marker='D', lw=2,
          label='AU (VAE)')
ax2r.plot(ms_vae, [r['id_twonn'] for r in vae], 'r--s', lw=1.5,
          label='TwoNN ID (VAE)')
ax2r.axhline(DHAT, color='green', ls='--', lw=1.2, alpha=0.6)
ax2r.text(1.2, DHAT + 0.7, r'$\hat{d}_{\mathrm{ID}} \approx 10$',
          color='green', fontsize=11)
ax2.set_xlabel('Bottleneck dimension $m$')
ax2.set_ylabel('MSE (log scale)', color='b')
ax2r.set_ylabel('AU / TwoNN ID', color='k')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='b')
ax2.set_title(r'VAE ($\beta = 4$): MSE, AU & TwoNN ID vs. $m$')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/E4_mnist_dualaxis.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/E4_mnist_dualaxis.pdf', bbox_inches='tight')
print('Saved: results/figures/E4_mnist_dualaxis.{png,pdf}')
