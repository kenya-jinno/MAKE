"""
EN_conv_vae_extended_m 図の再生成（Review_codex35 Minor 6, 10 対応）
- 凡例の per-seed mknee ラベルを削除（mknee は付録の探索的解析のみ）
- フォントを拡大
学習は行わず results/tables/EN_conv_vae_extended_m.json から再描画する．
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

with open('results/tables/EN_conv_vae_extended_m.json') as f:
    d = json.load(f)

seeds = ['42', '123', '777']
colors = ['tab:blue', 'tab:orange', 'tab:green']
DHAT = 10

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for s, c in zip(seeds, colors):
    res = d[s]['results']
    ms = [r['m'] for r in res]
    aus = [r['au'] for r in res]
    ax.plot(ms, aus, 'o-', color=c, ms=5, label=f'seed={s}')
ax.plot(ms, ms, '--', color='gray', lw=1, alpha=0.6, label='AU $=m$ (diagonal)')
ax.axvline(DHAT, color='red', ls='-.', lw=1.5, alpha=0.7,
           label=r'$\hat{d}_{\mathrm{ID}} \approx 10$ (FC-VAE)')
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('Active Units (AU)')
ax.set_title('Conv-VAE extended $m$ grid (KL annealing, MNIST)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for s, c in zip(seeds, colors):
    res = d[s]['results']
    ms = [r['m'] for r in res]
    tw = [r['twonn'] for r in res]
    ax2.plot(ms, tw, 'o-', color=c, ms=5, label=f'TwoNN (seed={s})')
ax2.axhline(DHAT, color='red', ls='-.', lw=1.5, alpha=0.7,
            label=r'$\hat{d}_{\mathrm{ID}} \approx 10$ (FC-VAE)')
ax2.set_xlabel('Bottleneck dimension $m$')
ax2.set_ylabel('TwoNN ID estimate')
ax2.set_title('Conv-VAE --- TwoNN ID vs. $m$')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/EN_conv_vae_extended_m.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/EN_conv_vae_extended_m.pdf', bbox_inches='tight')
print('Saved: results/figures/EN_conv_vae_extended_m.{png,pdf}')
