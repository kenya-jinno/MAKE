"""
EN_conv_vae_extended_m 図の v44 再生成（review43.md 項目4 対応）
- 参照線ラベルを "Step-1 ref. d̂_ID = 10" に変更
  （"(FC-VAE)" 表記が候補 FC-VAE 由来の値と誤読されるのを防ぐ；
   参照値は独立に学習した参照 AE 手続きに由来することはキャプションに明記）
- 旧図 EN_conv_vae_extended_m.{png,pdf} は残し，新図は *_v44 として保存
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
REF_LABEL = r'Step-1 ref. $\hat{d}_{\mathrm{ID}} = 10$'

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for s, c in zip(seeds, colors):
    res = d[s]['results']
    ms = [r['m'] for r in res]
    aus = [r['au'] for r in res]
    ax.plot(ms, aus, 'o-', color=c, ms=5, label=f'seed={s}')
ax.plot(ms, ms, '--', color='gray', lw=1, alpha=0.6, label='AU $=m$ (diagonal)')
ax.axvline(DHAT, color='red', ls='-.', lw=1.5, alpha=0.7, label=REF_LABEL)
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
ax2.axhline(DHAT, color='red', ls='-.', lw=1.5, alpha=0.7, label=REF_LABEL)
ax2.set_xlabel('Bottleneck dimension $m$')
ax2.set_ylabel('TwoNN ID estimate')
ax2.set_title('Conv-VAE --- TwoNN ID vs. $m$')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/EN_conv_vae_extended_m_v44.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/EN_conv_vae_extended_m_v44.pdf', bbox_inches='tight')
print('Saved: results/figures/EN_conv_vae_extended_m_v44.{png,pdf}')
