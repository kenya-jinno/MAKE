"""
E4_mnist_dualaxis 図の v43 再生成（MAKE投稿論文_修正指摘と改善案.md §11.2 対応）
- MSE と AU/TwoNN を別パネルに分離（2x2 構成）して軸の取り違えを防止
- 参照線ラベルを "Step-1 reference d̂_ID = 10" に統一（真値と誤読させない）
- 旧図 E4_mnist_dualaxis.{png,pdf} は残し，新図は *_v43 として保存
学習は行わず results/tables/E4_extended.json から再描画する．
"""

import json
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
REF_LABEL = r'Step-1 reference $\hat{d}_{\mathrm{ID}} = 10$'

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- (a) Top-left: AE MSE ---
ax = axes[0, 0]
ax.plot(ms_ae, [r['mse'] for r in ae], 'b-o', lw=2, label='MSE (AE)')
for x, lab in [(43, 'PCA\n80%'), (86, 'PCA\n90%'), (152, 'PCA\n95%')]:
    ax.axvline(x, color='gray', ls=':', lw=1)
    ax.text(x * 1.05, 8.5e-3, lab, color='gray', fontsize=9)
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('MSE (log scale)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('(a) AE: reconstruction MSE vs. $m$')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# --- (b) Top-right: AE TwoNN ID ---
ax = axes[0, 1]
ax.plot(ms_ae, [r['id_twonn'] for r in ae], 'r-s', lw=2, label='TwoNN ID (AE latents)')
ax.axhline(DHAT, color='green', ls='--', lw=1.4, alpha=0.8, label=REF_LABEL)
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('TwoNN ID')
ax.set_xscale('log')
ax.set_ylim(0, 12)
ax.set_title('(b) AE: latent TwoNN ID vs. $m$')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

# --- (c) Bottom-left: VAE MSE ---
ax = axes[1, 0]
ax.plot(ms_vae, [r['mse'] for r in vae], 'b-o', lw=2, label=r'MSE (VAE, $\beta = 4$)')
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('MSE (log scale)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(r'(c) VAE ($\beta = 4$): reconstruction MSE vs. $m$')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# --- (d) Bottom-right: VAE AU & TwoNN ID ---
ax = axes[1, 1]
ax.plot(ms_vae, [r['au'] for r in vae], color='orange', marker='D', lw=2,
        label='AU (VAE)')
ax.plot(ms_vae, [r['id_twonn'] for r in vae], 'r--s', lw=1.5,
        label='TwoNN ID (VAE latents)')
ax.plot(ms_vae, ms_vae, 'k:', lw=1.2, alpha=0.6, label='AU $=m$')
ax.axhline(DHAT, color='green', ls='--', lw=1.4, alpha=0.8, label=REF_LABEL)
ax.set_xlabel('Bottleneck dimension $m$')
ax.set_ylabel('AU / TwoNN ID')
ax.set_xscale('log')
ax.set_ylim(0, 45)  # AU=m 対角線は上に抜ける；鈍化領域の可読性を優先
ax.set_title(r'(d) VAE ($\beta = 4$): AU and latent TwoNN ID vs. $m$')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/E4_mnist_dualaxis_v43.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/E4_mnist_dualaxis_v43.pdf', bbox_inches='tight')
print('Saved: results/figures/E4_mnist_dualaxis_v43.{png,pdf}')
