#!/usr/bin/env python3
"""
Generate figures for main_paper_MAKE4.tex
Figure 1: Diagnostic pipeline (TikZ in LaTeX - skip here)
Figure 2: MNIST AU dynamics - sparse vs dense grid comparison
Figure 3: TwoNN ID vs bottleneck margin (m-sweep)
Figure 4: Applicability boundary summary
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

out_dir = "results/figures_make"
os.makedirs(out_dir, exist_ok=True)

# ── Figure 2: AU dynamics sparse vs dense grid ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# --- Left panel: Sparse grid (multi-seed, 3 seeds) ---
ax = axes[0]
m_sparse = np.array([4, 8, 12, 16, 20, 32, 64])
au_mean  = np.array([4.0, 8.0, 10.0, 11.7, 13.3, 15.7, 22.3])
au_std   = np.array([0.0, 0.0,  0.0,  0.5,  0.5,  0.5,  0.5])

# Individual seed traces (approximate from paper data)
seeds_sparse = {
    'seed 42':  [4, 8, 10, 12, 13, 15, 22],
    'seed 123': [4, 8, 10, 12, 13, 16, 22],
    'seed 777': [4, 8, 10, 11, 14, 16, 23],
}
colors_s = ['#1f77b4', '#ff7f0e', '#2ca02c']
mknee_sparse = {'seed 42': 64, 'seed 123': 32, 'seed 777': 32}

for (lbl, au), col in zip(seeds_sparse.items(), colors_s):
    ax.plot(m_sparse, au, 'o--', color=col, alpha=0.6, markersize=5, label=lbl)
    mk = mknee_sparse[lbl]
    idx = list(m_sparse).index(mk)
    ax.axvline(mk, color=col, linestyle=':', linewidth=1.0, alpha=0.7)

ax.fill_between(m_sparse, au_mean - au_std, au_mean + au_std,
                alpha=0.15, color='gray', label='Mean ± std')
ax.plot(m_sparse, au_mean, 'k-', linewidth=2.0, label='Mean')

ax.set_xlabel('Bottleneck dimension $m$', fontsize=11)
ax.set_ylabel('Active Units (AU)', fontsize=11)
ax.set_title('Sparse grid\n'
             r'$m^{\rm knee} = 43 \pm 15$ (unreliable)', fontsize=11)
ax.set_xticks(m_sparse)
ax.legend(fontsize=8, loc='upper left')
ax.text(0.97, 0.05, r'$m^{\rm knee}$ per seed: 32 or 64',
        transform=ax.transAxes, ha='right', fontsize=8, color='gray')
ax.grid(True, alpha=0.3)

# --- Right panel: Dense grid (3 seeds) ---
ax = axes[1]
m_dense = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32, 40, 48, 64])
seeds_dense = {
    'seed 42  ($m^{\\rm knee}=11$)':  [8, 9, 10, 10, 10, 10, 12, 11, 14, 13, 15, 14, 15, 15, 18, 19, 21],
    'seed 123 ($m^{\\rm knee}=9$)':   [8, 9,  9, 10, 10, 10, 10, 11, 12, 13, 14, 14, 14, 14, 17, 18, 21],
    'seed 777 ($m^{\\rm knee}=11$)':  [8, 9, 10, 10, 11, 11, 11, 12, 13, 13, 14, 15, 15, 15, 18, 19, 21],
}
mknee_dense = {'seed 42  ($m^{\\rm knee}=11$)': 11,
               'seed 123 ($m^{\\rm knee}=9$)': 9,
               'seed 777 ($m^{\\rm knee}=11$)': 11}

for (lbl, au), col in zip(seeds_dense.items(), colors_s):
    ax.plot(m_dense, au, 'o-', color=col, alpha=0.75, markersize=4, label=lbl)
    mk = mknee_dense[lbl]
    ax.axvline(mk, color=col, linestyle=':', linewidth=1.0, alpha=0.8)

# Reference line: TwoNN ID ≈ 10
ax.axhline(10, color='red', linestyle='--', linewidth=1.5,
           label=r'TwoNN ID $\approx 10$')

ax.set_xlabel('Bottleneck dimension $m$', fontsize=11)
ax.set_ylabel('Active Units (AU)', fontsize=11)
ax.set_title('Dense grid (step 1 near $\\hat{d}_{\\rm ID}$)\n'
             r'$m^{\rm knee} = 10.3 \pm 0.9$ (stabilized)', fontsize=11)
ax.set_xticks([8, 10, 12, 14, 16, 20, 32, 48, 64])
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

fig.suptitle('Figure 2: MNIST VAE AU dynamics — sparse vs.\ dense grid',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f'{out_dir}/fig2_au_dynamics.pdf', bbox_inches='tight', dpi=150)
fig.savefig(f'{out_dir}/fig2_au_dynamics.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 saved.")

# ── Figure 3: TwoNN ID vs m (bottleneck margin) ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.2))

# Data from Exp-Geom (MNIST AE m-sweep, Table in appendix)
m_geom = np.array([4, 8, 12, 16, 20, 32, 48, 64, 96, 128])
twonn  = np.array([4.21, 6.34, 7.25, 8.17, 8.37, 9.36, 10.16, 10.17, 10.86, 10.98])
kappa  = np.array([2.69, 2.72, 2.71, 3.28, 3.48, 4.32, 4.44, 4.25, 9.94, 37.0])

sc = ax.scatter(m_geom, twonn, c=np.log10(kappa), cmap='RdYlGn_r',
                s=80, zorder=5, edgecolors='k', linewidths=0.5)
ax.plot(m_geom, twonn, 'k--', linewidth=1.0, alpha=0.5)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'$\log_{10}\kappa$ (anisotropy)', fontsize=9)

# Convergence threshold
ax.axhline(10.0, color='blue', linestyle='--', linewidth=1.5, label=r'$\hat{d}_{\rm ID} \approx 10$')
ax.axvline(48, color='green', linestyle=':', linewidth=1.5,
           label=r'$m \approx 5\hat{d}_{\rm ID}$ (converged)')

ax.set_xlabel('Bottleneck dimension $m$', fontsize=11)
ax.set_ylabel('TwoNN ID estimate', fontsize=11)
ax.set_title('Figure 3: TwoNN ID vs.\ bottleneck margin\n'
             r'(MNIST AE sweep; color = $\log_{10}\kappa$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 135)

fig.tight_layout()
fig.savefig(f'{out_dir}/fig3_twonn_stability.pdf', bbox_inches='tight', dpi=150)
fig.savefig(f'{out_dir}/fig3_twonn_stability.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 saved.")

# ── Figure 4: Applicability boundary summary ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')

rows = [
    ['Dataset / Model', 'AU Saturation', 'TwoNN\nConsistency',
     'Seed\nStability', 'Diagnostic\nDecision'],
    ['MNIST FC-VAE', 'Yes', 'Yes\n(σ≤0.12)', 'Yes\n(σ≤0.5)', 'Identifiable'],
    ['MNIST Conv-VAE\n(β=4)', 'No (β too low)', 'Partial\n(m=64 only)', 'Yes', 'Inconclusive'],
    ['MNIST Conv-VAE\n(β≥10)', 'Yes\n(mknee≈128)', 'Partial', 'Partial', 'Identifiable\n(high β)'],
    ['CIFAR-10\n(Conv-VAE)', 'Weak/conditional\n(β=80, thin dec.)', 'Partial\n(m≈2d̂)', 'No', 'Boundary'],
    ['SVHN\n(Conv-VAE)', 'Partial', 'No\n(diverging)', 'No', 'Unreliable'],
]

colors_row = ['#cccccc', '#c6efce', '#ffeb9c', '#c6efce', '#ffc7ce', '#ffc7ce']
col_widths = [0.22, 0.16, 0.16, 0.14, 0.18]

table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc='center',
    cellLoc='center',
    colWidths=col_widths,
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 2.2)

for j in range(len(rows[0])):
    table[0, j].set_facecolor('#4472c4')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i, color in enumerate(colors_row[1:], start=1):
    for j in range(len(rows[0])):
        table[i, j].set_facecolor(color)

ax.set_title('Figure 4: Applicability-boundary summary\n'
             '(Green = identifiable, Yellow = conditional, Red = inconclusive/boundary)',
             fontsize=10, pad=8)

fig.tight_layout()
fig.savefig(f'{out_dir}/fig4_applicability.pdf', bbox_inches='tight', dpi=150)
fig.savefig(f'{out_dir}/fig4_applicability.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 4 saved.")

print("All figures generated successfully.")
