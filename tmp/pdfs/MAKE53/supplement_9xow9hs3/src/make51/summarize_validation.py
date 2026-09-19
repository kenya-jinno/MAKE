"""Tables and figures for completed MAKE51 model validation records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/make51';FIG=ROOT/'results/figures/MAKE51';V=OUT/'model_validation'
def read(p):return json.loads(p.read_text())
spec=read(V/'specification.json')
records={m:read(V/f'ordinary_m{m}_s42.json') for m in spec['grid']}
selection=read(V/'selection_test.json');hc=read(V/'native_hc_s42.json');DOLLAR='$'
assert len(records)==12 and len(set(r['split_sha256'] for r in records.values()))==1
assert all(r['seed']==42 for r in records.values())
assert all(np.isfinite(r['mc_val']['elbo']) for r in records.values())

def table(name,caption,label,cols,header,rows):
    text=(f'\\begin{{table}}[H]\n\\caption{{{caption}}}\\label{{{label}}}\n\\small\n'
          '\\setlength{\\tabcolsep}{3pt}\n'
          f'\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}@{{}}}}\n\\toprule\n'
          +header+' \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular*}\n\\end{table}\n')
    (OUT/f'{name}.tex').write_text(text)
lines=[]
for criterion,m in selection['choices'].items():
    r=records[m]
    lines.append(f"{criterion} & {m} & {1000*r['val']['mse']:.2f} & {1000*selection['test'][str(m)]['mse']:.2f} & {DOLLAR}{r['mc_val']['elbo']:.3f} \\pm {r['mc_val']['mc_se']:.3f}{DOLLAR} \\\\")
table('mcselection',r'New MNIST seed-42 capacity selection at $\beta=1$, Gaussian variance $1/2$. All criteria choose among the same twelve proxy-selected checkpoints. MSE is multiplied by $10^3$; ELBO is mean nats/example and its $\pm$ value is Monte Carlo SE ($K=32$), not a training-seed SD. Test is evaluated after validation choices are frozen.','tab:mcselection','lcccc',r'Criterion & Width & Val. MSE & Test MSE & MC ELBO $\pm$ MC SE',lines)
lines=[]
for m,r in records.items():
    lines.append(f"{m} & {r['best_epoch']} & {r['epochs_run']} & {1000*r['val']['mse']:.3f} & {r['val']['proxy']:.3f} & {DOLLAR}{r['mc_val']['elbo']:.3f} \\pm {r['mc_val']['mc_se']:.3f}{DOLLAR} \\\\")
table('mcgrid',r'Complete newly trained MNIST grid, seed 42, $\beta=1$, Gaussian likelihood. Epoch selected by the deterministic proxy; stop is the actual final training epoch. Proxy omits the Gaussian constant and posterior expectation; MC ELBO includes both. MSE $\times10^3$.','tab:mcgrid','rrrrrc',r'Width & Epoch & Stop & Val. MSE & Proxy & MC ELBO $\pm$ SE',lines)

lines=[]
for name,val,test,width in [
    ('HC, continuous/mean',hc['native']['val_mean'],hc['native']['test_mean'],hc['raw_count']),
    ('HC, continuous/sampled',hc['native']['val_sampled'],hc['native']['test_sampled'],hc['raw_count']),
    ('HC, binary/mean',hc['native']['val_binary_mean'],hc['native']['test_binary_mean'],hc['raw_count']),
    ('Transferred ordinary VAE',hc['transfer_val']['mse'],hc['transfer_test']['mse'],hc['grid_width'])]:
    lines.append(f"{name} & {width} & {val*1000:.3f} & {test*1000:.3f} & {'yes' if val<=hc['tau'] else 'no'} \\\\")
table('native_hc',r'New MNIST HC diagnostic and dimension transfer, seed 42. MSE $\times10^3$; target is $'+f'{1000*hc["tau"]:.3f}'+r'$ in the same units. HC width is the count above the gate threshold, although continuous inference retains nonbinary gate values. Sampled evaluation uses 32 latent draws and deterministic continuous gates. The ordinary VAE uses the nearest grid width.','tab:native_hc','lrrrr',r'Output / reconstruction & Width & Val. MSE & Test MSE & $Q$',lines)

choices=selection['choices'];m_mc=choices['MC ELBO'];m_proxy=choices['ELBO proxy'];m_mse=choices['MSE']
text=f"Validation MSE selects width {m_mse}, the proxy selects {m_proxy}, and Monte Carlo ELBO selects {m_mc}."
if m_mc!=m_proxy:
    differences=np.asarray(records[m_mc]['mc_val']['draw_means'])-np.asarray(records[m_proxy]['mc_val']['draw_means'])
    text+=f" The estimated ELBO difference between the MC-selected and proxy-selected widths is {differences.mean():.3f} nats/example (paired Monte Carlo SE {differences.std(ddof=1)/np.sqrt(len(differences)):.3f})."
else:
    text+=" Agreement in this run does not make the proxy an ELBO estimator."
last=hc['curves'][-1]
hctext=(f"The final training moving-average constraint is {last['constraint_ma_sse']:+.3f} "
        f"summed-error units, with {hc['raw_count']} counted gates and multiplier {last['lambda_value']:.3f}. "
        f"The transferred grid width is {hc['grid_width']}. ")
hctext+=("The endpoint training constraint is satisfied; this single outcome does not validate other datasets or seeds."
         if last['constraint_ma_sse']<=0 else
         "The endpoint training constraint remains violated. This diagnostic therefore does not promote the HC implementation to a validated pruning baseline.")
(OUT/'validation_text.json').write_text(json.dumps({'MC_INTERPRETATION':text,'HC_INTERPRETATION':hctext},indent=2))

plt.rcParams.update({'font.size':12,'legend.fontsize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
for ax,m in zip(axes,[6,64]):
    r=records[m];x=[v['epoch'] for v in r['curves']]
    ax.plot(x,[v['train_mse'] for v in r['curves']],label='Train subset')
    ax.plot(x,[v['mse'] for v in r['curves']],label='Validation',ls='--')
    ep=r['best_epoch'];ax.scatter(ep,r['curves'][ep-1]['mse'],c='black',marker='x',s=30)
    ax.set(title=f'MNIST, width {m}',xlabel='Epoch',ylabel='MSE',yscale='log');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
fig.savefig(FIG/'new_mnist_curves.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
c=hc['curves'];x=[r['epoch'] for r in c]
axes[0,0].plot(x,[r['constraint_ma_sse'] for r in c]);axes[0,0].axhline(0,color='.5',ls='--')
axes[0,0].set(ylabel='Training constraint (SSE)',yscale='symlog',title='Stochastic training constraint')
axes[0,1].plot(x,[r['val_mean_continuous_mse'] for r in c],label='Continuous gate')
axes[0,1].plot(x,[r['val_mean_binary_mse'] for r in c],label='Binary gate',ls='--')
axes[0,1].axhline(hc['tau'],color='.5',ls=':',label='Target')
axes[0,1].set(ylabel='Validation MSE',yscale='log',title='Posterior-mean reconstruction')
axes[0,1].legend(fontsize=11)
axes[1,0].plot(x,[r['gates'] for r in c]);axes[1,0].set(ylabel='Count',title='Gates above 0.5')
axes[1,1].plot(x,[r['lambda_value'] for r in c]);axes[1,1].set(ylabel='Multiplier',yscale='log',title='Constraint multiplier')
for ax in axes.flat:ax.set(xlabel='Epoch');ax.grid(alpha=.2)
fig.savefig(FIG/'native_hc.pdf',bbox_inches='tight');plt.close(fig)
print(json.dumps({'choices':choices,'MC_interpretation':text,'HC_interpretation':hctext},indent=2))
