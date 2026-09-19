"""Reaggregate immutable MAKE49 logs for MAKE51; no model training.

Run from the repository root with Python, NumPy, matplotlib and scikit-learn.
All analysis choices added in MAKE51 are retrospective and labelled as such.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import sklearn
import platform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/make51'
FIG = ROOT / 'results/figures/MAKE51'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def read(name):
    return json.loads((ROOT / name).read_text())
CFG = read('config/experiment_config_MAKE49.json')
DS = ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']
DN = dict(zip(DS, ['MNIST', 'Fashion-MNIST', 'dSprites', 'CIFAR-10']))
SEEDS = CFG['data_splits']['training_seeds']
BETAS = CFG['beta']['ladder']
CACHE = {d: read(f'results/make49/stage4_cache/{d}.json') for d in DS}
RAW = read('results/make49/stage4_methods.json')['results']
PRUNE = read('results/make49/stage7_b7b8.json')['results']
NAMES = {'B1_full_mse': 'Grid (MSE)', 'B2_full_elbo': 'Grid (ELBO proxy)',
 'B3_full_downstream': 'Grid (probe)', 'B4_coarse': 'Coarse search',
 'B5_fixed32': 'Fixed 32', 'B6a_fondue': 'FONDUE',
 'B6b_fondue_var': 'FONDUE-VAR (mixed kept)',
 'B6b_fondue_var_keepmixed_false': 'FONDUE-VAR (active only)',
 'proposed': 'Legacy ID window', 'ascending_scan_Q': 'Ascending + Q',
 'B6a_plus_Q': 'FONDUE + Q', 'B7_geco_l0': 'GECO adaptation',
 'B8_ard_vae': 'ARD adaptation', 'proposed_minus_AU':'ID-guided, no AU',
 'B4_plus_Q':'Coarse + Q', 'B6b_plus_Q':'FONDUE-VAR + Q'}
AUDIT_METHODS = ['B1_full_mse','B6a_fondue','B6a_plus_Q','B7_geco_l0',
        'B8_ard_vae','ascending_scan_Q','proposed']
MAIN = [m for m in AUDIT_METHODS if m not in ['B7_geco_l0','B8_ard_vae']]
def cached(ds, m, seed, beta=1):
    return CACHE[ds][f'vae|{ds}|m{m}|s{seed}|e300|es1|b{beta:g}']
def mean(v): return float(np.mean(v)) if len(v) else None
def sd(v): return float(np.std(v, ddof=1)) if len(v)>1 else 0.
def pm(v, digits=1, scale=1):
    if not len(v): return '---'
    return f'${mean(v)*scale:.{digits}f} \\pm {sd(v)*scale:.{digits}f}$'
def tex_table(name, caption, label, cols, header, rows, size='small'):
    txt = (f'\\begin{{table}}[H]\n\\caption{{{caption}}}\\label{{{label}}}\n'
           f'\\{size}\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n'
           f'\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}@{{}}}}\n'
           '\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(rows)+
           '\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n')
    (OUT / (name+'.tex')).write_text(txt)
    return txt

# Reconstruct the published deterministic split rule without loading images.
split_arrays={};split_rows=[]
for ds in DS:
    total=737280 if ds=='dSprites' else (50000 if ds=='CIFAR10' else 60000)
    perm=np.random.RandomState(20260912).permutation(total)
    pool=perm[:20000];pos=np.random.RandomState(20260913).permutation(20000)
    tr,va=pool[pos[:18000]],pool[pos[18000:]]
    te=perm[20000:23000] if ds=='dSprites' else np.random.RandomState(20260914).permutation(10000)[:3000]
    if ds=='dSprites':tr,va,te=sorted(tr),sorted(va),sorted(te)
    tr,va,te=[np.asarray(x,dtype=np.int64) for x in [tr,va,te]]
    assert len(np.intersect1d(tr,va))==0
    if ds=='dSprites':assert not len(np.intersect1d(np.concatenate([tr,va]),te))
    hh=hashlib.sha256()
    for name,arr in zip(['train','val','test'],[tr,va,te]):
        split_arrays[f'{ds}_{name}']=arr;hh.update(np.ascontiguousarray(arr).tobytes())
    split_rows.append(f"{DN[ds]} & \\texttt{{{hh.hexdigest()[:16]}}} \\\\")
np.savez_compressed(OUT/'reconstructed_split_indices.npz',**split_arrays)
tex_table('splits','Split identities reconstructed from the saved loader and fixed seeds. Tokens are the first 16 hexadecimal characters of SHA-256 over concatenated int64 train/validation/test index arrays, matching the loader convention. Official training/test pools have separate index spaces.','tab:splits','lc',r'Dataset & Reconstructed split token',split_rows)

ROWS=[]
for r in RAW+PRUNE:
    ds, method, s = r['dataset'], r['method'], r['seed']
    beta = r.get('beta',r.get('beta_context'))
    anchor = max(CFG['candidate_grids'][ds]); a=cached(ds,anchor,s,beta)
    qrec=next(x['quality'] for x in RAW if x['dataset']==ds and x['seed']==s
              and x['beta']==beta and x['method']=='ascending_scan_Q')
    assert not qrec['sd_available']
    assert abs(qrec['T_D']-1.1*a['val']['mse'])<1e-10
    f=r.get('final_model'); m=r['selected_m']
    actual=f.get('m_on_grid',m) if f else None
    if f:
        # Some early off-grid candidate caches were not retained. Their final
        # validation/test metrics and timing remain in the method-level record.
        cr=CACHE[ds].get(f'vae|{ds}|m{actual}|s{s}|e300|es1|b{beta:g}')
        if cr is not None:
            assert abs(cr['val']['mse']-f['val']['mse'])<1e-10
    trace=r.get('trace',[])
    anchor_charged = any(t['kind']=='vae' and re.search(fr'\|m{anchor}\|e300\|es1(?:\||$)',t['key']) for t in trace)
    if method in ('B7_geco_l0','B8_ard_vae'):
        anchor_charged=(actual==anchor)
    add_anchor=0. if anchor_charged else a['elapsed_seconds']
    ROWS.append(dict(dataset=ds,method=method,seed=s,beta=beta,
        selected_raw=m,final_m=actual,anchor_m=anchor,D_anchor=qrec['D_anchor'],T_D=qrec['T_D'],
        val_mse=f['val']['mse'] if f else None,test_mse=f['test']['mse'] if f else None,
        quality_met=bool(f and f['val']['mse']<=qrec['T_D']),
        original_seconds=r['total_seconds'],added_anchor_seconds=add_anchor,
        accounted_seconds=r['total_seconds']+add_anchor,
        state=r['termination_state'],id_fallback=bool(r.get('fallback')),
        budget_exhausted=bool(r.get('budget_exhausted',False)),
        compact=bool(actual and actual<anchor),
        calibration_seconds=(r.get('fondue_epochs_info') or {}).get('search_seconds',0.),
        final_constraint_ma=r.get('final_constraint_ma')))
assert len(ROWS)==1600
for ds in DS:
    for method in NAMES:
        for beta in BETAS:
            assert len([r for r in ROWS if r['dataset']==ds and r['method']==method and r['beta']==beta])==5
with (OUT/'method_results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
def select(ds,meth,beta=1):
    return [r for r in ROWS if r['dataset']==ds and r['method']==meth and r['beta']==beta]

for block, datasets in enumerate([DS[:2],DS[2:]],1):
    lines=[]
    for ds in datasets:
        lines += [f'\\multicolumn{{5}}{{l}}{{\\emph{{{DN[ds]}}}}} \\\\']
        for meth in MAIN:
            rs=select(ds,meth)
            lines += [f"{NAMES[meth]} & {pm([r['final_m'] for r in rs])} & "
                f"{pm([r['val_mse'] for r in rs],2,1000)} & {pm([r['test_mse'] for r in rs],2,1000)} & "
                f"{sum(r['quality_met'] for r in rs)}/5 \\\\"]
        lines += ['\\midrule']
    tex_table(f'main{block}',
        'Dimension selection at $\\beta=1$ ('+('Gaussian likelihoods' if block==1 else 'Bernoulli for dSprites; Gaussian for CIFAR-10')+
        '). Values are mean $\\pm$ sample SD over five candidate-training seeds. '
        '$m_{\\rm final}$ is the width of the evaluated ordinary VAE. '
        'MSE columns are multiplied by $10^3$; $Q$ counts validation target attainment. '
        'Historical timing ledgers are separated in Appendix~\\ref{app:timing}; replay costs use a single canonical snapshot.',f'tab:main{block}',
        'lcccc',r'Method & $m_{\rm final}$ & Val. MSE & Test MSE & $Q$',lines[:-1], 'footnotesize')

lines=[]
for ds in DS:
    for b in BETAS:
        rs=select(ds,'ascending_scan_Q',b)
        lines.append(f"{DN[ds]} & {b:g} & {pm([r['D_anchor'] for r in rs],3,1000)} & {pm([r['T_D'] for r in rs],3,1000)} \\\\")
tex_table('anchors','Quality anchors and targets, in units of $10^{-3}$ MSE, mean $\\pm$ SD over five seeds. Each decision uses its own seed-specific target, not the displayed mean. The relative margin is 10\\%; the absolute floor is inactive. Gaussian likelihood except Bernoulli dSprites.','tab:anchors','lccc',r'Dataset & $\beta$ & Anchor MSE & Target $T_{D,s}$',lines)

SPAN_METHODS=['B1_full_mse','B4_coarse','B5_fixed32','B6a_fondue','B6b_fondue_var','ascending_scan_Q','B6a_plus_Q','proposed']
lines=[]
for meth in SPAN_METHODS:
    vals=[]
    for ds in DS:
        ms=[mean([r['selected_raw'] for r in select(ds,meth,b) if r['selected_raw'] is not None]) for b in BETAS]
        vals.append(f'{max(ms)-min(ms):.1f}' if meth!='B8_ard_vae' else '---')
    lines.append(NAMES[meth]+' & '+' & '.join(vals)+r' \\')
tex_table('span','Range of seed-mean returned dimensions across $\\beta\\in\\{0.25,0.5,1,2,4\\}$. Pruning adaptations are restricted to the implementation audit. Fixed 32 is a design constant; fallback spans are not ID robustness. Gaussian likelihood except Bernoulli dSprites.','tab:span','lcccc',r'Method & MNIST & Fashion & dSprites & CIFAR-10',lines)

lines=[]
for ds in DS:
    r=next(r for r in RAW if r['dataset']==ds and r['method']=='proposed' and r['beta']==1)
    i=r['id_info']; cc=CACHE[ds]
    vals=[]
    for mr in i['d_hat_by_m_ref']:
        v=[v['twonn'] for k,v in cc.items() if k.startswith(f'ae|{ds}|m{mr}|')]
        vals.append(f"{mr}: {mean(v):.2f} $\\pm$ {sd(v):.2f}")
    lines.append(f"{DN[ds]} & "+'; '.join(vals)+f" & {i['relative_range']:.3f} & {i['estimator_rel_diff']:.3f} \\\\")
txt=(r'\begin{table}[H]\caption{ID-reference AE widths and TwoNN estimates (mean $\pm$ SD over three reference seeds); stability $R$ and estimator disagreement $A$. Limits are $R\le0.15$ and $A\le0.25$. The same reference bank is reused across candidate seeds and $\beta$.}\label{tab:id}'
     '\n\\footnotesize\\begin{tabularx}{\\textwidth}{lXcc}\\toprule\nDataset & Width: estimate & $R$ & $A$ \\\\\n\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\\end{tabularx}\\end{table}')
(OUT/'id.tex').write_text(txt)

lines=[]
for ds in DS:
    for meth in ['B1_full_mse','B2_full_elbo','B3_full_downstream']:
        rs=select(ds,meth); acc=[cached(ds,r['final_m'],r['seed'])['probe']['test_acc'] for r in rs]
        lines.append(f"{DN[ds]} & {NAMES[meth]} & {pm([r['final_m'] for r in rs])} & {pm([r['test_mse'] for r in rs],2,1000)} & {pm(acc,1,100)} & {pm([r['accounted_seconds'] for r in rs],0)} \\\\")
tex_table('selection','Selection-criterion comparison at $\\beta=1$, five seeds; test MSE $\\times10^3$ and probe test accuracy (\\%). Probes use class labels (shape for dSprites); only Grid (probe) uses them for capacity selection. Time is the historical ledger including anchor completion, not the canonical replay price. Recorded probe time is charged to that selector. Other displayed probe accuracies are auxiliary evaluations with uncharged costs. Likelihoods follow Table~\\ref{tab:data}.','tab:selection','llcccc',r'Dataset & Criterion & $m$ & Test MSE & Accuracy & Time (s)',lines,'footnotesize')

lines=[]
for ds in DS:
    for meth in ['B6b_fondue_var','B6b_fondue_var_keepmixed_false']:
        rs=select(ds,meth); valid=[r for r in rs if r['final_m'] is not None]
        lines.append(f"{DN[ds]} & {'yes' if meth=='B6b_fondue_var' else 'no'} & {pm([r['selected_raw'] for r in rs if r['selected_raw'] is not None])} & {len(valid)}/5 & {sum(r['quality_met'] for r in rs)}/5 & {pm([r['test_mse'] for r in valid],2,1000)} & {pm([r['accounted_seconds'] for r in rs],0)} \\\\")
tex_table('var','FONDUE-VAR at $\\beta=1$: both keep-mixed settings, valid-model and quality counts, test MSE $\\times10^3$, and recorded training time. Raw zero outputs remain in the dimension summary; nonexistent models have no MSE. Five seeds; likelihoods follow Table~\\ref{tab:data}.','tab:var','lcccccc',r'Dataset & Mixed kept & Raw $m$ & Valid & $Q$ & Test MSE & Time (s)',lines,'footnotesize')

lines=[]
for meth in AUDIT_METHODS:
    rr=select('MNIST',meth)
    vals={k:[] for k in ['anchor','ae','search','final','total']}
    for r in rr:
        raw=next(v for v in RAW+PRUNE if v['dataset']=='MNIST' and v['method']==meth and v['seed']==r['seed'] and v.get('beta',v.get('beta_context'))==1)
        av=cached('MNIST',64,r['seed'])['elapsed_seconds']; ae=0.; final=0.; search=0.
        if meth in ['B7_geco_l0','B8_ard_vae']:
            search=raw['elapsed_seconds']; final=raw.get('final_train_seconds',0.) if r['final_m']!=64 else 0.
        else:
            for t in raw['trace']:
                if t['kind']=='ae':ae+=t['seconds']
                elif re.search(r'\|m64\|e300\|es1(?:\||$)',t['key']):av=t['seconds']
                elif re.search(fr"\|m{r['final_m']}\|e300\|es1(?:\||$)",t['key']):final+=t['seconds']
                else:search+=t['seconds']
        assert abs(av+ae+search+final-r['accounted_seconds'])<1e-6
        for k,v in zip(vals,[av,ae,search,final,r['accounted_seconds']]):vals[k].append(v)
    lines.append(NAMES[meth]+' & '+' & '.join(f'{mean(v):.1f}' for v in vals.values())+r' \\')
tex_table('cost','Historical MNIST training-time decomposition at $\\beta=1$ (mean seconds, five seeds). The final model is charged once and removed from the other-candidate column. ID computation and post-training evaluation were not separately timed. FONDUE epoch calibration adds 4.14 s once per dataset--$\\beta$ setting and is not included below.','tab:cost','lrrrrr',r'Method & Anchor & ID AEs & Other training & Final & Total',lines,'footnotesize')

lines=[]
for ds in DS:
    for meth in AUDIT_METHODS+['B6b_fondue_var','B6b_fondue_var_keepmixed_false']:
        rs=select(ds,meth)
        rawstates={s:sum(r['state']==s for r in rs) for s in sorted(set(r['state'] for r in rs))}
        st=', '.join({'SUCCESS':'S','ID_UNRELIABLE':'I','BUDGET_EXHAUSTED':'B','DEGENERATE_SELECTION':'D','NO_COMPACT_ALTERNATIVE_FOUND':'N'}[s]+str(n) for s,n in rawstates.items())
        lines.append(f"{DN[ds]} & {NAMES[meth]} & {st} & {sum(r['id_fallback'] for r in rs)} & {sum(r['budget_exhausted'] for r in rs)} & {sum(r['quality_met'] for r in rs)} & {sum(r['compact'] for r in rs)} \\\\")
# Two pages, with explicit independent outcome columns.
for bi in range(2):
    tex_table(f'outcomes{bi}', 'Outcome audit at $\\beta=1$; all counts have denominator five. S: recorded success; I: ID unreliable; B: budget exceeded; D: degenerate; N: no compact alternative. FB is the fallback flag; budget, validation quality $Q$, and compression of the final VAE are independently recomputed or read from flags.','tab:outcomes'+str(bi),'llccccc',r'Dataset & Method & State & FB & Budget & $Q$ & Compact',lines[bi*18:(bi+1)*18],'footnotesize')

lines=[]
for ds in DS:
    for b in BETAS:
        fe=next(r['fondue_epochs_info'] for r in RAW if r['dataset']==ds and r['method']=='B6a_fondue' and r['beta']==b)
        flag='N/A (source)' if ds=='dSprites' else ('yes' if fe['stable'] else 'no')
        lines.append(f"{DN[ds]} & {b:g} & {fe['epochs']} & {fe['search_seconds']:.2f} & {flag} \\\\")
tex_table('epochs','FONDUE short-training epochs and calibration time (once per dataset--context, separate from selector time). dSprites uses the source choice without calibration. A stable flag for the remaining datasets means consecutive calibration predictions agreed.','tab:epochs','lcccc',r'Dataset & $\beta$ & Epochs & Calibration (s) & Stable',lines)

lines=[]
for ds in DS:
    for meth in ['B7_geco_l0','B8_ard_vae']:
        rs=select(ds,meth)
        c=sum(r['final_constraint_ma']<=0 for r in rs) if meth=='B7_geco_l0' else None
        lines.append(f"{DN[ds]} & {'GECO' if meth=='B7_geco_l0' else 'ARD'} & {pm([r['selected_raw'] for r in rs])} & {pm([r['final_m'] for r in rs])} & {str(c)+'/5' if c is not None else '---'} & {sum(r['quality_met'] for r in rs)}/5 \\\\")
tex_table('pruning','Raw retained counts versus transferred ordinary-VAE widths at the primary context. Constraint denotes the GECO training moving average at the final update; $Q$ denotes final ordinary-VAE validation quality. They evaluate different models and quantities. ARD has no GECO constraint.','tab:pruning','llcccc',r'Dataset & Adaptation & Raw count & Final width & Constraint & $Q$',lines,'footnotesize')
for ds in DS:
    for s in SEEDS:
        rr=[r for r in PRUNE if r['dataset']==ds and r['method']=='B8_ard_vae' and r['seed']==s]
        assert len(rr)==5
        assert all(r['relevance_score']==rr[0]['relevance_score'] and r['selected_m']==rr[0]['selected_m'] for r in rr)

lines=[]
for delta in [.001,.01,.1]:
    vals=[100*np.mean([sum(x>delta for x in cached('CIFAR10',m,s,b)['readouts']['B']['mu_var_per_dim'])==m for m in CFG['candidate_grids']['CIFAR10'] for s in SEEDS]) for b in BETAS]
    lines.append(f'{delta:g} & '+' & '.join(f'{v:.0f}' for v in vals)+r' \\')
tex_table('delta_rates','CIFAR-10 (Gaussian): percentage of the 50 width/seed candidates with AU equal to capacity. Thresholds are re-read on the same checkpoints within each column.','tab:delta_rates','lccccc',r'$\delta$ & $\beta=.25$ & $.5$ & $1$ & $2$ & $4$',lines)
lines=[]
for ds in DS:
    vals=[]
    floor=next(r['quality']['terms']['abs_floor'] for r in RAW if r['dataset']==ds and r['method']=='ascending_scan_Q')
    for rel in [.01,.05,.1,.25,.5]:
        ms=[]
        for s in SEEDS:
            grid=CFG['candidate_grids'][ds];D=cached(ds,grid[-1],s)['val']['mse'];T=D+max(rel*D,floor)
            ms.append(min(m for m in grid if cached(ds,m,s)['val']['mse']<=T))
        vals.append(pm(ms))
    lines.append(DN[ds]+' & '+' & '.join(vals)+r' \\')
tex_table('margin','Smallest quality-satisfying cached width at $\\beta=1$ as relative margin changes; five seed means $\\pm$ SD. The absolute floor 0.001 times validation variance remains fixed. Gaussian likelihood except Bernoulli dSprites.','tab:margin','lccccc',r'Dataset & 1\% & 5\% & 10\% & 25\% & 50\%',lines,'footnotesize')

# Paired AU analysis: explicitly select beta=1 to avoid cache-order dependence.
IID=read('results/make49/input_space_id.json')
FEATS=['D','KL','E','m','m_d','slope']; AU=['AU','AU_m','AU_d']
ORDER=['MNIST','dSprites','FashionMNIST','CIFAR10']
TABLES={}
for ds in ORDER:
    grid=CFG['candidate_grids'][ds]; dh=IID[ds]['twonn']; rows=[]
    for s in sorted(SEEDS):
        threshold=1.1*cached(ds,grid[-1],s)['val']['mse']
        for mi,m in enumerate(grid[1:],1):
            r=cached(ds,m,s); au=r['readouts']['B']['au']; w=r['curve_val_mse'][-20:]
            rows.append(dict(ds=ds,seed=s,m=m,D=r['val']['mse'],KL=r['val']['kl'],E=r['val']['elbo'],
              m_d=m/dh,slope=float(np.polyfit(np.arange(len(w)),w,1)[0]),AU=au,AU_m=au/m,AU_d=au/dh,
              y=int(cached(ds,grid[mi-1],s)['val']['mse']<=threshold)))
    TABLES[ds]=rows
def fit(train,feats,C=10.):
    X=np.array([[r[f] for f in feats] for r in train]);y=np.array([r['y'] for r in train])
    scaler=StandardScaler().fit(X); clf=LogisticRegression(C=C,max_iter=5000).fit(scaler.transform(X),y)
    # AUROC depends on ranking. Decision scores preserve ordering when sigmoid
    # probabilities round to exactly 0 or 1 under distribution shift.
    return lambda rows: clf.decision_function(scaler.transform([[r[f] for f in feats] for r in rows]))
dev=TABLES['MNIST']; tr=[r for r in dev if r['seed'] in sorted(SEEDS)[:3]]; va=[r for r in dev if r['seed'] in sorted(SEEDS)[3:]]
C=max([.01,.1,1.,10.],key=lambda c:roc_auc_score([r['y'] for r in va],fit(tr,FEATS,c)(va)))
def aucs(feats):
    pred=fit(dev,feats,C);out=[]
    for ds in ORDER[1:]:
        vals=[]
        for s in sorted(SEEDS):
            rows=[r for r in TABLES[ds] if r['seed']==s]
            vals.append(roc_auc_score([r['y'] for r in rows],pred(rows)))
        out.append(vals)
    return np.array(out)
base=aucs(FEATS); aug=aucs(FEATS+AU)
def noise_auc(seed):
    rng=np.random.default_rng(seed)
    for ds in ORDER:
        for r in TABLES[ds]:
            for i in range(3):r[f'n{i}']=float(rng.standard_normal())
    return aucs(FEATS+['n0','n1','n2'])
noise=noise_auc(7)
def interval(mat,hierarchical=True):
    rng=np.random.default_rng(20260918); bs=[]
    for _ in range(2000):
        di=rng.integers(0,3,3) if hierarchical else np.arange(3)
        bs.append(np.mean([mat[d,rng.integers(0,5,5)].mean() for d in di]))
    return [float(x) for x in np.percentile(bs,[2.5,97.5])]
noise_reps=np.array([noise_auc(i) for i in range(100)])
AUDIT={'C':C,'n_rows':{d:len(t) for d,t in TABLES.items()},
       'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
       'auc_score':'decision_function; no sigmoid saturation ties',
       'mean_delta_au':float((aug-base).mean()),'mean_delta_noise7':float((noise-base).mean()),
       'delta_au_hierarchical_ci':interval(aug-base),
       'paired_difference':float((aug-noise).mean()),'paired_hierarchical_ci':interval(aug-noise),
       'paired_stratified_ci':interval(aug-noise,False),
       'noise100_difference':float((aug-noise_reps.mean(0)).mean()),
       'noise100_hierarchical_ci':interval(aug-noise_reps.mean(0)),
       'base':base.tolist(),'au':aug.tolist(),'noise7':noise.tolist(),
       'noise100_mean':noise_reps.mean(0).tolist()}
noise_auc(7)
predictions=[];SAT=[]
for label,feats in [('base',FEATS),('au',FEATS+AU),('random7',FEATS+['n0','n1','n2'])]:
    predictor=fit(dev,feats,C)
    for ds in ORDER[1:]:
        scores=predictor(TABLES[ds]);prob=expit(scores)
        SAT.append({'dataset':ds,'predictor':label,'n':len(scores),'p_exactly_one':int(sum(prob==1)),
                    'p_exactly_zero':int(sum(prob==0)),'score_min':float(min(scores)),'score_max':float(max(scores))})
        for r,z,p in zip(TABLES[ds],scores,prob):
            predictions.append({'dataset':ds,'seed':r['seed'],'m':r['m'],'Y':r['y'],
                'predictor':label,'decision_score':z,'probability':p})
AUDIT['probability_saturation']=SAT
with (OUT/'au_predictions.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
(OUT/'au_paired.json').write_text(json.dumps(AUDIT,indent=2))
lines=[]
for i,ds in enumerate(ORDER[1:]):
    lines.append(f"{DN[ds]} & {len(TABLES[ds])} & {base[i].mean():.3f} & {aug[i].mean():.3f} & {noise[i].mean():.3f} & {(aug[i]-noise[i]).mean():+.3f} \\\\")
tex_table('au','Retrospective paired AU comparison at $\\beta=1$, using logistic decision scores to avoid saturated-probability ties. Values are means of five seed-specific AUROCs, each evaluated over all nonminimal capacities in that seed. Random features are three standard-normal columns with generator seed 7. All methods use the same rows and labels.','tab:au','lccccc',r'Evaluation condition & Rows & Base & Base + AU & Base + random & AU $-$ random',lines,'footnotesize')
lines=[]
for ds in ORDER[1:]:
    ss=[r for r in SAT if r['dataset']==ds]
    lines.append(DN[ds]+' & '+' & '.join(f"{r['p_exactly_one']}/{r['n']}" for r in ss)+r' \\')
tex_table('saturation','Numerical probability saturation in the current predictor reconstruction. Entries count predictions rounded to exactly one, before AUROC calculation. Decision scores preserve ordering among these rows.','tab:saturation','lccc',r'Evaluation condition & Base & Base + AU & Base + random',lines)

TRAJ=read('results/make49/stage3_e7a_e2.json')['runs']
ANAL=read('results/make49/stage3_e7a_analysis.json')
lines=[]
for ds,cap in [('MNIST',12),('MNIST',32),('MNIST',64),('dSprites',6),('dSprites',16),('dSprites',32)]:
    rs=[r for r in TRAJ if r['dataset']==ds and r['m']==cap]
    a5=[r['readouts']['5']['A']['active_plus_mixed'] for r in rs];b5=[r['readouts']['5']['B']['au'] for r in rs]
    af=[r['readouts']['300']['A']['active_plus_mixed'] for r in rs];bf=[r['readouts']['300']['B']['au'] for r in rs]
    da=mean([abs(r['readouts']['300']['A']['active_plus_mixed']-r['readouts']['50']['A']['active_plus_mixed']) for r in rs])
    db=mean([abs(r['readouts']['300']['B']['au']-r['readouts']['50']['B']['au']) for r in rs])
    lines.append(f"{DN[ds]} & {cap} & {pm(a5)} & {pm(b5)} & {pm(af)} & {pm(bf)} & {da:.1f}/{db:.1f} \\\\")
tex_table('readouts','Variable-type count $V$ (active plus mixed) and AU count $A$, five seeds, $\\beta=1$, Gaussian MNIST and Bernoulli dSprites. Mean $\\pm$ SD quantify between-seed variation. The last column gives mean absolute within-run changes from epoch 50 to 300 for $V/A$, not a temporal variance.','tab:readouts','lcccccc',r'Dataset & $m$ & $V_5$ & $A_5$ & $V_{300}$ & $A_{300}$ & Change',lines,'footnotesize')

SYN=read('results/make49/stage6_e6_synthetic.json')['runs'];lines=[]
for name in ['SwissRoll','Torus']:
    for m in [1,2,3,4,6,8]:
        rs=[r for r in SYN if r['manifold']==name and r['m']==m and r['standardized']]
        pct=[100*r['ablation_delta']/r['mse'] for r in rs]
        lines.append(f"{'Swiss roll' if name=='SwissRoll' else name} & {m} & {pm([r['au'] for r in rs])} & {pm([r['mse'] for r in rs],4)} & {pm([r['mse_after_ablation'] for r in rs],4)} & {mean(pct):+.3f} \\\\")
tex_table('synthetic','Synthetic scale/ablation diagnostic, Gaussian likelihood, $\\beta=1$, $\\delta=0.01$, three training seeds. Data were standardized before the split; these are descriptive validation results, not leakage-free generalization estimates. Changes are mean per-run relative MSE changes (\\%).','tab:synthetic','lccccc',r'Data & $m$ & AU & MSE before & MSE after & Change',lines,'footnotesize')

plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':11,
 'lines.linewidth':1.6,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
colors=['#1f4d7a','#ae3d37','#31806d','#8061a7','#936f22','#333333','#7295b3']
fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
for ax,ds in zip(axes.flat,DS):
    for i,meth in enumerate(MAIN):
        if meth=='B8_ard_vae':continue
        vals=[select(ds,meth,b) for b in BETAS]
        ax.scatter([mean([r['accounted_seconds'] for r in v]) for v in vals],
                   [mean([r['val_mse'] for r in v]) for v in vals],s=28,color=colors[i],marker=['o','s','^','D','P','X','v'][i],label=NAMES[meth])
    ax.set(xscale='log',yscale='log',xlabel='Recorded training time (s)',ylabel='Validation MSE',title=DN[ds]);ax.grid(alpha=.2)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
save(fig,'quality_cost')
fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
show=['proposed','ascending_scan_Q','B6a_fondue','B6a_plus_Q']
for ax,ds in zip(axes.flat,DS):
    for i,meth in enumerate(show):
        rs=[select(ds,meth,b) for b in BETAS]
        ax.errorbar(BETAS,[mean([r['selected_raw'] for r in v]) for v in rs],
            yerr=[sd([r['selected_raw'] for r in v]) for v in rs],capsize=2,marker=['o','s','^','D'][i],ls=['-','--',':','-.'][i],color=colors[i],label=NAMES[meth])
    ax.set(xscale='log',xlabel=r'$\beta$',ylabel='Returned dimension',title=DN[ds]);ax.set_xticks(BETAS,labels=['.25','.5','1','2','4']);ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
save(fig,'beta')
fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
for ax,ds in zip(axes.flat,DS):
    grid=CFG['candidate_grids'][ds]
    for i,delta in enumerate([.001,.01,.1]):
        v=np.array([[sum(x>delta for x in cached(ds,m,s)['readouts']['B']['mu_var_per_dim']) for s in SEEDS] for m in grid])
        ax.errorbar(grid,v.mean(1),yerr=v.std(1,ddof=1),capsize=2,marker=['o','s','^'][i],ls=['-','--',':'][i],color=colors[i],label=fr'$\delta={delta:g}$')
    ax.plot(grid,grid,color='.6',ls=':',label='AU = capacity');ax.set(xlabel='Capacity',ylabel='AU count',title=DN[ds]);ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
save(fig,'delta')
fig,axes=plt.subplots(3,2,figsize=(8,8),layout='constrained')
for ax,(ds,cap) in zip(axes.flat,[('MNIST',12),('dSprites',6),('MNIST',32),('dSprites',16),('MNIST',64),('dSprites',32)]):
    rs=[r for r in TRAJ if r['dataset']==ds and r['m']==cap];xx=np.arange(1,301)
    for key,color,ls,label in [('train_mse',colors[0],'-','Train subset'),('val_mse',colors[1],'--','Validation')]:
        arr=np.array([r['curves'][key] for r in rs]);av=arr.mean(0);ss=arr.std(0,ddof=1)
        ax.plot(xx,av,c=color,ls=ls,label=label);ax.fill_between(xx,av-ss,av+ss,color=color,alpha=.15)
    for r in rs:ax.scatter(r['best_epoch'],r['curves']['val_mse'][r['best_epoch']-1],s=16,color='black',marker='x')
    ax.set(title=f'{DN[ds]}, m={cap}',xlabel='Epoch',ylabel='MSE',yscale='log');ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
save(fig,'learning')
fig,axes=plt.subplots(3,2,figsize=(8,8),layout='constrained')
for ax,(ds,cap) in zip(axes.flat,[('MNIST',12),('dSprites',6),('MNIST',32),('dSprites',16),('MNIST',64),('dSprites',32)]):
    rs=[r for r in TRAJ if r['dataset']==ds and r['m']==cap];xs=[1,2,5,50,300]
    for i,(a,b,label) in enumerate([('A','active_plus_mixed','Variable types'),('B','au','AU')]):
        arr=np.array([[r['readouts'][str(e)][a][b] for r in rs] for e in xs]);ax.errorbar(xs,arr.mean(1),yerr=arr.std(1,ddof=1),color=colors[i],marker=['o','s'][i],ls=['-','--'][i],capsize=2,label=label)
    ax.set(title=f'{DN[ds]}, m={cap}',xlabel='Epoch',ylabel='Count',xscale='log');ax.set_xticks([1,5,50,300],labels=['1','5','50','300']);ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
save(fig,'readouts')
REF=read('results/make49/stage5_e3_reference.json')
fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
for i,ds in enumerate(DS):
    sr=REF[ds]['sample_size'];p=REF[ds]['pca_space']
    axes[0].plot([r['n'] for r in sr],[r['rel_diff'] for r in sr],marker=['o','s','^','D'][i],color=colors[i],label=DN[ds])
    axes[1].plot([r['pca_dim'] for r in p],[r['twonn'] for r in p],marker=['o','s','^','D'][i],color=colors[i])
axes[0].axhline(.25,color='.5',ls='--');axes[0].set(xlabel='ID sample size',ylabel='Relative estimator disagreement',xscale='log')
axes[1].set(xlabel='PCA dimension',ylabel='TwoNN estimate')
for ax in axes:ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
save(fig,'id')

# Source manifest supports exact reaggregation and does not pretend to be a preregistration.
paths=[ROOT/'config/experiment_config_MAKE49.json',*sorted((ROOT/'src/make49').glob('*.py')),
       *sorted((ROOT/'results/make49').glob('*.json')),*sorted((ROOT/'results/make49/stage4_cache').glob('*.json')),
       Path(__file__)]
(OUT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(json.dumps(AUDIT,indent=2))
print('Generated tables, figures and 1600 audited method rows.')
