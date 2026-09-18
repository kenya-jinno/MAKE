"""Retrospective search replay on the immutable MAKE49 candidate snapshot.

No test values are read by a search policy. Costs are one canonical price per
candidate key; historical timing snapshots remain in a separate lineage CSV.
"""
from pathlib import Path
import csv, hashlib, json, re, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/make51';FIG=ROOT/'results/figures/MAKE51'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
def read(p):return json.loads((ROOT/p).read_text())
CFG=read('config/experiment_config_MAKE49.json')
DS=['MNIST','FashionMNIST','dSprites','CIFAR10']
DN=dict(zip(DS,['MNIST','Fashion-MNIST','dSprites','CIFAR-10']))
SEEDS=CFG['data_splits']['training_seeds'];BETAS=CFG['beta']['ladder']
CACHE={d:read(f'results/make49/stage4_cache/{d}.json') for d in DS}
RAW=read('results/make49/stage4_methods.json')['results']
PRUNE=read('results/make49/stage7_b7b8.json')['results']
def key(ds,m,s,b=1,ep=300,es=1):return f'vae|{ds}|m{m}|s{s}|e{ep}|es{es}|b{b:g}'
def candidate(ds,m,s,b=1):return CACHE[ds][key(ds,m,s,b)]
def info(ds):return next(r['id_info'] for r in RAW if r['dataset']==ds and r['method']=='proposed' and r['beta']==1)
ID={d:info(d) for d in DS}
BANK={d:sum(v['elapsed_seconds'] for k,v in CACHE[d].items() if k.startswith('ae|')) for d in DS}
assert all(sum(k.startswith('ae|') for k in CACHE[d])==9 for d in DS)
def avg(xs):return float(np.mean(xs))
def pm(xs,d=1):return f'${np.mean(xs):.{d}f} \\pm {np.std(xs,ddof=1):.{d}f}$'
def dumpcsv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(name,caption,label,cols,header,rows,size='small'):
    text=(f'\\begin{{table}}[H]\n\\caption{{{caption}}}\\label{{{label}}}\n\\{size}\n'
          '\\setlength{\\tabcolsep}{3pt}\n'
          f'\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}@{{}}}}\n\\toprule\n'
          +header+' \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular*}\n\\end{table}\n')
    (OUT/f'{name}.tex').write_text(text)

SPEC=dict(date='2026-09-19',status='retrospective, not preregistered',
          primary=dict(coefficient=2,R=.15,A=.25),
          coefficient_grid=[1,1.5,2,3,4],R_grid=[.1,.15,.25,.5],A_grid=[.15,.25,.4,.85,2],
          local_rule='start at ceil-grid c*ID; if feasible descend until first failure; otherwise ascend until first feasible',
          fallback='ascending Q',control='same local rule from lower middle grid index, without ID bank',
          certification='evaluate all unqueried widths below local output; choose smallest feasible',
          budget='finish cached path (at most entire grid); flag original soft time limit after accounting',
          times='canonical candidate snapshot, not newly measured wall-clock',
          score_use='validation MSE only; test metrics attached after decisions')
(OUT/'replay_specification.json').write_text(json.dumps(SPEC,indent=2))

def replay(ds,s,b,policy,c=2,R=.15,A=.25):
    grid=CFG['candidate_grids'][ds];anchor=grid[-1]
    threshold=1.1*candidate(ds,anchor,s,b)['val']['mse']
    vals={m:candidate(ds,m,s,b)['val']['mse'] for m in grid}
    feasible={m:vals[m]<=threshold for m in grid}
    optimum=next(m for m in grid if feasible[m]);asked=[anchor]
    passed=ID[ds]['relative_range']<=R and ID[ds]['estimator_rel_diff']<=A
    id_policy=policy in ['Legacy window','ID local','ID certified']
    fallback=id_policy and not passed
    def request(m):
        if m not in asked:asked.append(m)
        return feasible[m]
    def ascending():
        for m in grid:
            if request(m):return m
    def local(start):
        ix=grid.index(start)
        if request(start):
            selected=start
            for m in reversed(grid[:ix]):
                if not request(m):break
                selected=m
            return selected
        for m in grid[ix+1:]:
            if request(m):return m
        raise AssertionError('anchor must satisfy Q')
    if policy=='Full grid Q':
        for m in grid:request(m)
        chosen=optimum
    elif policy=='Ascending Q' or (fallback and policy!='Legacy window'):
        chosen=ascending()
    elif policy=='Legacy window':
        old=next(r for r in RAW if r['dataset']==ds and r['seed']==s and r['beta']==b and r['method']=='proposed')
        for t in old['trace']:
            if t['kind']=='vae':
                m=int(re.search(r'\|m(\d+)\|',t['key'])[1]);request(m)
        chosen=old['selected_m']
    else:
        start=(grid[(len(grid)-1)//2] if policy=='Midpoint local' else next((m for m in grid if m>=c*ID[ds]['d_hat']),anchor))
        chosen=local(start)
        if policy=='ID certified':
            for m in grid:
                if m>=chosen:break
                if request(m):chosen=m;break
    assert len(asked)==len(set(asked)) and len(asked)<=len(grid)
    if policy!='Legacy window':assert feasible[chosen]
    candidate_cost=sum(candidate(ds,m,s,b)['elapsed_seconds'] for m in asked)
    bank=BANK[ds] if id_policy else 0.
    # Only now attach held-out test summaries; they never inform the policy.
    return dict(dataset=ds,seed=s,beta=b,policy=policy,c=c,R=R,A=A,
                id_accepted=bool(passed) if id_policy else None,fallback=bool(fallback),
                selected_m=chosen,full_grid_min=optimum,excess_width=chosen-optimum,
                excess_grid_steps=grid.index(chosen)-grid.index(optimum),exact=chosen==optimum,
                quality_met=feasible[chosen],T_D=threshold,
                requested_count=len(asked),requested_widths=';'.join(map(str,asked)),
                canonical_keys=';'.join(key(ds,m,s,b) for m in asked),
                candidate_seconds=candidate_cost,reference_seconds=bank,
                cold_seconds=candidate_cost+bank,amortized25_seconds=candidate_cost+bank/25,
                soft_budget_exceeded=(candidate_cost+bank)>CFG['budget_and_termination']['max_wallclock_seconds_per_dataset_method_seed'][ds],
                val_mse=vals[chosen],test_mse=candidate(ds,chosen,s,b)['test']['mse'],
                nonmonotone_Q=any(feasible[a] and not feasible[z] for a,z in zip(grid,grid[1:])))

POLICIES=['Full grid Q','Ascending Q','Legacy window','ID local','Midpoint local','ID certified']
ROWS=[replay(d,s,b,p) for d in DS for b in BETAS for s in SEEDS for p in POLICIES]
dumpcsv('search_replay.csv',ROWS)
SENS=[replay(d,s,b,'ID local',c,R,A) for d in DS for b in BETAS for s in SEEDS
      for c in SPEC['coefficient_grid'] for R in SPEC['R_grid'] for A in SPEC['A_grid']]
dumpcsv('search_sensitivity.csv',SENS)

for block,datasets in enumerate([DS[:2],DS[2:]],1):
    lines=[]
    for ds in datasets:
        lines.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{DN[ds]}}}}} \\\\')
        for policy in POLICIES:
            rr=[r for r in ROWS if r['dataset']==ds and r['policy']==policy and r['beta']==1]
            lines.append(f"{policy} & {pm([r['selected_m'] for r in rr])} & {sum(r['quality_met'] for r in rr)}/5 & {sum(r['exact'] for r in rr)}/5 & {avg([r['excess_width'] for r in rr]):.1f} & {avg([r['requested_count'] for r in rr]):.1f} & {avg([r['cold_seconds'] for r in rr]):.0f} \\\\")
        lines.append('\\midrule')
    table(f'replay{block}',r'Retrospective search replay at $\beta=1$, five seeds. $Q$: target attainment; exact: agreement with the full-grid smallest qualifying width; $\Delta m$: mean excess width. Calls include the anchor. Cost uses one canonical saved time per key plus the full reference bank for ID policies, in seconds; it is not new wall-clock timing. Gaussian likelihood except Bernoulli dSprites.',f'tab:replay{block}','lcccccc',r'Policy & Width & $Q$ & Exact & $\Delta m$ & Calls & Cold cost',lines[:-1])

lines=[]
for ds in DS:
    for c in SPEC['coefficient_grid']:
        rr=[r for r in SENS if r['dataset']==ds and r['beta']==1 and r['c']==c and r['R']==.15 and r['A']==.25]
        lines.append(f"{DN[ds]} & {c:g} & {pm([r['selected_m'] for r in rr])} & {sum(r['quality_met'] for r in rr)}/5 & {sum(r['exact'] for r in rr)}/5 & {avg([r['requested_count'] for r in rr]):.1f} & {avg([r['cold_seconds'] for r in rr]):.0f} \\\\")
table('coefficient',r'Coefficient sensitivity for ID-local replay at $\beta=1$, with $R\le0.15$ and $A\le0.25$. All five coefficients are reported; rejected datasets use the same ascending fallback. Five seeds; likelihoods as in Table~\ref{tab:data}.','tab:coefficient','lcccccc',r'Dataset & $c$ & Width & $Q$ & Exact & Calls & Cold cost (s)',lines)

lines=[]
for R,A in [(.1,.25),(.15,.25),(.25,.25),(.15,.4),(.15,.85),(.5,2)]:
    for ds in DS:
        rr=[r for r in SENS if r['dataset']==ds and r['beta']==1 and r['c']==2 and r['R']==R and r['A']==A]
        lines.append(f"{R:g}/{A:g} & {DN[ds]} & {'ID' if rr[0]['id_accepted'] else 'FB'} & {avg([r['selected_m'] for r in rr]):.1f} & {sum(r['quality_met'] for r in rr)}/5 & {sum(r['exact'] for r in rr)}/5 & {avg([r['requested_count'] for r in rr]):.1f} \\\\")
table('reliability',r'Reliability-threshold sensitivity at $\beta=1$, $c=2$. ID: local ID start; FB: ascending fallback. Extreme tolerances are stress conditions, not recommended thresholds. Each result uses five seeds and the same reference bank. The full factorial analysis is supplied in the supplement.','tab:reliability','llccccc',r'$R_{\max}/A_{\max}$ & Dataset & Use & Mean width & $Q$ & Exact & Calls',lines)

lines=[]
for ds in DS:
    rr=[r for r in ROWS if r['dataset']==ds and r['policy']=='ID local']
    lines.append(f"{DN[ds]} & {BANK[ds]:.1f} & {avg([r['cold_seconds'] for r in rr]):.1f} & {avg([r['candidate_seconds'] for r in rr]):.1f} & {avg([r['amortized25_seconds'] for r in rr]):.1f} & {sum(r['exact'] for r in rr)}/25 & {sum(r['nonmonotone_Q'] for r in rr)}/25 \\\\")
table('amortization',r'ID-local costs across five weights and five candidate seeds (25 tasks per dataset). Cold charges the bank each time; warm assumes an existing bank; batch allocates its training cost once across 25 tasks. All exclude unrecorded ID/evaluation overhead. Nonmono counts qualifying masks that change from pass to fail as width increases.','tab:amortization','lcccccc',r'Dataset & Bank (s) & Cold (s) & Warm (s) & Batch (s) & Exact & Nonmono',lines)

# Resolve historical prices to full logical keys, without claiming weight identity.
LINEAGE=[]
for r in RAW+PRUNE:
    ds,s,b=r['dataset'],r['seed'],r.get('beta',r.get('beta_context'))
    a=CFG['candidate_grids'][ds][-1];f=r.get('final_model');m=f.get('m_on_grid',r['selected_m']) if f else None
    for role,width in [('anchor',a),('final',m)]:
        if width is None:continue
        ck=key(ds,width,s,b);cur=CACHE[ds].get(ck)
        match=[(j,t) for j,t in enumerate(r.get('trace',[])) if t['kind']=='vae' and re.search(fr'\|m{width}\|e300\|es1(?:\||$)',t['key'])]
        if match:
            j,t=match[0];old=t['seconds'];source=f'stage4_methods.json: ({ds},{r["method"]},{s},{b:g}).trace[{j}]';tk=t['key']
        elif role=='final' and 'final_train_seconds' in r:
            old=r['final_train_seconds'];source=f'stage7_b7b8.json: ({ds},{r["method"]},{s},{b:g}).final_train_seconds';tk='wrapper candidate key reconstructed'
        else:
            old=None;source='not charged in original ledger; MAKE50 used current snapshot';tk='not in trace'
        same=None if role=='anchor' or not f or not cur else f['val']==cur['val'] and f['test']==cur['test']
        LINEAGE.append(dict(dataset=ds,method=r['method'],seed=s,beta=b,role=role,
                            cache_key=ck,original_trace_key=tk,original_seconds=old,
                            original_source=source,canonical_seconds=cur['elapsed_seconds'] if cur else None,
                            canonical_source=f'results/make49/stage4_cache/{ds}.json:{ck}',
                            final_metric_dicts_identical=same,checkpoint_identity='not available',
                            requested_epochs=300,early_stopping=True))
dumpcsv('cost_lineage.csv',LINEAGE)
lines=[]
for ds in DS:
    rr=[r for r in LINEAGE if r['dataset']==ds and r['beta']==1 and r['original_seconds'] is not None and r['canonical_seconds'] is not None]
    changes=[r for r in rr if abs(r['original_seconds']-r['canonical_seconds'])>1e-8]
    lines.append(f"{DN[ds]} & {len(rr)} & {len(changes)} & {max(abs(r['original_seconds']-r['canonical_seconds']) for r in rr):.2f} \\\\")
table('timing_audit',r'Anchor/final timing audit at $\beta=1$. Entries are method-role-seed records, not independent training runs. Different prices can be attached to the same logical cache key; full key, source and both durations are supplied in \texttt{cost\_lineage.csv}. Stored summary metrics do not establish checkpoint identity.','tab:timing_audit','lccc',r'Dataset & Matched entries & Different times & Max. difference (s)',lines)

# Reconstruct paper Algorithm 1 transitions from saved oracle observations.
FONDUE=[]
for r in RAW:
    if r['method']!='B6a_fondue':continue
    lower=0;upper=math.inf;p=r['search_path'][0]['p'];ok=True
    for step in r['search_path']:
        ok &= p==step['p']
        if step['gap']<=r['threshold']:lower=p;p=min(2*p,upper)
        else:upper=p;p=math.floor((lower+upper)/2)
    record=dict(dataset=r['dataset'],seed=r['seed'],beta=r['beta'],
                path_matches=bool(ok),returned_matches=p==r['selected_m'],
                normal_termination=p==lower)
    assert all(record[k] for k in ['path_matches','returned_matches','normal_termination'])
    FONDUE.append(record)
(OUT/'fondue_control_flow.json').write_text(json.dumps(FONDUE,indent=2))

# The archived ARD score already contains the squared finite step. Dividing
# by p exactly recovers the correctly normalized finite-step weighted score.
ARD=[]
for r in PRUNE:
    if r['method']!='B8_ard_vae' or r['beta_context']!=1:continue
    ds,s=r['dataset'],r['seed'];p=np.asarray(r['prior_var']);q=np.asarray(r['relevance_score'])
    assert np.all(p>0) and np.all(q>=0)
    fixed=q/p;order=np.argsort(fixed)[::-1]
    count=int(np.searchsorted(np.cumsum(fixed[order])/fixed.sum(),.99)+1)
    grid=CFG['candidate_grids'][ds];mapped=min(grid,key=lambda m:abs(m-count))
    cr=candidate(ds,mapped,s);T=1.1*candidate(ds,grid[-1],s)['val']['mse']
    ARD.append(dict(dataset=ds,seed=s,old_raw=r['selected_m'],normalized_raw=count,
                    old_grid=r['final_model']['m_on_grid'],normalized_grid=mapped,
                    normalized_score=fixed.tolist(),val_mse=cr['val']['mse'],test_mse=cr['test']['mse'],
                    quality_met=cr['val']['mse']<=T))
(OUT/'ard_normalized_readout.json').write_text(json.dumps(ARD,indent=2))
lines=[]
for ds in DS:
    rr=[r for r in ARD if r['dataset']==ds]
    lines.append(f"{DN[ds]} & {pm([r['old_raw'] for r in rr])} & {pm([r['normalized_raw'] for r in rr])} & {pm([r['normalized_grid'] for r in rr])} & {sum(r['quality_met'] for r in rr)}/5 & {pm([1000*r['test_mse'] for r in rr],2)} \\\\")
table('ard_normalized',r'Corrected finite-step relevance readout from saved scores and prior variances at the primary context. The same stored perturbations are normalized by their squared step before variance weighting; no new ARD training is involved. This validates an algebraic readout correction, not exact Jacobian relevance or the source training procedure. Test MSE $\times10^3$ is for transferred ordinary VAEs.','tab:ard_normalized','lccccc',r'Dataset & Old raw & Corrected raw & Grid width & $Q$ & Test MSE',lines)

# Actual early-stopped selected-candidate validation curves, including CIFAR-10.
plt.rcParams.update({'font.size':12,'legend.fontsize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained');curve_rows=[]
for ax,ds in zip(axes.flat,DS):
    for s in SEEDS:
        rr=next(r for r in ROWS if r['dataset']==ds and r['seed']==s and r['beta']==1 and r['policy']=='Ascending Q')
        cr=candidate(ds,rr['selected_m'],s)
        xs=np.arange(1,len(cr['curve_val_mse'])+1)
        ax.plot(xs,cr['curve_val_mse'],alpha=.8,label=f"s={s}, m={rr['selected_m']}")
        ax.scatter(cr['best_epoch'],cr['curve_val_mse'][cr['best_epoch']-1],c='black',marker='x',s=18)
        curve_rows.append(dict(dataset=ds,seed=s,width=rr['selected_m'],best_epoch=cr['best_epoch'],epochs_run=cr['epochs_run'],cache_key=key(ds,rr['selected_m'],s)))
    ax.set(title=DN[ds],xlabel='Epoch',ylabel='Validation MSE',yscale='log');ax.legend(fontsize=11,loc='upper right');ax.grid(alpha=.2)
fig.savefig(FIG/'selected_curves.pdf',bbox_inches='tight');plt.close(fig)
dumpcsv('selected_curve_keys.csv',curve_rows)

summary={d:{p:{k:avg([r[k] for r in ROWS if r['dataset']==d and r['beta']==1 and r['policy']==p]) for k in ['selected_m','excess_width','requested_count','cold_seconds','candidate_seconds','exact']} for p in POLICIES} for d in DS}
(OUT/'replay_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(f'Replay rows={len(ROWS)}; sensitivity rows={len(SENS)}; lineage rows={len(LINEAGE)}; corrected ARD rows={len(ARD)}')
