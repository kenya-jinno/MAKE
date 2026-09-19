"""Localize only figure text, reusing MAKE51's plotting instructions and records."""
from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'results/figures/NN51';FIG.mkdir(parents=True,exist_ok=True)
font_path=Path('/usr/local/texlive/2026/texmf-dist/fonts/truetype/public/ipaex/ipaexg.ttf')
if not font_path.exists():
    import subprocess
    font_path=Path(subprocess.check_output(['kpsewhich','ipaexg.ttf'],text=True).strip())
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font_path)).get_name(),
 'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':11,
 'lines.linewidth':1.6,'pdf.fonttype':42,'axes.unicode_minus':False})
def read(name):return json.loads((ROOT/name).read_text())
CFG=read('config/experiment_config_MAKE49.json')
DS=['MNIST','FashionMNIST','dSprites','CIFAR10']
DN=dict(zip(DS,['MNIST','Fashion-MNIST','dSprites','CIFAR-10']))
SEEDS=CFG['data_splits']['training_seeds'];BETAS=CFG['beta']['ladder']
CACHE={d:read(f'results/make49/stage4_cache/{d}.json') for d in DS}
def cache_key(ds,m,seed,beta=1):return f'vae|{ds}|m{m}|s{seed}|e300|es1|b{beta:g}'
def cached(ds,m,seed,beta=1):return CACHE[ds][cache_key(ds,m,seed,beta)]
candidate=cached
def mean(v):return float(np.mean(v))
def sd(v):return float(np.std(v,ddof=1))
METHOD_ROWS=list(csv.DictReader((ROOT/'results/make51/method_results.csv').open()))
for r in METHOD_ROWS:
    r['beta']=float(r['beta']);r['selected_raw']=float(r['selected_raw'])
def select(ds,meth,beta=1):
    return [r for r in METHOD_ROWS if r['dataset']==ds and r['method']==meth and r['beta']==beta]
NAMES={'proposed':'従来ID探索窓','ascending_scan_Q':'昇順 + Q',
 'B6a_fondue':'FONDUE','B6a_plus_Q':'FONDUE + Q'}
trajectory=read('results/make49/stage3_e7a_e2.json')
TRAJ=trajectory['runs']
colors=['#1f4d7a','#ae3d37','#31806d','#8061a7','#936f22','#333333','#7295b3']
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
translations={
 'Returned dimension':'返却次元数','Capacity':'潜在次元数','AU count':'AU数',
 'AU = capacity':'AU = 潜在次元数','Train subset':'学習部分集合','Validation':'検証',
 'Epoch':'エポック','Variable types':'変数型','Count':'個数','ID sample size':'ID推定の標本数',
 'Relative estimator disagreement':'推定量の相対不一致','PCA dimension':'PCA次元数',
 'TwoNN estimate':'TwoNN推定値','Validation MSE':'検証MSE',
 'Stochastic training constraint':'確率的学習制約','Training constraint (SSE)':'学習制約（二乗誤差和）',
 'Continuous gate':'連続ゲート','Binary gate':'二値ゲート','Target':'目標',
 'Posterior-mean reconstruction':'事後平均による再構成','Gates above 0.5':'0.5を超えるゲート',
 'Multiplier':'乗数','Constraint multiplier':'制約の乗数'
}
def localize(code):
    for en,ja in translations.items():
        code=code.replace("'"+en+"'",repr(ja))
    code=code.replace("f'MNIST, width {m}'","f'MNIST, {m}次元'")
    return code

# Keep data transforms, markers, intervals and axes identical to MAKE51.
old=(ROOT/'src/make51/rebuild_review.py').read_text()
code=old[old.index("save(fig,'quality_cost')")+len("save(fig,'quality_cost')"):old.index('# Source manifest')]
exec(compile(localize(code),'MAKE51_plot_instructions','exec'))

# Actual selected-candidate histories from the same keys and stopping epochs.
key=cache_key
ROWS=list(csv.DictReader((ROOT/'results/make51/search_replay.csv').open()))
for r in ROWS:
    r['seed']=int(r['seed']);r['beta']=float(r['beta']);r['selected_m']=int(r['selected_m'])
selected=(ROOT/'src/make51/replay_search.py').read_text()
code=selected[selected.index("fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained');curve_rows=[]"):
              selected.index("dumpcsv('selected_curve_keys.csv'")]
exec(compile(localize(code),'MAKE51_selected_plot_instructions','exec'))

# The MC grid and native HC records are not re-estimated or retrained.
V=ROOT/'results/make51/model_validation'
records={m:json.loads((V/f'ordinary_m{m}_s42.json').read_text()) for m in CFG['candidate_grids']['MNIST']}
hc=json.loads((V/'native_hc_s42.json').read_text())
summary=(ROOT/'src/make51/summarize_validation.py').read_text()
code=summary[summary.index("fig,axes=plt.subplots(1,2"):summary.index("print(json.dumps")]
exec(compile(localize(code),'MAKE51_validation_plot_instructions','exec'))
assert len(list(FIG.glob('*.pdf')))==8
print('Generated 8 Japanese figure PDFs from unchanged MAKE51 records and plotting instructions.')
