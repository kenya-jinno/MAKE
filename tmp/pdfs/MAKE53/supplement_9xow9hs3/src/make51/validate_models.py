"""Bounded MAKE51 validation: a new MNIST grid and native HC diagnostic.

This does not recreate missing CUDA checkpoints. It saves its own CPU
checkpoints, curves, fixed-evaluation MC ELBOs, and a frozen selection record.
Run from the repository root; existing completed records are reused.
"""
from pathlib import Path
import hashlib, json, math, platform, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.make49.models import FCVAE
from src.make49.pruning_methods import HardConcreteGate

OUT = ROOT / 'results/make51/model_validation'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
SPEC = dict(date='2026-09-19', dataset='MNIST', grid=GRID, training_seed=42,
            max_epochs=300, patience=30, batch_size=128, lr=.001, beta=1,
            checkpoint='best posterior-mean ELBO proxy', mc_samples=32,
            mc_seed=20260919, mc_gaussian_variance=.5, device='cpu',
            train_curve_subset=2000, train_curve_seed=20260919,
            native_hc_seed=42, native_hc_epochs=300,
            scope='one new seed, full MNIST grid; separate from archived CUDA results',
            python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, platform=platform.platform(), threads=4)
spec_path=OUT/'specification.json'
if spec_path.exists():
    assert json.loads(spec_path.read_text())==SPEC
else:
    spec_path.write_text(json.dumps(SPEC,indent=2))

tr=torchvision.datasets.MNIST(root=str(ROOT/'data'),train=True,download=False)
te=torchvision.datasets.MNIST(root=str(ROOT/'data'),train=False,download=False)
pool=tr.data.float().reshape(-1,784)/255
testpool=te.data.float().reshape(-1,784)/255
idx=np.random.RandomState(20260912).permutation(60000)[:20000]
pos=np.random.RandomState(20260913).permutation(20000)
it,iv=idx[pos[:18000]],idx[pos[18000:]]
ie=np.random.RandomState(20260914).permutation(10000)[:3000]
Xtr,Xv,Xt=pool[it],pool[iv],testpool[ie]
curve_idx=np.random.RandomState(SPEC['train_curve_seed']).choice(18000,2000,replace=False)
Xcurve=Xtr[curve_idx]
split_hash=hashlib.sha256(b''.join(np.asarray(a,dtype=np.int64).tobytes() for a in [it,iv,ie])).hexdigest()

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

@torch.no_grad()
def evaluate(model,X):
    model.eval(); rec=kl=0.
    for xb in X.split(512):
        mu,lv=model.encode(xb);xh=model.decode(mu)
        rec+=float((xh-xb).square().sum())
        kl+=float(.5*(mu.square()+lv.exp()-1-lv).sum())
    rec/=len(X);kl/=len(X)
    return dict(mse=rec/784,rec=rec,kl=kl,proxy=-(rec+kl))

@torch.no_grad()
def mc_elbo(model,X):
    model.eval();mus=[];lvs=[]
    for xb in X.split(512):
        mu,lv=model.encode(xb);mus.append(mu);lvs.append(lv)
    mu,lv=torch.cat(mus),torch.cat(lvs)
    kl=float(.5*(mu.square()+lv.exp()-1-lv).sum()/len(X))
    gen=torch.Generator().manual_seed(SPEC['mc_seed']);values=[]
    for k in range(SPEC['mc_samples']):
        # Common 64-dimensional draws, truncated for smaller models.
        eps=torch.randn(len(X),max(GRID),generator=gen)[:,:mu.shape[1]]
        z=mu+eps*torch.exp(.5*lv);rec=0.
        for start in range(0,len(X),512):
            rec+=float((model.decode(z[start:start+512])-X[start:start+512]).square().sum())
        values.append(-(rec/len(X)+kl+392*math.log(math.pi)))
    return dict(elbo=float(np.mean(values)),mc_se=float(np.std(values,ddof=1)/np.sqrt(len(values))),
                draw_means=values,kl=kl,K=SPEC['mc_samples'],evaluation_seed=SPEC['mc_seed'])

for m in [64,6]+[m for m in GRID if m not in [64,6]]:
    path=OUT/f'ordinary_m{m}_s42.json'
    if path.exists():continue
    torch.manual_seed(42);np.random.seed(42)
    model=FCVAE(784,m);opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=None;curves=[];t0=time.perf_counter()
    for ep in range(1,301):
        model.train();perm=torch.randperm(len(Xtr))
        for ids in perm.split(128):
            xb=Xtr[ids];opt.zero_grad();xh,mu,lv=model(xb)
            loss=((xh-xb).square().sum()+.5*(mu.square()+lv.exp()-1-lv).sum())/len(xb)
            loss.backward();opt.step()
        v=evaluate(model,Xv);t=evaluate(model,Xcurve)
        curves.append(dict(epoch=ep,train_mse=t['mse'],**v))
        if best is None or v['proxy']>best['proxy']:
            best=dict(epoch=ep,proxy=v['proxy'],state={k:x.clone() for k,x in model.state_dict().items()})
        if ep%50==0:print(f'MNIST m={m} epoch={ep} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
        if ep-best['epoch']>=30:break
    elapsed=time.perf_counter()-t0
    model.load_state_dict(best['state'])
    torch.save(dict(state_dict=best['state'],m=m,seed=42,best_epoch=best['epoch'],split_sha256=split_hash),OUT/f'ordinary_m{m}_s42.pt')
    rec=dict(m=m,seed=42,epochs_run=ep,best_epoch=best['epoch'],training_seconds=elapsed,
             split_sha256=split_hash,curves=curves,val=evaluate(model,Xv),mc_val=mc_elbo(model,Xv))
    write(path,rec)
    print(f'COMPLETE ordinary m={m}, ep={ep}, MC ELBO={rec["mc_val"]["elbo"]:.4f}',flush=True)

# Freeze capacity choices from validation before accessing test outcomes.
records=[json.loads((OUT/f'ordinary_m{m}_s42.json').read_text()) for m in GRID]
choices={'MSE':min(records,key=lambda r:r['val']['mse'])['m'],
         'ELBO proxy':max(records,key=lambda r:r['val']['proxy'])['m'],
         'MC ELBO':max(records,key=lambda r:r['mc_val']['elbo'])['m']}
write(OUT/'frozen_selection.json',dict(choices=choices,criterion='validation only',specification=SPEC))
tests={}
for m in sorted(set(choices.values())):
    model=FCVAE(784,m);model.load_state_dict(torch.load(OUT/f'ordinary_m{m}_s42.pt',weights_only=True)['state_dict'])
    tests[str(m)]=evaluate(model,Xt)
write(OUT/'selection_test.json',dict(choices=choices,test=tests))

# Instrumented local HC implementation; explicitly a diagnostic, not ARM.
native_path=OUT/'native_hc_s42.json'
if not native_path.exists():
    torch.manual_seed(42);np.random.seed(42)
    model=FCVAE(784,64);gate=HardConcreteGate(64)
    lam_raw=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam(list(model.parameters())+list(gate.parameters())+[lam_raw],lr=.001)
    tau=1.1*next(r for r in records if r['m']==64)['val']['mse'];ma=None;curves=[]
    t0=time.perf_counter()

    @torch.no_grad()
    def native_eval(X,mode='mean',binary=False,K=1):
        model.eval();gate.eval();gen=torch.Generator().manual_seed(20260920);total=0.
        for k in range(K):
            for xb in X.split(512):
                mu,lv=model.encode(xb)
                z=mu if mode=='mean' else mu+torch.randn(mu.shape,generator=gen)*torch.exp(.5*lv)
                g=gate.open_mask().float().unsqueeze(0) if binary else gate(len(xb))
                total+=float((model.decode(z*g)-xb).square().sum())
        return total/(K*len(X)*784)

    for ep in range(1,301):
        model.train();gate.train()
        for xb in Xtr.split(128):
            opt.zero_grad();mu,lv=model.encode(xb)
            z=mu+torch.randn_like(mu)*torch.exp(.5*lv);g=gate(len(xb));xh=model.decode(z*g)
            C=(xh-xb).square().sum()/len(xb)-784*tau
            ma=C.detach() if ma is None else .99*ma+.01*C.detach()
            lam=torch.clamp(F.softplus(lam_raw)**2,1e-6,1e6)
            kl=(.5*(mu.square()+lv.exp()-1-lv)*g).sum()/len(xb)
            loss=kl+lam.detach()*(ma+C-C.detach())-lam*ma.detach()
            if ma.item()<=0:loss=loss+gate.expected_l0()
            loss.backward();opt.step()
        curves.append(dict(epoch=ep,constraint_ma_sse=float(ma),
                           train_sampled_gate_mse_ma=float(ma)/784+tau,
                           lambda_value=float(torch.clamp(F.softplus(lam_raw.detach())**2,1e-6,1e6)),
                           gates=int(gate.open_mask().sum()),
                           val_mean_continuous_mse=native_eval(Xv),
                           val_mean_binary_mse=native_eval(Xv,binary=True)))
        if ep%50==0:print(f'HC epoch={ep} C_ma={float(ma):.3f} gates={curves[-1]["gates"]}',flush=True)
    torch.save(dict(model=model.state_dict(),gate=gate.state_dict(),lambda_raw=lam_raw.detach()),OUT/'native_hc_s42.pt')
    m_raw=int(gate.open_mask().sum());m_grid=min(GRID,key=lambda m:abs(m-m_raw))
    transferred=FCVAE(784,m_grid);transferred.load_state_dict(torch.load(OUT/f'ordinary_m{m_grid}_s42.pt',weights_only=True)['state_dict'])
    native=dict(val_mean=native_eval(Xv),test_mean=native_eval(Xt),
                val_sampled=native_eval(Xv,'sampled',K=32),test_sampled=native_eval(Xt,'sampled',K=32),
                val_binary_mean=native_eval(Xv,binary=True),test_binary_mean=native_eval(Xt,binary=True))
    write(native_path,dict(seed=42,tau=tau,raw_count=m_raw,grid_width=m_grid,curves=curves,
                          native=native,transfer_val=evaluate(transferred,Xv),transfer_test=evaluate(transferred,Xt),
                          elapsed_seconds=time.perf_counter()-t0,split_sha256=split_hash,
                          inference='continuous deterministic hard-concrete gate unless binary is specified',
                          training='sampled posterior and sampled hard-concrete gates on training minibatches'))
print('MODEL VALIDATION COMPLETE',flush=True)
