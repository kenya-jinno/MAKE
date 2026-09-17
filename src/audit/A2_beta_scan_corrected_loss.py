"""修正損失(標準規約)のもとで β の作動点を決める"""
import os, sys, json, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import DataLoader, TensorDataset
from src.models.vae import VAE
from src.metrics.structure import active_units
from src.metrics.intrinsic_dim import twonn_estimate
import torchvision
dev='cuda'
tr=torchvision.datasets.MNIST(root='~/.cache/datasets',train=True,download=False)
te=torchvision.datasets.MNIST(root='~/.cache/datasets',train=False,download=False)
Xtr=(tr.data[:20000].float().view(-1,784)/255.).numpy()
Xte=(te.data[:3000].float().view(-1,784)/255.).numpy()
def loss_fixed(rec,x,mu,lv,beta):
    return F.mse_loss(rec,x,reduction='sum')/x.size(0) + beta*(-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/x.size(0))
def run(m,beta,seed=42,epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    v=VAE(input_dim=784,hidden_dims=[512,256],latent_dim=m).to(dev)
    opt=torch.optim.Adam(v.parameters(),lr=1e-3)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr)),batch_size=128,shuffle=True); v.train()
    for _ in range(epochs):
        for (x,) in dl:
            x=x.to(dev); opt.zero_grad(); r,mu,lv=v(x)
            loss_fixed(r,x,mu,lv,beta).backward(); opt.step()
    v.eval()
    with torch.no_grad():
        X=torch.tensor(Xte).to(dev); r,mu,_=v(X)
        return active_units(mu.cpu().numpy(),1e-2), F.mse_loss(r,X).item(), float(twonn_estimate(mu.cpu().numpy()))
print(f"{'beta':>6} {'m':>4} {'AU':>4} {'MSE':>9} {'TwoNN':>7}")
res=[]
for beta in [0.05,0.1,0.25,0.5,1.0]:
    for m in [16,32,64]:
        au,mse,tn=run(m,beta)
        print(f"{beta:6.2f} {m:4d} {au:4d} {mse:9.5f} {tn:7.2f}", flush=True)
        res.append(dict(beta=beta,m=m,au=au,mse=mse,twonn=tn))
json.dump(res,open(os.path.join('results','audit_MAKE49','A2_beta_scan_corrected_loss.json'),'w'),indent=1)
