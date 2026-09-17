"""U3 検証: 現行損失(KL=torch.mean) vs 修正損失(KL=潜在次元で和) の AU 比較"""
import os, sys, json, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import DataLoader, TensorDataset
from src.models.vae import VAE
from src.metrics.structure import active_units
import torchvision

dev = 'cuda'
tr = torchvision.datasets.MNIST(root='~/.cache/datasets', train=True, download=False)
te = torchvision.datasets.MNIST(root='~/.cache/datasets', train=False, download=False)
Xtr = (tr.data[:20000].float().view(-1,784)/255.).numpy()
Xte = (te.data[:3000].float().view(-1,784)/255.).numpy()

def loss_current(rec,x,mu,lv,beta):   # 現行: KL を潜在次元でも平均
    return F.mse_loss(rec,x,reduction='sum')/x.size(0) + beta*(-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp()))
def loss_fixed(rec,x,mu,lv,beta):     # 修正: KL は潜在次元で和、バッチで平均
    return F.mse_loss(rec,x,reduction='sum')/x.size(0) + beta*(-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/x.size(0))

def run(m, lossfn, beta, seed, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    v = VAE(input_dim=784, hidden_dims=[512,256], latent_dim=m).to(dev)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr)), batch_size=128, shuffle=True)
    v.train()
    for _ in range(epochs):
        for (x,) in dl:
            x = x.to(dev); opt.zero_grad()
            r,mu,lv = v(x); l = lossfn(r,x,mu,lv,beta); l.backward(); opt.step()
    v.eval()
    with torch.no_grad():
        X = torch.tensor(Xte).to(dev); r,mu,_ = v(X)
        mse = F.mse_loss(r,X).item(); Z = mu.cpu().numpy()
    return active_units(Z, 1e-2), mse

out=[]
print(f"{'m':>4} {'β_eff(現行)':>12} | {'AU現行':>7} {'MSE現行':>9} | {'AU修正':>7} {'MSE修正':>9}")
for m in [8, 16, 32, 64]:
    a1,e1 = run(m, loss_current, 4.0, 42)
    a2,e2 = run(m, loss_fixed,   4.0, 42)
    print(f"{m:4d} {4.0/m:12.4f} | {a1:7d} {e1:9.5f} | {a2:7d} {e2:9.5f}")
    out.append(dict(m=m, beta_eff_current=4.0/m, au_current=a1, mse_current=e1, au_fixed=a2, mse_fixed=e2))
json.dump(out, open(os.path.join('results','audit_MAKE49','A2_loss_convention_ablation.json'),'w'), indent=1)
