"""段階 3 校正の診断: dSprites の崩壊が likelihood 由来かを切り分ける。

校正実行で dSprites Conv-VAE は beta=1 でも beta=0.05 でも完全崩壊した
（AU=0, KL=0, val_rec≈173.5 ＝ 全ゼロ予測の 173.74 とほぼ同じ）。

dSprites は二値画像で白画素は 4.24% しかない。設定表 §2 は全データで
Gaussian（MSE）likelihood を指定しているが、FONDUE を含む dSprites の
標準的な設定は Bernoulli（BCE）である。

**この診断は likelihood を都合よく選ぶためのものではない。**
崩壊の原因が likelihood にあるかを確かめ、設定表の欠落を埋める判断材料を作る。

実行: プロジェクト直下で python3 src/make49/stage3_dsprites_likelihood.py
"""
import json, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, DEVICE, provenance, set_seed
from src.make49.datasets import load_dsprites
from src.make49.models import build
from src.metrics.structure import active_units

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)
EPOCHS, M, DELTA = 60, 16, CFG['au']['delta_primary']


def kl_term(mu, lv, B):
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B


def run(likelihood, beta, Xtr, Xva, epochs=EPOCHS):
    set_seed(CFG['data_splits']['training_seeds'][0])
    model = build({'kind': 'conv', 'in_ch': 1, 'size': 64}, M).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.as_tensor(Xtr)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xtr_t[perm[i:i + 128]].to(DEVICE)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            B = xb.size(0)
            if likelihood == 'gaussian':
                rec = F.mse_loss(xh, xb, reduction='sum') / B
            else:
                rec = F.binary_cross_entropy(xh.clamp(1e-6, 1 - 1e-6), xb, reduction='sum') / B
            (rec + beta * kl_term(mu, lv, B)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        recs, kls, mus = [], [], []
        Xva_t = torch.as_tensor(Xva)
        for i in range(0, len(Xva_t), 512):
            xb = Xva_t[i:i + 512].to(DEVICE)
            xh, mu, lv = model(xb)
            B = xb.size(0)
            recs.append((F.mse_loss(xh, xb, reduction='sum') / B).item() * B)
            kls.append(kl_term(mu, lv, B).item() * B)
            mus.append(mu.cpu())
        N = len(Xva_t)
        mse_sum = sum(recs) / N
        kl = sum(kls) / N
        Z = torch.cat(mus).numpy()
    return {'likelihood': likelihood, 'beta': beta, 'au': int(active_units(Z, DELTA)),
            'val_sse_per_sample': mse_sum, 'val_kl': kl}


def main():
    d = load_dsprites()
    Xtr, Xva = d['X_train'], d['X_val']
    zero_sse = float((Xva ** 2).sum(axis=(1, 2, 3)).mean())
    mean_sse = float(((Xva - Xva.mean(0)) ** 2).sum(axis=(1, 2, 3)).mean())
    print(f"dSprites 参照値: 全ゼロ予測 SSE={zero_sse:.2f}  平均画像予測 SSE={mean_sse:.2f}")
    print(f"学習 {EPOCHS} epochs, m={M}, seed={CFG['data_splits']['training_seeds'][0]}\n")
    print(f"{'likelihood':>11} {'beta':>6} {'AU':>6} {'val SSE':>10} {'val KL':>9} {'判定':>14}")
    rows = []
    for lik in ['gaussian', 'bernoulli']:
        for beta in [1.0, 0.25]:
            r = run(lik, beta, Xtr, Xva)
            better = r['val_sse_per_sample'] < mean_sse * 0.9
            verdict = '学習している' if (r['au'] > 0 and better) else '崩壊'
            r['verdict'] = verdict
            rows.append(r)
            print(f"{lik:>11} {beta:6.2f} {r['au']:3d}/{M:<2d} {r['val_sse_per_sample']:10.2f} "
                  f"{r['val_kl']:9.3f} {verdict:>14}")
    out = {'_provenance': provenance(),
           '_reference': {'zero_prediction_sse': zero_sse, 'mean_image_sse': mean_sse},
           'runs': rows}
    with open(os.path.join(OUT, 'stage3_dsprites_likelihood.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {OUT}/stage3_dsprites_likelihood.json")


if __name__ == '__main__':
    main()
