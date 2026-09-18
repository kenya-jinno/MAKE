"""段階 3 校正の仕上げ: dSprites を Bernoulli 尤度にしたとき、
beta 梯子が「遮断の効く範囲」を含むかを確認する。

尤度診断で Bernoulli にすると崩壊が解消することが分かったが、
beta=1 で AU=14/16、beta=0.25 で 16/16 と、梯子の下側では遮断が起きない。
梯子 {0.25,0.5,1,2,4} の上側で AU < m になるかを確認する。

実行: プロジェクト直下で python3 src/make49/stage3_dsprites_bce_ladder.py
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
EPOCHS, M, DELTA = 60, 16, CFG['au']['delta_primary']
LADDER = CFG['beta']['ladder']
EXTENDED = LADDER + [8.0, 16.0]        # 上方向に 2 点だけ足して範囲を確認する


def run(beta, Xtr, Xva):
    set_seed(CFG['data_splits']['training_seeds'][0])
    model = build({'kind': 'conv', 'in_ch': 1, 'size': 64}, M).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.as_tensor(Xtr); n = len(Xtr_t)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xtr_t[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            rec = F.binary_cross_entropy(xh.clamp(1e-6, 1 - 1e-6), xb, reduction='sum') / B
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
            (rec + beta * kl).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sse, kls, mus = 0.0, 0.0, []
        Xva_t = torch.as_tensor(Xva)
        for i in range(0, len(Xva_t), 512):
            xb = Xva_t[i:i + 512].to(DEVICE); B = xb.size(0)
            xh, mu, lv = model(xb)
            sse += F.mse_loss(xh, xb, reduction='sum').item()
            kls += (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()
            mus.append(mu.cpu())
        N = len(Xva_t)
        Z = torch.cat(mus).numpy()
    return {'beta': beta, 'au': int(active_units(Z, DELTA)),
            'val_sse_per_sample': sse / N, 'val_kl': kls / N,
            'in_registered_ladder': beta in LADDER}


def main():
    d = load_dsprites(); Xtr, Xva = d['X_train'], d['X_val']
    print(f"dSprites + Bernoulli 尤度、m={M}, {EPOCHS} epochs, seed 42")
    print(f"梯子={LADDER} / 追加確認点={EXTENDED[len(LADDER):]}\n")
    print(f"{'beta':>6} {'AU':>7} {'val SSE':>9} {'val KL':>9} {'遮断':>6} {'梯子内':>7}")
    rows = []
    for b in EXTENDED:
        r = run(b, Xtr, Xva)
        pruned = r['au'] < M
        r['pruning'] = bool(pruned)
        rows.append(r)
        print(f"{b:6.2f} {r['au']:3d}/{M:<3d} {r['val_sse_per_sample']:9.3f} {r['val_kl']:9.3f} "
              f"{'あり' if pruned else 'なし':>6} {str(r['in_registered_ladder']):>7}")
    ok = [r['beta'] for r in rows if r['pruning'] and r['in_registered_ladder']]
    print(f"\n梯子内で遮断が起きる beta: {ok if ok else 'なし'}")
    with open(os.path.join(OUT, 'stage3_dsprites_bce_ladder.json'), 'w') as f:
        json.dump({'_provenance': provenance(), '_spec': {'epochs': EPOCHS, 'm': M,
                   'ladder': LADDER, 'extended': EXTENDED, 'delta': DELTA,
                   'likelihood': 'bernoulli'}, 'runs': rows}, f, ensure_ascii=False, indent=1)
    print(f"書き出し: {OUT}/stage3_dsprites_bce_ladder.json")


if __name__ == '__main__':
    main()
