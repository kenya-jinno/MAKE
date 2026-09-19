"""段階 3 校正の追加確認: 事前登録した beta 梯子が各データで使える範囲かを調べる。

校正実行で dSprites が beta=1 で完全に事後崩壊（AU=0, KL=0）したため、
梯子 {0.25, 0.5, 1, 2, 4} が各データで「学習が成立する範囲」を含むかを確認する。

**これは beta を都合よく選ぶための実験ではない。**
梯子が実行可能かどうかの feasibility 確認であり、結果は設定表 §9 に記録する。
短い学習（既定 60 epochs・1 seed）で崩壊の有無だけを見る。

実行: プロジェクト直下で python3 src/make49/stage3_beta_feasibility.py
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, DEVICE, provenance, set_seed, train_vae
from src.make49.datasets import LOADERS
from src.make49.models import build
from src.metrics.structure import active_units

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)
EPOCHS = 60
DELTA = CFG['au']['delta_primary']
LADDER = CFG['beta']['ladder']
EXTENDED = [0.05, 0.1] + LADDER          # 下方向に 2 点だけ足して範囲を確認する

CONFIGS = [
    {'name': 'dSprites_Conv', 'dataset': 'dSprites', 'm': 16,
     'spec': {'kind': 'conv', 'in_ch': 1, 'size': 64}, 'kw': {}},
    {'name': 'CIFAR10_Conv', 'dataset': 'CIFAR10', 'm': 64,
     'spec': {'kind': 'conv', 'in_ch': 3, 'size': 32}, 'kw': {}},
]


def main():
    print(f"beta 梯子の実行可能性확認 / 梯子={LADDER} / 追加確認点={EXTENDED[:2]}")
    print(f"学習 {EPOCHS} epochs・1 seed・early stopping なし（崩壊の有無のみ見る）\n")
    res = {'_provenance': provenance(), '_spec': {'epochs': EPOCHS, 'ladder': LADDER,
                                                  'extended': EXTENDED, 'delta': DELTA}}
    for c in CONFIGS:
        data = LOADERS[c['dataset']](**c['kw'])
        Xtr, Xva = data['X_train'], data['X_val']
        rows = []
        print(f"=== {c['name']} (m={c['m']}) ===")
        print(f"{'beta':>6} {'AU':>6} {'val_rec':>9} {'val_KL':>9} {'val_ELBO':>10} {'崩壊':>6}")
        for b in EXTENDED:
            set_seed(CFG['data_splits']['training_seeds'][0])
            model = build(c['spec'], c['m'])
            r = train_vae(model, Xtr, Xva, beta=b, max_epochs=EPOCHS,
                          patience=EPOCHS + 1, early_stopping=False)
            model = r['model']; model.eval()
            with torch.no_grad():
                mus = []
                for i in range(0, len(Xva), 512):
                    xb = torch.as_tensor(Xva[i:i + 512]).to(DEVICE)
                    mus.append(model(xb)[1].cpu())
                Z = torch.cat(mus).numpy()
            au = int(active_units(Z, DELTA))
            h = r['history']
            rec, kl, el = h['val_rec'][-1], h['val_kl'][-1], h['val_elbo'][-1]
            collapsed = (au == 0) or (kl < 1e-3)
            rows.append({'beta': b, 'au': au, 'val_rec': rec, 'val_kl': kl,
                         'val_elbo': el, 'collapsed': bool(collapsed),
                         'in_registered_ladder': b in LADDER})
            print(f"{b:6.2f} {au:3d}/{c['m']:<2d} {rec:9.3f} {kl:9.3f} {el:10.3f} "
                  f"{'はい' if collapsed else 'いいえ':>6}")
        res[c['name']] = rows
        usable = [r['beta'] for r in rows if not r['collapsed'] and r['in_registered_ladder']]
        print(f"  梯子内で崩壊しない beta: {usable if usable else 'なし'}\n")
    with open(os.path.join(OUT, 'stage3_beta_feasibility.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"書き出し: {OUT}/stage3_beta_feasibility.json")


if __name__ == '__main__':
    main()
