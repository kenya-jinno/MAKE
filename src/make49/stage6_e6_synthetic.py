"""段階 6 / E6: 人工条件での切り分け（方針 §5.5 E6）。

**大規模な新理論は開発しない。原稿の不正確な主張を除くことを優先する。**

段階 1 の A4 で判明した問題を修正して Table 4 相当を再計算する:
  - 合成多様体データが**未標準化**で、座標スケールが著しく異方的だった
    （Swiss Roll の座標別 std [6.57, 0.29, 6.99]、全分散 31.4 対 Torus 3.3）
  - そのため MSE を多様体間で比較できず、AU の絶対閾値 δ もスケール依存だった

修正: **標準化してから学習・評価する。** 損失規約は段階 1 の A1 修正版を使う。

確認する主張:
  (a) トーラスで AU が低くても再構成が不十分な例があるか（AU 合格 ≠ 品質達成）
  (b) AU 未満の座標を無効化すると再構成が変わるか
      （**小さい posterior-mean 分散だけでは座標が無情報とはいえない**）

実行: プロジェクト直下で python3 src/make49/stage6_e6_synthetic.py
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, DEVICE, provenance, set_seed
from src.make49.models import FCVAE
from src.data.synthetic import generate_swiss_roll, generate_torus

OUT = os.path.join('results', 'make49')
os.makedirs(OUT, exist_ok=True)
SEEDS = CFG['data_splits']['training_seeds'][:3]
M_LIST = [1, 2, 3, 4, 6, 8]
EPOCHS, BETA, DELTA = 300, CFG['beta']['primary'], CFG['au']['delta_primary']


def make(name, n=5000, seed=0):
    if name == 'SwissRoll':
        X, d, _ = generate_swiss_roll(n, noise=0.01, seed=seed)
    else:
        X, d, _ = generate_torus(n, R=3.0, r=1.0, noise=0.01, seed=seed)
    return X.astype(np.float32), d


def train(X, m, seed, standardize):
    set_seed(seed)
    mu_, sd_ = X.mean(0), X.std(0) + 1e-8
    Xs = ((X - mu_) / sd_ if standardize else X).astype(np.float32)
    n = len(Xs)
    ntr = int(n * 0.9)
    Xtr, Xva = torch.as_tensor(Xs[:ntr]), torch.as_tensor(Xs[ntr:]).to(DEVICE)
    model = FCVAE(Xs.shape[1], m, hidden=(128, 64)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            xb = Xtr[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            rec = F.mse_loss(xh, xb, reduction='sum') / B
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
            (rec + BETA * kl).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        mu, lv = model.encode(Xva)
        xh = model.decode(mu)
        mse = F.mse_loss(xh, Xva).item()
        Z = mu.cpu().numpy()
        v = Z.var(0)
        au = int((v > DELTA).sum())
        # (b) AU 未満の座標を無効化して再構成が変わるか
        mu2 = mu.clone()
        for j in np.where(v <= DELTA)[0]:
            mu2[:, j] = 0.0
        mse_ablate = F.mse_loss(model.decode(mu2), Xva).item()
    return {'m': m, 'seed': seed, 'standardized': standardize,
            'mse': mse, 'au': au, 'mse_after_ablation': mse_ablate,
            'ablation_delta': mse_ablate - mse,
            'mu_var_sorted': [float(x) for x in np.sort(v)[::-1]]}


def main():
    print("E6: 合成多様体の再確認（A4 の標準化修正後、損失は A1 修正版、β=1、3 seeds）\n")
    rows = []
    for name in ['SwissRoll', 'Torus']:
        X, d_true = make(name)
        print(f"=== {name}（真の次元 {d_true}、埋め込み次元 3）===")
        print(f"  未標準化: 座標別 std {np.round(X.std(0), 3)}  全分散 {X.var():.2f}")
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        print(f"  標準化後: 座標別 std {np.round(Xs.std(0), 3)}  全分散 {Xs.var():.2f}")
        print(f"\n  {'m':>3s} {'MSE(標準化)':>12s} {'AU':>4s} {'無効化後 MSE':>13s} {'差':>10s}")
        for m in M_LIST:
            rs = [train(X, m, s, True) for s in SEEDS]
            mse = np.mean([r['mse'] for r in rs]); au = np.mean([r['au'] for r in rs])
            ab = np.mean([r['mse_after_ablation'] for r in rs])
            dl = np.mean([r['ablation_delta'] for r in rs])
            for r in rs:
                r['manifold'] = name; r['d_true'] = d_true
            rows += rs
            print(f"  {m:3d} {mse:12.5f} {au:4.1f} {ab:13.5f} {dl:+10.5f}")
        # 未標準化との比較（原稿 Table 4 の条件）
        print(f"\n  --- 参考: 未標準化（原稿 Table 4 の条件）---")
        print(f"  {'m':>3s} {'MSE(未標準化)':>14s} {'AU':>4s}")
        for m in M_LIST:
            rs = [train(X, m, s, False) for s in SEEDS]
            for r in rs:
                r['manifold'] = name; r['d_true'] = d_true
            rows += rs
            print(f"  {m:3d} {np.mean([r['mse'] for r in rs]):14.5f} "
                  f"{np.mean([r['au'] for r in rs]):4.1f}")
        print()
    with open(os.path.join(OUT, 'stage6_e6_synthetic.json'), 'w') as f:
        json.dump({'_provenance': provenance(),
                   '_spec': {'seeds': SEEDS, 'm_list': M_LIST, 'epochs': EPOCHS,
                             'beta': BETA, 'delta': DELTA, 'n': 5000},
                   'runs': rows}, f, ensure_ascii=False, indent=1)
    print(f"書き出し: {OUT}/stage6_e6_synthetic.json")


if __name__ == '__main__':
    main()
