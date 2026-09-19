"""ARD-VAE: 原著（Saha et al., arXiv:2501.10901）に対応する実装。

MAKE51 残作業 §3 への対応。既存実装では (a) 事前更新と (b) 関連度重みの
Jacobian 評価が原著と対応するか未検証だった。本実装は原著の式をそのまま用いる。

原著の式:

  共役更新（Eq. 8, 9）        a_l = a_l^0 + n/2
                              b_l = b_l^0 + sum_i (z_l^i - mu_l^alpha)^2 / 2
                              推定分散 sigma_hat^2 = b_L / a_L

  目的関数（Eq. 15）          KL = -L/2 - (1/2) sum log sigma_i^2
                                   + (1/2) sum log sigma_hat_i^2
                                   + (1/2) sum (mu_i^2 + sigma_i^2)/sigma_hat_i^2
                              = KL( N(mu, sigma^2) || N(0, sigma_hat^2) )

  関連度重み（Eq. 19）        w_l = (1/N) sum_i || J_i[:, l] ||^2
                              J = [ d xhat / d mu_1 ... d xhat / d mu_L ] in R^{D x L}

  関連軸                      w ⊙ sigma_hat^2 の累積 99%

**確認できた事項**: 原著は alpha を周辺化した Student's t を動機づけに用いるが、
最適化する目的関数（Eq. 15）は **N(0, sigma_hat^2) の Gaussian 近似そのもの**である。
したがって従来「Student's t ではなく Gaussian 近似を用いた逸脱」と記していたのは
**過剰な申告**であり、本実装は目的関数に関して原著と一致する。

**残る差**: 原著は各軸の平均 mu_l^alpha を引いてから b_l を計算する（Eq. 9）。
無情報ハイパー事前（a^0 = b^0 = 0）かつ mu^alpha = 0 とすると
sigma_hat^2 = E[z^2] に帰着する。本実装は両方を実装して比較する。

実行: プロジェクト直下で python3 src/make52/ard_native.py [--dataset MNIST]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.make49.common import CFG, DEVICE, set_seed, reconstruction_term, SPLIT_SEED
from src.make49.datasets import LOADERS
from src.make49.models import build
from src.make52.geco_arm import provenance, SPECS, KW

OUT = ROOT / 'results/make52'; OUT.mkdir(exist_ok=True)


def conjugate_sigma_hat(model, Xa, center, batch=512):
    """原著 Eq.(8)(9) の共役更新。center=True なら軸平均を引く。"""
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(Xa), batch):
            xb = torch.as_tensor(Xa[i:i + batch]).to(DEVICE)
            mu, lv = model.encode(xb)
            zs.append((mu + torch.randn_like(mu) * torch.exp(0.5 * lv)).cpu())
    z = torch.cat(zs)
    n = len(z)
    mu_alpha = z.mean(0) if center else torch.zeros(z.shape[1])
    a_L = n / 2.0                                    # a^0 = 0
    b_L = ((z - mu_alpha) ** 2).sum(0) / 2.0         # b^0 = 0
    return torch.clamp(b_L / a_L, 1e-8, 1e8).to(DEVICE), float(a_L)


def jacobian_weight(model, X, n_samples=100, batch=25):
    """原著 Eq.(19): w_l = (1/N) sum_i || d xhat_i / d mu_l ||^2（自動微分で厳密に）。"""
    model.eval()
    m = model.fc_mu.out_features
    acc = torch.zeros(m, device=DEVICE); cnt = 0
    for i in range(0, min(n_samples, len(X)), batch):
        xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
        with torch.no_grad():
            mu0, _ = model.encode(xb)
        mu = mu0.clone().requires_grad_(True)
        xh = model.decode(mu)
        D = xh[0].numel()
        # 各出力次元ごとの勾配を足すのは高価なので、
        # ||J||_F^2 = sum_k sum_l (d xhat_k / d mu_l)^2 を
        # ランダム射影ではなく出力次元の走査で厳密に求める
        for k in range(D):
            g = torch.autograd.grad(xh.flatten(1)[:, k].sum(), mu, retain_graph=(k < D - 1))[0]
            acc += (g ** 2).sum(0)
        cnt += xb.size(0)
    return (acc / cnt).detach()


def jacobian_weight_fast(model, X, n_samples=100, batch=25):
    """出力次元が大きい場合の同値計算。

    ||J[:, l]||^2 = sum_k (d xhat_k / d mu_l)^2 は、
    xhat の各成分の mu についての勾配を全部集める必要があるが、
    vmap 付きの jacrev で一括計算する。
    """
    from torch.func import jacrev, vmap
    model.eval()
    outs = []
    for i in range(0, min(n_samples, len(X)), batch):
        xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
        with torch.no_grad():
            mu0, _ = model.encode(xb)

        def dec(m1):
            return model.decode(m1.unsqueeze(0)).flatten()

        J = vmap(jacrev(dec))(mu0)          # (B, D, L)
        outs.append((J ** 2).sum(1).sum(0).detach())   # 軸ごとに ||.||^2 を集計
    return torch.stack(outs).sum(0) / min(n_samples, len(X))


def run_ard(data, spec, likelihood, m_start, seed, epochs=300, lr=1e-3,
            alpha_frac=.1, update_every=10, center=True, cum_var=.99):
    set_seed(seed)
    model = build(spec, m_start).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = data['X_train']
    n_a = max(256, int(len(X) * alpha_frac))
    Xa, Xs = X[:n_a], X[n_a:]
    Xs_t = torch.as_tensor(Xs); n = len(Xs_t)
    sigma_hat, a_L = conjugate_sigma_hat(model, Xa, center)
    curves = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_elbo': []}
    t0 = time.time(); steps = 0
    for ep in range(1, epochs + 1):
        if ep % update_every == 1 or update_every == 1:
            sigma_hat, a_L = conjugate_sigma_hat(model, Xa, center)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xs_t[perm[i:i + 128]].to(DEVICE)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            rec = reconstruction_term(xh, xb, likelihood)
            pv = sigma_hat.unsqueeze(0)
            # 原著 Eq.(15)
            kl = 0.5 * ((lv.exp() + mu.pow(2)) / pv - 1 - lv + torch.log(pv)).sum(1).mean()
            (rec + kl).backward(); opt.step(); steps += 1
        if ep % 5 == 0 or ep == epochs:
            from src.make52.geco_arm import eval_pass
            tr = eval_pass(model, data['X_train'][:2000], likelihood)
            va = eval_pass(model, data['X_val'], likelihood)
            curves['epoch'].append(ep); curves['train_mse'].append(tr['mse'])
            curves['val_mse'].append(va['mse']); curves['val_elbo'].append(va['elbo'])
    train_seconds = time.time() - t0

    sigma_hat, a_L = conjugate_sigma_hat(model, Xa, center)
    t1 = time.time()
    try:
        w = jacobian_weight_fast(model, data['X_val'], n_samples=100)
        wmode = 'exact Jacobian (torch.func.jacrev)'
    except Exception as e:
        w = jacobian_weight(model, data['X_val'], n_samples=100)
        wmode = f'exact Jacobian (loop over outputs); jacrev failed: {type(e).__name__}'
    jac_seconds = time.time() - t1

    score = (w * sigma_hat).cpu().numpy()
    order = np.argsort(score)[::-1]
    cum = np.cumsum(score[order]) / max(score.sum(), 1e-12)
    sel = int(np.searchsorted(cum, cum_var) + 1)

    from src.make52.geco_arm import eval_pass
    return {'selected_m': sel, 'm_start': m_start, 'seed': seed, 'center_mu_alpha': center,
            'weight_mode': wmode, 'a_L': a_L,
            'train_seconds': train_seconds, 'jacobian_seconds': jac_seconds,
            'update_steps': steps, 'curves': curves,
            'sigma_hat': [float(x) for x in sigma_hat.cpu().numpy()],
            'jacobian_weight': [float(x) for x in w.cpu().numpy()],
            'relevance_score': [float(x) for x in score],
            'eval_full': eval_pass(model, data['X_val'], likelihood),
            'eval_test_full': eval_pass(model, data['X_test'], likelihood)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MNIST')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--epochs', type=int, default=300)
    args = ap.parse_args()
    ds = args.dataset
    grid = CFG['candidate_grids'][ds]; anchor = grid[-1]
    lik = CFG['likelihood']['by_dataset'][ds]
    data = LOADERS[ds](**KW[ds])
    from src.make49.cache import CandidateCache
    rng = np.random.RandomState(SPLIT_SEED + 3)
    nn = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
    X_est = data['X_train'][rng.choice(len(data['X_train']), nn, replace=False)]
    cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=1.0)

    prov = provenance(); prov['estimator'] = 'ARD-VAE 原著対応（Eq.8,9,15,19）'
    print(f"ARD-VAE（原著対応）  {ds}  尤度 {lik}  初期容量 {anchor}\n")
    out = []
    for s in [int(x) for x in args.seeds.split(',')]:
        for center in [True, False]:
            r = run_ard(data, SPECS[ds], lik, anchor, s, epochs=args.epochs, center=center)
            m_grid = min(grid, key=lambda g: abs(g - r['selected_m']))
            t = cache.get_vae(int(m_grid), s)
            r['transferred'] = {'m_on_grid': int(m_grid), 'val': t['val'], 'test': t['test']}
            r['dataset'] = ds; r['provenance'] = prov
            out.append(r)
            print(f"  seed {s:<6d} 中心化 {str(center):5s}  幅 {r['selected_m']:3d}/{anchor}  "
                  f"転用 {m_grid:3d}  val MSE {r['eval_full']['mse']:.5f}  "
                  f"Jacobian {r['jacobian_seconds']:.0f}s  学習 {r['train_seconds']:.0f}s", flush=True)
            json.dump(out, open(OUT / f'ard_native_{ds}.json', 'w'), ensure_ascii=False)
    print(f"\n書き出し: {OUT}/ard_native_{ds}.json")


if __name__ == '__main__':
    main()
