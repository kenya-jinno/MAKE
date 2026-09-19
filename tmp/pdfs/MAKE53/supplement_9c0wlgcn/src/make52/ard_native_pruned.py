"""ARD-VAE: 選択軸だけを残した native モデルの品質評価。

Reviewer 2 No.3 への直接回答（MAKE52_revision_review.md §4.2）。
評価規則は `results/make52/native_pruned_spec.json` に**実行前に**登録した。

4 条件を分けて評価する:
  unpruned        : 全 L 軸、決定論的再構成
  pruned_masked   : 非選択軸を 0 に固定、復号器は再学習しない
                    （原著の「非関連軸は復号器出力をほとんど変えない」の直接検証）
  pruned_finetuned: pruned_masked から復号器のみ 30 epochs 微調整（予算は事前固定）
  transfer        : 選択幅に最も近い格子点で通常 VAE を新規学習（既存キャッシュ）

実行: プロジェクト直下で python3 src/make52/ard_native_pruned.py [--dataset MNIST]
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
from src.make52.geco_arm import provenance, SPECS, KW, eval_pass
from src.make52.ard_native import conjugate_sigma_hat, jacobian_weight_fast


def jacobian_weight_vjp(model, X, n_samples=100, batch=16, chunk=256):
    """原著 Eq.(19) の w_l = (1/N) sum_i ||J_i[:, l]||^2 を VJP の積み上げで計算する。

    jacrev は (B, D, L) を一度に確保するため、出力次元 D が大きいと GPU に載らない
    （dSprites は D=4096）。出力次元を chunk に分けて VJP を積み上げると、
    確保するのは (B, L) だけで済む。数値は jacrev 版と一致する。
    """
    model.eval()
    m = model.fc_mu.out_features
    acc = torch.zeros(m, device=DEVICE); cnt = 0
    for i in range(0, min(n_samples, len(X)), batch):
        xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
        with torch.no_grad():
            mu0, _ = model.encode(xb)
        mu = mu0.clone().requires_grad_(True)
        xh = model.decode(mu).flatten(1)
        D = xh.shape[1]
        for a in range(0, D, chunk):
            b = min(a + chunk, D)
            for k in range(a, b):
                g = torch.autograd.grad(xh[:, k].sum(), mu, retain_graph=True)[0]
                acc += (g ** 2).sum(0)
        cnt += xb.size(0)
        del xh, mu
        torch.cuda.empty_cache()
    return (acc / cnt).detach()

OUT = ROOT / 'results/make52'
FINETUNE_EPOCHS = 30          # 事前固定。結果を見て延長しない


def train_ard(data, spec, likelihood, m_start, seed, epochs, lr=1e-3,
              alpha_frac=.1, update_every=10):
    """ARD-VAE を学習し、モデルと sigma_hat を返す（原著 Eq.8,9,15）。"""
    set_seed(seed)
    model = build(spec, m_start).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = data['X_train']; n_a = max(256, int(len(X) * alpha_frac))
    Xa, Xs = X[:n_a], X[n_a:]
    Xs_t = torch.as_tensor(Xs); n = len(Xs_t)
    sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep % update_every == 1 or update_every == 1:
            sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=True)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xs_t[perm[i:i + 128]].to(DEVICE)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            pv = sigma_hat.unsqueeze(0)
            kl = 0.5 * ((lv.exp() + mu.pow(2)) / pv - 1 - lv + torch.log(pv)).sum(1).mean()
            (reconstruction_term(xh, xb, likelihood) + kl).backward()
            opt.step()
    sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=True)
    return model, sigma_hat, Xa, time.time() - t0


def eval_masked(model, X, likelihood, mask, batch=512, mc=0):
    """非選択軸を 0 に固定して評価する。"""
    model.eval(); n = len(X); mse = rec = kl = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.as_tensor(X[i:i + batch]).to(DEVICE); B = xb.size(0)
            mu, lv = model.encode(xb)
            if mc:
                acc = 0.0
                for _ in range(mc):
                    z = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)) * mask
                    acc = acc + F.mse_loss(model.decode(z), xb, reduction='sum')
                mse += (acc / mc).item() / xb[0].numel()
            else:
                mse += F.mse_loss(model.decode(mu * mask), xb, reduction='sum').item() / xb[0].numel()
            rec += reconstruction_term(model.decode(mu * mask), xb, likelihood).item() * B
            kl += (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()
    return {'mse': mse / n, 'rec': rec / n, 'kl': kl / n, 'elbo': -(rec / n + kl / n)}


def finetune_decoder(model, data, likelihood, mask, epochs=FINETUNE_EPOCHS, lr=1e-3):
    """符号化器と事前分散を固定し、復号器のみを微調整する。"""
    for p in model.enc.parameters(): p.requires_grad_(False)
    for p in model.fc_mu.parameters(): p.requires_grad_(False)
    for p in model.fc_lv.parameters(): p.requires_grad_(False)
    dec_params = [p for nme, p in model.named_parameters()
                  if p.requires_grad and ('dec' in nme or 'fc_dec' in nme)]
    opt = torch.optim.Adam(dec_params, lr=lr)
    Xt = torch.as_tensor(data['X_train']); n = len(Xt)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xt[perm[i:i + 128]].to(DEVICE)
            opt.zero_grad()
            with torch.no_grad():
                mu, lv = model.encode(xb)
            z = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)) * mask
            reconstruction_term(model.decode(z), xb, likelihood).backward()
            opt.step()
    return time.time() - t0


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

    prov = provenance(); prov['estimator'] = 'ARD-VAE 原著対応 + native 枝刈り評価'
    print(f"ARD-VAE native 枝刈り評価  {ds}  初期容量 {anchor}  微調整 {FINETUNE_EPOCHS} epochs\n")
    out = []
    for s in [int(x) for x in args.seeds.split(',')]:
        model, sigma_hat, Xa, tr_sec = train_ard(data, SPECS[ds], lik, anchor, s, args.epochs)
        # Jacobian は (B, D, L) を確保するので、出力次元の大きいデータでは
        # バッチを小さくしないと GPU メモリが足りない（CIFAR-10: 3072x256）
        if ds == 'dSprites':
            # D=4096 では jacrev の (B, D, L) が載らないため VJP 積み上げを用いる
            w = jacobian_weight_vjp(model, data['X_val'], n_samples=100, batch=8)
        else:
            jb = 25 if SPECS[ds]['kind'] == 'fc' else 4
            w = jacobian_weight_fast(model, data['X_val'], n_samples=100, batch=jb)
        score = (w * sigma_hat).cpu().numpy()
        order = np.argsort(score)[::-1]
        cum = np.cumsum(score[order]) / max(score.sum(), 1e-12)
        k = int(np.searchsorted(cum, 0.99) + 1)
        mask = torch.zeros(anchor, device=DEVICE); mask[order[:k].copy()] = 1.0

        r = {'dataset': ds, 'seed': s, 'm_start': anchor, 'selected_m': k,
             'train_seconds': tr_sec, 'provenance': prov,
             'unpruned': eval_pass(model, data['X_val'], lik),
             'unpruned_test': eval_pass(model, data['X_test'], lik),
             'pruned_masked': eval_masked(model, data['X_val'], lik, mask),
             'pruned_masked_test': eval_masked(model, data['X_test'], lik, mask),
             'pruned_masked_K32': eval_masked(model, data['X_val'], lik, mask, mc=32)}
        ft_sec = finetune_decoder(model, data, lik, mask)
        r['finetune_seconds'] = ft_sec
        r['pruned_finetuned'] = eval_masked(model, data['X_val'], lik, mask)
        r['pruned_finetuned_test'] = eval_masked(model, data['X_test'], lik, mask)
        m_grid = min(grid, key=lambda g: abs(g - k))
        t = cache.get_vae(int(m_grid), s)
        r['transfer'] = {'m_on_grid': int(m_grid), 'val': t['val'], 'test': t['test']}
        out.append(r)
        print(f"  seed {s:<6d} 選択 {k:3d}/{anchor}  "
              f"未枝刈り {r['unpruned']['mse']:.5f}  "
              f"枝刈り {r['pruned_masked']['mse']:.5f}  "
              f"微調整後 {r['pruned_finetuned']['mse']:.5f}  "
              f"転用({m_grid}) {t['val']['mse']:.5f}", flush=True)
        json.dump(out, open(OUT / f'ard_pruned_{ds}.json', 'w'), ensure_ascii=False)
    print(f"\n書き出し: {OUT}/ard_pruned_{ds}.json")


if __name__ == '__main__':
    main()
