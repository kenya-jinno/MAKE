"""ゼロ中心 ARD の native 評価（MAKE53_revision_review.md §5.2）。

指摘: Table 10 にはゼロ中心と sample-centred の両設定があるが、
Table 11 の native マスク評価は sample-centred 変種だけである。
原著に沿った ARD の比較という意味で限定が残る。

本実験は §5.2 の範囲をそのまま実施する:
  - MNIST と dSprites の 2 データセット、3 seeds、**ゼロ中心設定**
  - Full と Mask を同じモデルで評価する（Adapt は必須としない。参考として併記）
  - 共通の外部 Q に加え、Full から Mask への対応のある MSE 変化を報告する

§5.3 に従い、checkpoint・選択軸・relevance・mask・epoch 履歴を保存する。

**原著と共通する仕組みと、実験上の変更の区別**:
  共通 : 共役更新（Eq.8,9 で mu^alpha = 0）、目的関数（Eq.15）、
         Jacobian による relevance（Eq.19）、累積 99% の選択規則
  変更 : データ量（train 18,000 例）、prior 更新用 pool（先頭 10%）、
         更新間隔 10 epochs、学習率 1e-3、Adam、300 epochs、
         観測尤度は本研究の設定（Gaussian / Bernoulli）

実行: プロジェクト直下で python3 src/make53/ard_zero_centred.py
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.make49.common import CFG, DEVICE, SPLIT_SEED
from src.make49.datasets import LOADERS
from src.make52.geco_arm import provenance, SPECS, KW, eval_pass
from src.make52.ard_native import conjugate_sigma_hat, jacobian_weight_fast
from src.make52.ard_native_pruned import (train_ard, eval_masked, finetune_decoder,
                                          jacobian_weight_vjp)
from src.make53 import artifacts as AR

OUT = ROOT / 'results/make53'; OUT.mkdir(exist_ok=True)
CODE = ['src/make52/ard_native.py', 'src/make52/ard_native_pruned.py',
        'src/make53/ard_zero_centred.py', 'src/make53/artifacts.py']


def train_ard_zero(data, spec, likelihood, m_start, seed, epochs, lr=1e-3,
                   alpha_frac=.1, update_every=10):
    """train_ard と同一手順だが、共役更新で mu^alpha = 0（ゼロ中心）とする。"""
    from src.make49.common import set_seed, reconstruction_term
    from src.make49.models import build
    set_seed(seed)
    model = build(spec, m_start).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = data['X_train']; n_a = max(256, int(len(X) * alpha_frac))
    Xa, Xs = X[:n_a], X[n_a:]
    Xs_t = torch.as_tensor(Xs); n = len(Xs_t)
    sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=False)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep % update_every == 1 or update_every == 1:
            sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=False)
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
        if ep % 10 == 0 or ep == epochs:
            hist.append({'epoch': ep, 'val_mse': eval_pass(model, data['X_val'], likelihood)['mse']})
    sigma_hat, _ = conjugate_sigma_hat(model, Xa, center=False)
    return model, sigma_hat, time.time() - t0, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='MNIST,dSprites')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--epochs', type=int, default=300)
    args = ap.parse_args()
    prov = provenance(); prov['estimator'] = 'ARD-VAE ゼロ中心（原著 Eq.8,9 で mu^alpha=0）'
    prov['code_fingerprint'] = AR.code_fingerprint(CODE)
    out, checks_all = [], []

    for ds in args.datasets.split(','):
        grid = CFG['candidate_grids'][ds]; anchor = grid[-1]
        lik = CFG['likelihood']['by_dataset'][ds]
        data = LOADERS[ds](**KW[ds])
        from src.make49.cache import CandidateCache
        rng = np.random.RandomState(SPLIT_SEED + 3)
        nn = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
        X_est = data['X_train'][rng.choice(len(data['X_train']), nn, replace=False)]
        cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=1.0)
        print(f"\n=== {ds}  ゼロ中心 ARD  初期容量 {anchor}  尤度 {lik} ===", flush=True)

        for s in [int(x) for x in args.seeds.split(',')]:
            model, sigma_hat, tr_sec, hist = train_ard_zero(
                data, SPECS[ds], lik, anchor, s, args.epochs)
            jfn = jacobian_weight_vjp if ds == 'dSprites' else jacobian_weight_fast
            jb = 8 if ds == 'dSprites' else 25
            w = jfn(model, data['X_val'], n_samples=100, batch=jb)
            score = (w * sigma_hat).cpu().numpy()
            order = np.argsort(score)[::-1]
            cum = np.cumsum(score[order]) / max(score.sum(), 1e-12)
            k = int(np.searchsorted(cum, 0.99) + 1)
            sel_ids = order[:k].copy()
            mask_t = torch.zeros(anchor, device=DEVICE); mask_t[sel_ids] = 1.0
            mask = mask_t.cpu().numpy().astype(int)

            state_full = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            ev = {'full': eval_pass(model, data['X_val'], lik),
                  'full_test': eval_pass(model, data['X_test'], lik),
                  'mask': eval_masked(model, data['X_val'], lik, mask_t),
                  'mask_test': eval_masked(model, data['X_test'], lik, mask_t)}
            # 機能確認（§5.3）
            ck = [AR.check_mask_count(k, mask),
                  AR.check_full_mask_identity(model, data['X_val'], eval_masked,
                                              eval_pass, lik, anchor)]
            ft_sec = finetune_decoder(model, data, lik, mask_t)
            state_adapt = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            ck.append(AR.check_encoder_unchanged(state_full, state_adapt))
            ev['adapt'] = eval_masked(model, data['X_val'], lik, mask_t)
            ev['adapt_test'] = eval_masked(model, data['X_test'], lik, mask_t)

            m_grid = min(grid, key=lambda g: abs(g - k))
            t = cache.get_vae(int(m_grid), s)
            ev['transfer'] = {'m_on_grid': int(m_grid), 'val': t['val'], 'test': t['test']}

            tag = f'ard_zero_{ds}_s{s}'
            path = AR.save_run(tag, state_full, state_adapt, SPECS[ds], score,
                               sigma_hat.cpu().numpy(), order, sel_ids, mask,
                               {'dataset': ds, 'seed': s, 'centering': 'zero (mu^alpha=0)',
                                'epochs': args.epochs, 'finetune_epochs': 30,
                                'train_seconds': tr_sec, 'finetune_seconds': ft_sec,
                                'split_index_hash': data['split']['index_hash'],
                                'val_history': hist, 'provenance': prov}, ev)
            r = {'dataset': ds, 'seed': s, 'selected_m': k, 'm_start': anchor,
                 'artifact_dir': path, 'checks': ck, **ev,
                 'train_seconds': tr_sec, 'finetune_seconds': ft_sec}
            out.append(r); checks_all += [dict(c, tag=tag) for c in ck]
            print(f"  seed {s:<6d} 選択 {k:3d}/{anchor}  Full {ev['full']['mse']:.5f}  "
                  f"Mask {ev['mask']['mse']:.5f}  Adapt {ev['adapt']['mse']:.5f}  "
                  f"転用({m_grid}) {t['val']['mse']:.5f}  確認 "
                  f"{sum(c['pass'] for c in ck)}/{len(ck)}", flush=True)
            json.dump(out, open(OUT / 'ard_zero_centred.json', 'w'), ensure_ascii=False)

    print()
    AR.report(checks_all, OUT / 'functional_checks.json')
    print(f"\n書き出し: {OUT}/ard_zero_centred.json, {OUT}/functional_checks.json")
    print(f"成果物: results/make53/checkpoints/ 配下（model state・選択軸・relevance・mask・履歴）")


if __name__ == '__main__':
    main()
