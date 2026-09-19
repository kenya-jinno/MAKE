"""GECO + L0-ARM: 原著（De Boom et al., arXiv:2003.10901）に対応する実装。

MAKE51 残作業 §3 への対応。既存の実装は hard concrete 緩和を用いており、
原著の L0-ARM 勾配推定量とは異なる旨を逸脱として明記していた。
本実装は原著の推定量そのものを用いる。

原著 Eq.(12) の ARM 推定量:

    grad_gamma L ~= E_u[ ( F(1[u > sigma(-gamma)]) - F(1[u < sigma(gamma)]) )
                         * (u - 1/2) ]  +  beta * grad_gamma sum_j sigma(gamma_j)

u ~ Uniform(0,1)^n を 1 組引き、2 つの対蹠的なゲート標本で順伝播を 2 回行う。
F は「ゲートを与えたときの損失」であり、本実装では原著 Eq.(13) の
KL + lambda * C をゲート依存部分として用いる。

原著 Algorithm 1 の要点はそのまま保つ:
  - KL はゲートでマスクする
  - 制約 C は移動平均で近似し、勾配は現ステップの C から流す
  - lambda は二乗 softplus で正に保ち [lam_min, lam_max] にクランプ
  - **L0 項は制約充足時のみ加える**
  - 推論時は sigma(gamma) <= 0.5 のゲートを閉じる
  - 制約は画素についての**和**（Eq.8）

あわせて MAKE51 残作業 §4.2・§4.3 に対応する量を保存する:
  - epoch ごとの train / validation 曲線（train も評価モード）
  - 真の ELBO（K サンプルの重点サンプリング推定ではなく、beta=1 の ELBO）
  - 参照 AE 学習・候補学習・評価の個別時間（CPU と GPU を混ぜない）

実行: プロジェクト直下で python3 src/make52/geco_arm.py [--dataset MNIST] [--seeds 42]
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.make49.common import CFG, DEVICE, set_seed, reconstruction_term
from src.make49.datasets import LOADERS
from src.make49.models import build

OUT = ROOT / 'results/make52'
OUT.mkdir(exist_ok=True)
SPECS = {'MNIST': {'kind': 'fc', 'input_dim': 784},
         'FashionMNIST': {'kind': 'fc', 'input_dim': 784},
         'dSprites': {'kind': 'conv', 'in_ch': 1, 'size': 64},
         'CIFAR10': {'kind': 'conv', 'in_ch': 3, 'size': 32}}
KW = {'MNIST': {'flatten': True}, 'FashionMNIST': {'flatten': True},
      'dSprites': {}, 'CIFAR10': {}}


def provenance():
    try:
        drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        drv = None
    return {'python': sys.version.split()[0], 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'driver': drv, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'platform': platform.platform(), 'estimator': 'L0-ARM (原著 Eq.12)'}


def eval_pass(model, X, likelihood, gates=None, batch=512, mc=0):
    """評価モード。mc>0 なら事後標本による再構成、それ以外は決定論的 g(mu)。"""
    model.eval()
    n = len(X); mse = rec = kl = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.as_tensor(X[i:i + batch]).to(DEVICE); B = xb.size(0)
            mu, lv = model.encode(xb)
            if mc:
                acc = 0.0
                for _ in range(mc):
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                    if gates is not None: z = z * gates
                    acc = acc + F.mse_loss(model.decode(z), xb, reduction='sum')
                mse += (acc / mc).item() / xb[0].numel()
            else:
                z = mu * gates if gates is not None else mu
                xh = model.decode(z)
                mse += F.mse_loss(xh, xb, reduction='sum').item() / xb[0].numel()
            z0 = mu * gates if gates is not None else mu
            rec += reconstruction_term(model.decode(z0), xb, likelihood).item() * B
            kl += (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()
    return {'mse': mse / n, 'rec': rec / n, 'kl': kl / n,
            'elbo': -(rec / n + kl / n)}      # beta=1 の真の ELBO


def run_arm(data, spec, likelihood, m_start, seed, tau, epochs=300,
            alpha_ma=.99, lam_min=1e-6, lam_max=1e6, lr=1e-3, l0_w=1.0):
    set_seed(seed)
    model = build(spec, m_start).to(DEVICE)
    gamma = torch.zeros(m_start, device=DEVICE, requires_grad=True)   # sigma(0)=0.5 から開始
    lam_raw = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam(list(model.parameters()) + [gamma, lam_raw], lr=lr)
    Xtr = torch.as_tensor(data['X_train']); n = len(Xtr)
    D = int(np.prod(data['X_train'].shape[1:]))
    tau_sum = tau * D                     # 原著 Eq.(8) は画素についての和
    C_ma = None
    curves = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_elbo': [],
              'open_gates': [], 'constraint_ma': [], 'lambda': []}
    t0 = time.time(); steps = 0

    def gated_loss(xb, g):
        """ゲート g を与えたときのゲート依存損失（原著 Eq.13 の KL + lambda*C）。

        ARM 推定量は**標本ごとに異なる F の値**を必要とするため、
        バッチ平均する前の per-sample の量も返す。
        平均してから渡すと標本間の差が消え、推定量が機能しない
        （src/make52/test_arm_estimator.py の対照を参照）。
        """
        mu, lv = model.encode(xb)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        xh = model.decode(z * g)
        rec_sum = F.mse_loss(xh, xb, reduction='none').flatten(1).sum(1)   # per-sample
        C_i = rec_sum - tau_sum                                            # per-sample
        kl_elem = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        kl_i = (kl_elem * g).sum(1)                                        # per-sample
        return kl_i.mean(), C_i.mean(), kl_i, C_i

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xtr[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            sg = torch.sigmoid(gamma)
            # ARM: 一様乱数を 1 組引き、対蹠的な 2 つの二値ゲートで 2 回順伝播する
            u = torch.rand(B, m_start, device=DEVICE)
            g1 = (u < sg.unsqueeze(0)).float()          # 1[u < sigma(gamma)]
            g2 = (u > torch.sigmoid(-gamma).unsqueeze(0)).float()   # 1[u > sigma(-gamma)]
            kl1, C1, kl1_i, C1_i = gated_loss(xb, g1)
            kl2, C2, kl2_i, C2_i = gated_loss(xb, g2)
            lam = torch.clamp(F.softplus(lam_raw) ** 2, lam_min, lam_max)
            C_cur = 0.5 * (C1 + C2)
            C_ma = C_cur.detach() if C_ma is None else alpha_ma * C_ma + (1 - alpha_ma) * C_cur.detach()
            C_eff = C_ma + (C_cur - C_cur.detach())     # 値は移動平均、勾配は現ステップ
            hit = C_ma.item() <= 0

            # モデル側（theta, phi）: g1 の経路を通常どおり最小化する
            loss_model = kl1 + lam.detach() * C_eff
            # lambda 側: min-max のため符号を反転
            loss_lambda = -lam * C_ma.detach()

            # gamma 側: ARM 推定量（原著 Eq.12）。F の差に (u - 1/2) を掛ける
            #
            # 重要: F は**ゲート標本ごとに異なる値**でなければ ARM が機能しない。
            # C1/C2 は既にバッチ平均されているため、標本ごとの差が平均で潰れる。
            # 標本ごとの制約違反を用いる。
            with torch.no_grad():
                F1_i = (kl1_i + lam * C1_i).detach()      # per-sample（平均しない）
                F2_i = (kl2_i + lam * C2_i).detach()
                arm = ((F2_i - F1_i).unsqueeze(1) * (u - 0.5)).mean(0)
            surrogate_gamma = (arm * gamma).sum()
            if hit:                                      # 制約充足時のみ L0 を加える
                surrogate_gamma = surrogate_gamma + l0_w * torch.sigmoid(gamma).sum()

            (loss_model + loss_lambda + surrogate_gamma).backward()
            opt.step(); steps += 1

        if ep % 5 == 0 or ep == epochs:
            with torch.no_grad():
                g_inf = (torch.sigmoid(gamma) > 0.5).float()
            tr = eval_pass(model, data['X_train'][:2000], likelihood, g_inf)
            va = eval_pass(model, data['X_val'], likelihood, g_inf)
            curves['epoch'].append(ep); curves['train_mse'].append(tr['mse'])
            curves['val_mse'].append(va['mse']); curves['val_elbo'].append(va['elbo'])
            curves['open_gates'].append(int(g_inf.sum().item()))
            curves['constraint_ma'].append(float(C_ma.item()))
            curves['lambda'].append(float(torch.clamp(F.softplus(lam_raw) ** 2,
                                                      lam_min, lam_max).item()))
    train_seconds = time.time() - t0

    with torch.no_grad():
        sg = torch.sigmoid(gamma)
        mask = (sg > 0.5).float()
    t1 = time.time()
    res = {'selected_m': int(mask.sum().item()), 'm_start': m_start, 'seed': seed,
           'tau_per_pixel': tau, 'tau_sum': tau_sum,
           'train_seconds': train_seconds, 'update_steps': steps,
           'final_constraint_ma': float(C_ma.item()),
           'constraint_satisfied': bool(C_ma.item() <= 0),
           'gate_sigmoid': [float(x) for x in sg.cpu().numpy()],
           'curves': curves,
           'eval_mean': eval_pass(model, data['X_val'], likelihood, mask),
           'eval_test_mean': eval_pass(model, data['X_test'], likelihood, mask),
           'eval_sampled_K32': eval_pass(model, data['X_val'], likelihood, mask, mc=32)}
    res['eval_seconds'] = time.time() - t1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MNIST')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--epochs', type=int, default=300)
    args = ap.parse_args()
    ds = args.dataset
    seeds = [int(x) for x in args.seeds.split(',')]
    grid = CFG['candidate_grids'][ds]; anchor = grid[-1]
    lik = CFG['likelihood']['by_dataset'][ds]
    data = LOADERS[ds](**KW[ds])

    from src.make49.cache import CandidateCache
    from src.make49.common import SPLIT_SEED
    rng = np.random.RandomState(SPLIT_SEED + 3)
    nn = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
    X_est = data['X_train'][rng.choice(len(data['X_train']), nn, replace=False)]
    cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=1.0)

    prov = provenance()
    print(f"GECO + L0-ARM（原著推定量）  {ds}  尤度 {lik}  初期容量 {anchor}")
    print(f"  {prov['estimator']} / {prov['gpu']} / torch {prov['torch']}\n")
    out = []
    for s in seeds:
        a = cache.get_vae(anchor, s)
        tau = a['val']['mse'] * (1 + CFG['quality_criterion_Q']['epsilon_D']['rel_primary'])
        r = run_arm(data, SPECS[ds], lik, anchor, s, tau, epochs=args.epochs)
        # 選択された次元を通常の VAE へ転用した結果（native と分けて報告する）
        m_grid = min(grid, key=lambda g: abs(g - r['selected_m'])) if r['selected_m'] >= 1 else None
        if m_grid:
            t = cache.get_vae(int(m_grid), s)
            r['transferred'] = {'m_on_grid': int(m_grid), 'val': t['val'], 'test': t['test'],
                                'candidate_seconds': t['elapsed_seconds']}
        r['dataset'] = ds; r['provenance'] = prov
        out.append(r)
        print(f"  seed {s:<6d} 幅 {r['selected_m']:3d}/{anchor}  "
              f"制約充足 {r['constraint_satisfied']}  C_ma {r['final_constraint_ma']:+8.2f}  "
              f"val MSE {r['eval_mean']['mse']:.5f}  ELBO {r['eval_mean']['elbo']:9.3f}  "
              f"{r['train_seconds']:.0f}s", flush=True)
        json.dump(out, open(OUT / f'geco_arm_{ds}.json', 'w'), ensure_ascii=False)
    print(f"\n書き出し: {OUT}/geco_arm_{ds}.json")


if __name__ == '__main__':
    main()
