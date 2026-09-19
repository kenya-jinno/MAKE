"""B7（GECO + L0）と B8（ARD-VAE）の実装。

いずれも**学習中に次元数を決める**手法なので、候補グリッドを走査しない。
大きい初期容量から始めて 1 回学習し、残った次元数を返す。

## B7: GECO + L0（De Boom et al., arXiv:2003.10901）

原著 Eq.(13):
    L(x, λ) = KL(q(z|x) || p(z)) + ||ν||_0 + λ · E_z[C(x, g(z ⊙ ν))]
    C = ||x - g(z⊙ν)||^2 - τ            （τ は再構成誤差の上限）

原著の実装上の要点（Algorithm 1 と本文）:
  - KL は**ゲートでマスクする**（剪定済み次元は損失に効かない）
  - 制約 C は**移動平均**で期待値を近似し、勾配は現ステップの C のみから流す
  - λ には二乗 softplus をかけて正に保ち、[λ_min, λ_max] にクランプする
  - **L0 項は制約が満たされたとき（C ≤ 0）にのみ加える**
  - 推論時は σ(γ_j) ≤ 0.5 のゲートを閉じる。開いたゲート数が選択次元

**原著からの逸脱（明記する）**: 原著は L0-ARM 勾配推定器を使うが、
本実装は **hard concrete 緩和**（Louizos et al. 2018）を用いる。
原著本文も hard concrete を同じ L0 目的に対する代替推定器として挙げている。
目的関数は同一であり、推定器のみが異なる。

## B8: ARD-VAE（Saha et al., arXiv:2501.10901）

潜在軸ごとに異なる事前分散を許す階層事前を置く:
    p(z | α) = Π N(z_l; 0, α_l^{-1}),  p(α) = Π Gamma(a_l^0, b_l^0)

無情報ハイパー事前（a^0 = b^0 = 0）のもとで、符号化データ D_z から
    a_L = n/2,  b_L = Σ_i z_il^2 / 2   →   σ̂^2_l = b_L/a_L = E[z_l^2]

関連軸の判定（原著 §3.4）: 関連度スコア σ̂^2_w = w_σ ⊙ σ̂^2 の
**累積 99%** を説明する軸を関連ありとする。w_σ は復号器の軸別感度。

**原著からの逸脱（明記する）**: 原著は α を周辺化した Student's t を
KL の目標分布に使う。本実装は **N(0, σ̂^2) の Gaussian 近似**を用いる。
σ̂^2 は原著と同じ共役更新で推定する。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CFG, DEVICE, set_seed, reconstruction_term
from .models import build

# ── B7 の hard concrete ゲート ──────────────────────────────────────────
GAMMA, ZETA, BETA_HC = -0.1, 1.1, 2.0 / 3.0


class HardConcreteGate(nn.Module):
    """Louizos et al. (2018) の hard concrete 緩和。L0 の期待値が閉形式で書ける。"""

    def __init__(self, m, init_log_alpha=2.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((m,), float(init_log_alpha)))

    def forward(self, batch):
        if self.training:
            u = torch.rand(batch, self.log_alpha.numel(), device=self.log_alpha.device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA_HC)
        else:
            s = torch.sigmoid(self.log_alpha / BETA_HC)
            s = s.unsqueeze(0).expand(batch, -1)
        return torch.clamp(s * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)

    def expected_l0(self):
        """開いているゲート数の期待値（閉形式）。"""
        return torch.sigmoid(self.log_alpha - BETA_HC * np.log(-GAMMA / ZETA)).sum()

    def open_mask(self):
        """推論時: 原著に合わせ σ(γ) > 0.5 相当のゲートを開とする。"""
        z = torch.clamp(torch.sigmoid(self.log_alpha / BETA_HC) * (ZETA - GAMMA) + GAMMA,
                        0.0, 1.0)
        return (z > 0.5)


def run_B7_geco_l0(data, spec, likelihood, m_start, seed, tau,
                   epochs=300, alpha_ma=0.99, lam_min=1e-6, lam_max=1e6, lr=1e-3):
    """GECO + L0。

    tau は**画素平均 MSE** で受け取るが、制約 C は原著 Eq.(8) に合わせて
    **画素についての和** ||x - x'||^2 で評価する（tau も同じ尺度へ換算する）。
    平均で評価すると C が入力次元数分の 1 に潰れ、L0 項（最大 m）に対して
    相対的に無視できる大きさになり、lambda も育たず過剰剪定を起こす。
    """
    set_seed(seed)
    model = build(spec, m_start).to(DEVICE)
    gate = HardConcreteGate(m_start).to(DEVICE)
    lam_raw = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()) + [lam_raw], lr=lr)
    Xtr = torch.as_tensor(data['X_train'])
    n = len(Xtr)
    D = int(np.prod(data['X_train'].shape[1:]))
    tau_sum = tau * D          # 画素平均の目標を画素和の尺度へ換算する
    C_ma = None
    import time
    t0 = time.time()
    for ep in range(epochs):
        model.train(); gate.train()
        for i in range(0, n, 128):
            xb = Xtr[i:i + 128].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            mu, lv = model.encode(xb)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            g = gate(B)
            xh = model.decode(z * g)
            # 制約 C: 原著 Eq.(8) に合わせ画素**和**の二乗誤差 − tau_sum
            rec_sum = F.mse_loss(xh, xb, reduction='none').flatten(1).sum(1)
            C = (rec_sum - tau_sum).mean()
            C_ma = C.detach() if C_ma is None else alpha_ma * C_ma + (1 - alpha_ma) * C.detach()
            # KL はゲートでマスクする（剪定済み次元は効かせない）
            kl_elem = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
            kl = (kl_elem * g).sum(1).mean()
            lam = torch.clamp(F.softplus(lam_raw) ** 2, lam_min, lam_max)

            # 値は移動平均、勾配は現ステップの C から流す（原著の指示）
            C_eff = C_ma + (C - C.detach())

            # min-max: θ,φ,γ について最小化し、λ について最大化する。
            # 単一の最小化器で扱うため λ 側は符号を反転した項を加える。
            loss_model = kl + lam.detach() * C_eff
            if C_ma.item() <= 0:      # 制約充足時のみ L0 を加える（原著 Algorithm 1 line 24）
                loss_model = loss_model + gate.expected_l0()
            loss_lambda = -lam * C_ma.detach()     # λ は C>0 で増え、C<0 で減る

            (loss_model + loss_lambda).backward()
            opt.step()
    model.eval(); gate.eval()
    mask = gate.open_mask()
    sel = int(mask.sum().item())
    return {'selected_m': sel, 'm_start': m_start, 'tau': tau, 'tau_sum': tau_sum,
            'constraint_scale': 'sum over input dimensions (原著 Eq.8)',
            'elapsed_seconds': time.time() - t0,
            'open_gates': mask.cpu().numpy().tolist(),
            'final_constraint_ma': float(C_ma.item())}


def run_B8_ard(data, spec, likelihood, m_start, seed, epochs=300, lr=1e-3,
               alpha_frac=0.1, update_every=10, cum_var=0.99):
    """ARD-VAE。σ̂^2 を保留部分集合 X_α から共役更新し、関連軸を累積 99% で決める。"""
    set_seed(seed)
    import time
    model = build(spec, m_start).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = data['X_train']
    n_a = max(256, int(len(X) * alpha_frac))
    Xa = torch.as_tensor(X[:n_a])            # X_alpha（σ̂^2 推定用）
    Xs = torch.as_tensor(X[n_a:])            # X_sgd（最適化用）
    n = len(Xs)
    prior_var = torch.ones(m_start, device=DEVICE)
    t0 = time.time()

    def update_prior():
        model.eval()
        with torch.no_grad():
            acc = torch.zeros(m_start, device=DEVICE); cnt = 0
            for i in range(0, len(Xa), 512):
                xb = Xa[i:i + 512].to(DEVICE)
                mu, lv = model.encode(xb)
                acc += (mu.pow(2) + lv.exp()).sum(0); cnt += xb.size(0)
        return torch.clamp(acc / cnt, 1e-6, 1e6)      # σ̂^2 = b_L / a_L = E[z^2]

    for ep in range(epochs):
        if ep % update_every == 0:
            prior_var = update_prior()
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            xb = Xs[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            rec = reconstruction_term(xh, xb, likelihood)
            # KL( N(mu, sigma^2) || N(0, prior_var) )
            pv = prior_var.unsqueeze(0)
            kl = 0.5 * ((lv.exp() + mu.pow(2)) / pv - 1 - lv + torch.log(pv)).sum(1).mean()
            (rec + kl).backward(); opt.step()

    prior_var = update_prior()
    # 復号器の軸別感度 w_sigma（有限差分）
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X[:512]).to(DEVICE)
        mu, _ = model.encode(xb)
        base = model.decode(mu)
        w = torch.zeros(m_start, device=DEVICE)
        for j in range(m_start):
            z2 = mu.clone()
            z2[:, j] += torch.sqrt(prior_var[j])
            w[j] = (model.decode(z2) - base).pow(2).flatten(1).mean(1).mean()
    score = (w * prior_var).cpu().numpy()
    order = np.argsort(score)[::-1]
    cum = np.cumsum(score[order]) / max(score.sum(), 1e-12)
    sel = int(np.searchsorted(cum, cum_var) + 1)
    return {'selected_m': sel, 'm_start': m_start,
            'elapsed_seconds': time.time() - t0,
            'relevance_score': score.tolist(),
            'prior_var': prior_var.cpu().numpy().tolist(),
            'cum_var_criterion': cum_var}
