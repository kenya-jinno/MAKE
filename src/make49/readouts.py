"""E7a の読み取り A/B/C。設定表 §1.2 の定義をそのまま実装する。

A: FONDUE の変数型分類（U12 の定義）。H2 の主比較は active + mixed（= m - passive）。
B: 標準 AU。Var_x(mu_j) > delta。
C: FONDUE 本体の IDE_z - IDE_mu。連続量なので A/B とは別パネルで扱う。

A と B は閾値で離散化した個数、C は連続量である。
**A/B/C の生の SD を一列に並べて最小を最良と順位づけしない**（設定表 §1.2）。
"""
import numpy as np
import torch

from .common import CFG, DEVICE
from ..metrics.intrinsic_dim import twonn_estimate

# U12: commit f6d6e66b の variable_filter.py で確認した閾値
VAR_THRESHOLD = 0.1
MEAN_ERROR_RANGE = 0.1


def encode_stats(model, X, batch=512):
    """mu と sigma^2 をデータ全体について集める。sigma^2 は一度だけ指数化する（U14）。"""
    model.eval()
    mus, vars_ = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
            mu, lv = model.encode(xb)
            mus.append(mu.cpu().numpy())
            vars_.append(torch.exp(lv).cpu().numpy())     # 単一指数化
    return np.concatenate(mus), np.concatenate(vars_)


def readout_A_variable_types(sigma2):
    """FONDUE の変数型分類（U12）。

    passive: データ間分散 < 0.1 かつ 平均 in [0.9, 1.1]
    active : データ間分散 < 0.1 かつ 平均 <= 0.1
    mixed  : それ以外
    """
    v = sigma2.var(axis=0)
    mn = sigma2.mean(axis=0)
    low_var = v < VAR_THRESHOLD
    passive = low_var & (np.abs(mn - 1.0) <= MEAN_ERROR_RANGE)
    active = low_var & (mn <= MEAN_ERROR_RANGE)
    mixed = ~(passive | active)
    m = sigma2.shape[1]
    n_a, n_m, n_p = int(active.sum()), int(mixed.sum()), int(passive.sum())
    assert n_a + n_m + n_p == m, f"内訳の合計が m と一致しない: {n_a}+{n_m}+{n_p} != {m}"
    return {'active': n_a, 'mixed': n_m, 'passive': n_p,
            'active_plus_mixed': n_a + n_m,      # H2 の主比較
            'm': m}


def readout_B_au(mu, delta=None):
    """標準 AU。Var_x(mu_j) > delta。"""
    if delta is None:
        delta = CFG['au']['delta_primary']
    v = mu.var(axis=0)
    m = mu.shape[1]
    au = int((v > delta).sum())
    return {'au': au, 'm': m,
            'degenerate_zero': au == 0, 'degenerate_full': au == m,
            'mu_var_scale': float(np.median(v)),   # 絶対閾値の妥当性確認用（段階3 C6）
            # δ 感度と V1/V2 の再評価を**再学習なし**で行うために次元別分散を保存する
            # （設定表 §5.5 E4「AU 閾値や V1/V2 は保存した潜在統計から再評価できる」）
            'mu_var_per_dim': [float(x) for x in np.sort(v)[::-1]]}


def readout_C_ide_gap(mu, sigma2, rng):
    """FONDUE 本体の IDE_z - IDE_mu。z は事後からの 1 サンプル。"""
    z = mu + rng.standard_normal(mu.shape) * np.sqrt(sigma2)
    ide_mu = float(twonn_estimate(mu))
    ide_z = float(twonn_estimate(z))
    return {'ide_mu': ide_mu, 'ide_z': ide_z, 'gap': ide_z - ide_mu}


def all_readouts(model, X_est, seed):
    mu, s2 = encode_stats(model, X_est)
    rng = np.random.default_rng(seed)
    return {'A': readout_A_variable_types(s2),
            'B': readout_B_au(mu),
            'C': readout_C_ide_gap(mu, s2, rng)}
