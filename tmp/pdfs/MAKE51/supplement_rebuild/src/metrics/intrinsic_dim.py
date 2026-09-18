"""
内在次元推定メトリクス
TwoNN, MLE, PCA による内在次元推定。
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def twonn_estimate(X: np.ndarray) -> float:
    """
    TwoNN による内在次元推定。
    参考: Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"

    Args:
        X: データ行列 (n_samples, n_features)
    Returns:
        id_estimate: 推定された内在次元
    """
    n = X.shape[0]
    if n < 10:
        return float(X.shape[1])

    # 最近傍 2 点を計算
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)

    r1 = distances[:, 1]  # 1番目の近傍距離
    r2 = distances[:, 2]  # 2番目の近傍距離

    # r2/r1 の比を計算（ゼロ除算を防ぐ）
    mask = r1 > 1e-10
    mu = r2[mask] / r1[mask]

    # 経験分布から直線フィットで内在次元を推定
    mu_sorted = np.sort(mu)
    n_valid = len(mu_sorted)

    # P(mu > x) = 1 - F(x) を直線近似（対数スケール）
    # ln(1 - F(mu)) = -d * ln(mu) → 傾きが -d
    i_vals = np.arange(1, n_valid + 1)
    empirical_cdf = i_vals / n_valid
    mask2 = (empirical_cdf < 1.0) & (mu_sorted > 1.0 + 1e-10)

    if mask2.sum() < 2:
        return float(X.shape[1])

    y = np.log(1 - empirical_cdf[mask2])
    x = np.log(mu_sorted[mask2])

    # 最小二乗フィット（切片なし）
    slope = -np.dot(x, y) / np.dot(x, x)
    return float(max(1.0, slope))


def mle_estimate(X: np.ndarray, k: int = 10) -> float:
    """
    MLE による内在次元推定 (Levina & Bickel, 2004)。

    Args:
        X: データ行列 (n_samples, n_features)
        k: 近傍数
    Returns:
        id_estimate: 推定された内在次元
    """
    n = X.shape[0]
    k = min(k, n - 1)
    if k < 2:
        return float(X.shape[1])

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)

    # 各点での局所推定
    # m_i = (1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)
    r_k = distances[:, k:k+1]  # (n, 1)
    r_j = distances[:, 1:k]    # (n, k-1)

    # ゼロ距離を除外
    valid_mask = (r_k[:, 0] > 1e-10) & np.all(r_j > 1e-10, axis=1)
    if valid_mask.sum() < 5:
        return float(X.shape[1])

    log_ratios = np.log(r_k[valid_mask] / r_j[valid_mask])  # (n_valid, k-1)
    m_i = 1.0 / (np.mean(log_ratios, axis=1))  # 各点の局所推定値

    return float(max(1.0, np.mean(m_i)))


def pca_estimate(X: np.ndarray, threshold: float = 0.95) -> int:
    """
    PCA ベースの内在次元推定。
    分散の threshold 割合を説明するために必要な主成分数を返す。

    Args:
        X: データ行列 (n_samples, n_features)
        threshold: 説明分散の閾値（デフォルト: 0.95）
    Returns:
        n_components: 必要な主成分数
    """
    n_components = min(X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        return 1

    pca = PCA(n_components=n_components)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dims = int(np.searchsorted(cumvar, threshold) + 1)
    return max(1, n_dims)
