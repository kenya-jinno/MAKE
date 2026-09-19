"""
構造保存メトリクス
Active Units, CKA, Trustworthiness, Continuity, 測地線距離相関。
"""

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from typing import Tuple


def active_units(Z: np.ndarray, threshold: float = 0.01) -> int:
    """
    活性ユニット数を計算する。
    各潜在次元の分散が threshold を超えるものをカウントする。

    Args:
        Z: 潜在表現 (n_samples, latent_dim)
        threshold: 分散の閾値
    Returns:
        n_active: 活性ユニット数
    """
    variances = np.var(Z, axis=0)
    return int(np.sum(variances > threshold))


def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment (CKA) の計算。
    2 つの表現行列の類似度を [0, 1] で返す。

    Args:
        X: 表現行列 (n_samples, dim_x)
        Y: 表現行列 (n_samples, dim_y)
    Returns:
        cka: CKA スコア [0, 1]
    """
    def centering(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def hsic(K: np.ndarray, L: np.ndarray) -> float:
        n = K.shape[0]
        KH = centering(K)
        LH = centering(L)
        return float(np.trace(KH @ LH) / ((n - 1) ** 2))

    K = X @ X.T
    L = Y @ Y.T

    numerator = hsic(K, L)
    denom = np.sqrt(hsic(K, K) * hsic(L, L))

    if denom < 1e-10:
        return 0.0
    return float(numerator / denom)


def trustworthiness(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """
    Trustworthiness（信頼性）を計算する。
    潜在空間で近傍な点が元空間でも近傍であるかを測る。

    Args:
        X: 元空間データ (n_samples, n_features)
        Z: 潜在空間データ (n_samples, latent_dim)
        k: 近傍数
    Returns:
        trust: Trustworthiness スコア [0, 1]
    """
    k = min(k, X.shape[0] - 1)
    return float(sklearn_trustworthiness(X, Z, n_neighbors=k))


def continuity(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """
    Continuity（連続性）を計算する。
    元空間で近傍な点が潜在空間でも近傍であるかを測る。
    Trustworthiness の双対。

    Args:
        X: 元空間データ (n_samples, n_features)
        Z: 潜在空間データ (n_samples, latent_dim)
        k: 近傍数
    Returns:
        cont: Continuity スコア [0, 1]
    """
    k = min(k, X.shape[0] - 1)
    # Continuity は X と Z を入れ替えた Trustworthiness に等しい
    return float(sklearn_trustworthiness(Z, X, n_neighbors=k))


def geodesic_distance_correlation(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """
    測地線距離と潜在空間ユークリッド距離のスピアマン相関を計算する。

    Args:
        X: 元空間データ (n_samples, n_features)
        Z: 潜在空間データ (n_samples, latent_dim)
        k: 近傍グラフ構築に使う近傍数
    Returns:
        corr: スピアマン相関係数 [-1, 1]
    """
    n = X.shape[0]
    k = min(k, n - 1)

    # k-近傍グラフを構築して測地線距離を計算
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 疎グラフ行列を作成
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            rows.append(i)
            cols.append(j)
            vals.append(d)

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # 最短経路（測地線距離の近似）
    geo_dist = shortest_path(graph, directed=False)

    # 無限大の距離（非連結成分）を処理
    finite_mask = np.isfinite(geo_dist)
    if finite_mask.sum() < 10:
        return 0.0

    # 潜在空間ユークリッド距離
    from sklearn.metrics import pairwise_distances
    latent_dist = pairwise_distances(Z)

    # 上三角部分のみ使用（対称行列の重複を避ける）
    triu_idx = np.triu_indices(n, k=1)
    geo_vals = geo_dist[triu_idx]
    lat_vals = latent_dist[triu_idx]

    finite_mask = np.isfinite(geo_vals)
    if finite_mask.sum() < 10:
        return 0.0

    corr, _ = spearmanr(geo_vals[finite_mask], lat_vals[finite_mask])
    return float(corr) if not np.isnan(corr) else 0.0
