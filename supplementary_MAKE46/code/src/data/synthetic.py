"""
合成多様体データ生成モジュール
Swiss Roll, Torus, 球面, 線形部分空間等の合成多様体を生成する。
"""

import numpy as np
from typing import Tuple, Optional


def generate_swiss_roll(
    n_samples: int,
    noise: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Swiss Roll 多様体を生成する（d=2, ambient=3）。

    Args:
        n_samples: サンプル数
        noise: 等方的ガウスノイズの標準偏差
        seed: 乱数シード
    Returns:
        X: データ配列 (n_samples, 3)
        d_true: 内在次元 = 2
        T_basis: 接空間基底 (n_samples, 3, 2)
    """
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n_samples))
    height = rng.uniform(0, 1, n_samples)

    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)

    X = np.column_stack([x, y, z])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    # 接空間基底を計算（∂x/∂t, ∂x/∂h）
    # ∂/∂t: (cos(t) - t*sin(t), 0, sin(t) + t*cos(t))
    dt_x = np.cos(t) - t * np.sin(t)
    dt_y = np.zeros(n_samples)
    dt_z = np.sin(t) + t * np.cos(t)
    dt = np.column_stack([dt_x, dt_y, dt_z])
    # 正規化
    dt_norm = np.linalg.norm(dt, axis=1, keepdims=True) + 1e-8
    dt = dt / dt_norm

    # ∂/∂h: (0, 1, 0)
    dh = np.zeros((n_samples, 3))
    dh[:, 1] = 1.0

    T_basis = np.stack([dt, dh], axis=2)  # (n_samples, 3, 2)

    return X, 2, T_basis


def generate_torus(
    n_samples: int,
    R: float = 3.0,
    r: float = 1.0,
    noise: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    トーラス多様体を生成する（d=2, ambient=3）。

    Args:
        n_samples: サンプル数
        R: 大半径
        r: 小半径
        noise: 等方的ガウスノイズの標準偏差
        seed: 乱数シード
    Returns:
        X: データ配列 (n_samples, 3)
        d_true: 内在次元 = 2
        T_basis: 接空間基底 (n_samples, 3, 2)
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    X = np.column_stack([x, y, z])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    # 接空間基底を計算
    # ∂/∂theta: (-(R+r*cos(phi))*sin(theta), (R+r*cos(phi))*cos(theta), 0)
    dtheta = np.column_stack([
        -(R + r * np.cos(phi)) * np.sin(theta),
        (R + r * np.cos(phi)) * np.cos(theta),
        np.zeros(n_samples)
    ])
    dtheta_norm = np.linalg.norm(dtheta, axis=1, keepdims=True) + 1e-8
    dtheta = dtheta / dtheta_norm

    # ∂/∂phi: (-r*sin(phi)*cos(theta), -r*sin(phi)*sin(theta), r*cos(phi))
    dphi = np.column_stack([
        -r * np.sin(phi) * np.cos(theta),
        -r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi) * np.ones(n_samples)
    ])
    dphi_norm = np.linalg.norm(dphi, axis=1, keepdims=True) + 1e-8
    dphi = dphi / dphi_norm

    T_basis = np.stack([dtheta, dphi], axis=2)  # (n_samples, 3, 2)

    return X, 2, T_basis


def generate_sphere(
    n_samples: int,
    d: int,
    N: int,
    noise: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    d次元球面をN次元アンビエント空間に埋め込んで生成する。

    Args:
        n_samples: サンプル数
        d: 球面の次元（d=2 → S^2, 単位球面）
        N: アンビエント空間の次元（N >= d+1）
        noise: 法線方向ノイズの標準偏差
        seed: 乱数シード
    Returns:
        X: データ配列 (n_samples, N)
        d_true: 内在次元 = d
        T_basis: 接空間基底 (n_samples, N, d)
    """
    rng = np.random.RandomState(seed)
    assert N >= d + 1, "N must be >= d+1"

    # d+1次元の単位球面をサンプリング
    coords = rng.randn(n_samples, d + 1)
    coords = coords / (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-8)

    # N次元に埋め込む（残り次元はゼロ）
    X = np.zeros((n_samples, N))
    X[:, :d + 1] = coords

    if noise > 0:
        # 法線方向ノイズ（球面の法線 = 点自身の方向）
        normal = X.copy()
        X += normal * rng.randn(n_samples, 1) * noise

    # 接空間基底：各点での接空間は法線ベクトル（点そのもの）と直交する部分空間
    T_basis = np.zeros((n_samples, N, d))
    for i in range(n_samples):
        n_vec = X[i] / (np.linalg.norm(X[i]) + 1e-8)
        # グラム-シュミットで接空間基底を構築
        basis_vecs = []
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            # 法線成分を除去
            v = e_j - np.dot(e_j, n_vec) * n_vec
            for b in basis_vecs:
                v = v - np.dot(v, b) * b
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                basis_vecs.append(v / norm)
                if len(basis_vecs) == d:
                    break
        for k, bv in enumerate(basis_vecs):
            T_basis[i, :, k] = bv

    return X, d, T_basis


def generate_linear_subspace(
    n_samples: int,
    d: int,
    N: int,
    noise: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    d次元線形部分空間をN次元アンビエント空間に埋め込む。

    Args:
        n_samples: サンプル数
        d: 部分空間の次元
        N: アンビエント空間の次元（N > d）
        noise: 等方的ガウスノイズの標準偏差
        seed: 乱数シード
    Returns:
        X: データ配列 (n_samples, N)
        d_true: 内在次元 = d
        T_basis: 接空間基底 (n_samples, N, d) — 一定（線形空間のため）
    """
    rng = np.random.RandomState(seed)

    # ランダム正規直交基底を生成
    A = rng.randn(N, d)
    Q, _ = np.linalg.qr(A)
    basis = Q[:, :d]  # (N, d)

    # 係数をサンプリング
    coeffs = rng.randn(n_samples, d)  # (n_samples, d)
    X = coeffs @ basis.T  # (n_samples, N)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    T_basis = np.broadcast_to(basis[np.newaxis, :, :], (n_samples, N, d)).copy()

    return X, d, T_basis


def add_normal_noise(
    X: np.ndarray,
    manifold_basis: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    法線方向のみにノイズを加える。

    Args:
        X: データ配列 (n_samples, N)
        manifold_basis: 接空間基底 (n_samples, N, d)
        sigma: ノイズの標準偏差
    Returns:
        X_noisy: ノイズ付きデータ (n_samples, N)
    """
    n_samples, N = X.shape
    rng = np.random.RandomState(None)

    noise = rng.randn(n_samples, N) * sigma

    # 接空間への射影を除去（法線成分のみ残す）
    d = manifold_basis.shape[2]
    for i in range(n_samples):
        T = manifold_basis[i]  # (N, d)
        # 接空間への射影行列 P = T @ T^T
        P = T @ T.T  # (N, N)
        # 法線成分 = (I - P) @ noise[i]
        noise[i] = noise[i] - P @ noise[i]

    return X + noise
