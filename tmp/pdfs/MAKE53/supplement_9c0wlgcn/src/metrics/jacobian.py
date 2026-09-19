"""
ヤコビアン解析メトリクス
デコーダのヤコビアン、プルバック計量、接空間近似誤差。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def compute_decoder_jacobian(
    decoder: nn.Module,
    z: torch.Tensor
) -> torch.Tensor:
    """
    デコーダのヤコビアン行列を自動微分で計算する。

    Args:
        decoder: デコーダネットワーク
        z: 潜在変数 (latent_dim,)
    Returns:
        J: ヤコビアン行列 (ambient_dim, latent_dim)
    """
    z = z.detach().requires_grad_(True)
    if z.dim() == 1:
        z_input = z.unsqueeze(0)
    else:
        z_input = z

    output = decoder(z_input).squeeze(0)  # (ambient_dim,)
    ambient_dim = output.shape[0]
    latent_dim = z.shape[-1]

    J = torch.zeros(ambient_dim, latent_dim)
    for i in range(ambient_dim):
        grad = torch.autograd.grad(
            output[i], z,
            create_graph=False, retain_graph=True
        )[0]
        J[i] = grad.detach()

    return J


def analyze_singular_values(
    J: np.ndarray,
    threshold_ratio: float = 0.1
) -> Tuple[np.ndarray, int, float]:
    """
    ヤコビアン行列の特異値を解析する。

    Args:
        J: ヤコビアン行列 (ambient_dim, latent_dim)
        threshold_ratio: 有意な特異値の閾値（最大特異値に対する比）
    Returns:
        singular_values: 特異値配列（降順）
        n_significant: 有意な特異値数
        condition_number: 条件数（最大/最小有意特異値）
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s = s / (s[0] + 1e-10)  # 正規化

    threshold = threshold_ratio
    n_significant = int(np.sum(s > threshold))

    if n_significant < 1:
        n_significant = 1

    significant_sv = s[:n_significant]
    condition_number = float(significant_sv[0] / (significant_sv[-1] + 1e-10))

    return s, n_significant, condition_number


def pullback_metric(
    decoder: nn.Module,
    z: torch.Tensor
) -> torch.Tensor:
    """
    プルバック計量 G(z) = J_g(z)^T J_g(z) を計算する。

    Args:
        decoder: デコーダネットワーク
        z: 潜在変数 (latent_dim,)
    Returns:
        G: プルバック計量テンソル (latent_dim, latent_dim)
    """
    J = compute_decoder_jacobian(decoder, z)  # (ambient_dim, latent_dim)
    G = J.T @ J  # (latent_dim, latent_dim)
    return G


def tangent_space_approximation_error(
    decoder: nn.Module,
    encoder: nn.Module,
    X: np.ndarray,
    d_true: int,
    k: int = 20,
    device: str = 'cpu'
) -> float:
    """
    デコーダヤコビアンが張る部分空間と、
    局所 PCA で推定した接空間の間の平均主角を計算する。

    Args:
        decoder: デコーダネットワーク
        encoder: エンコーダネットワーク
        X: データ配列 (n_samples, N)
        d_true: 多様体の真の内在次元
        k: 局所 PCA の近傍数
        device: 計算デバイス
    Returns:
        mean_angle: 平均主角 [度] （小さいほど良い）
    """
    n_samples, N = X.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(X)

    # サンプル点を制限（計算コスト削減）
    n_eval = min(100, n_samples)
    eval_idx = np.random.choice(n_samples, n_eval, replace=False)

    angles = []
    decoder.eval()
    encoder.eval()

    for i in eval_idx:
        # 局所 PCA で接空間推定
        neighbors = X[indices[i, 1:], :]  # (k, N)
        center = X[i]
        local_data = neighbors - center
        if local_data.shape[0] < d_true:
            continue

        _, _, Vt_local = np.linalg.svd(local_data, full_matrices=False)
        T_local = Vt_local[:d_true, :].T  # (N, d_true) — 接空間基底

        # デコーダヤコビアンの列空間
        x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(x_tensor).squeeze(0)

        J = compute_decoder_jacobian(decoder, z.to(device))  # (N, latent_dim)
        J_np = J.cpu().numpy()

        # J の列空間の基底（SVD で取得）
        U_J, s_J, _ = np.linalg.svd(J_np, full_matrices=False)
        n_sig = min(d_true, int(np.sum(s_J > 1e-6)))
        if n_sig < 1:
            continue
        T_decoder = U_J[:, :n_sig]  # (N, n_sig)

        # 主角の計算（コサイン類似度行列から）
        M = T_local.T @ T_decoder  # (d_true, n_sig)
        try:
            sv = np.linalg.svd(M, compute_uv=False)
            sv = np.clip(sv, -1, 1)
            principal_angles = np.arccos(sv[:min(d_true, n_sig)])
            angles.append(np.mean(principal_angles) * 180 / np.pi)
        except Exception:
            continue

    return float(np.mean(angles)) if angles else float('nan')
