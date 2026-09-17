"""
位相的メトリクス
Persistent Homology と RTD (Representation Topology Divergence)。
"""

import numpy as np
from typing import Optional


def compute_rtd(X: np.ndarray, Z: np.ndarray, max_points: int = 2000) -> float:
    """
    RTD (Representation Topology Divergence) を計算する。
    元空間と潜在空間のパーシステントホモロジーのワッサースタイン距離。

    ripser と persim がインストールされている場合のみ動作する。
    インストールされていない場合は 0.0 を返す。

    Args:
        X: 元空間データ (n_samples, n_features)
        Z: 潜在空間データ (n_samples, latent_dim)
        max_points: 最大点数（計算コスト削減のためサブサンプリング）
    Returns:
        rtd: RTD スコア（小さいほど位相的構造が保存されている）
    """
    try:
        from ripser import ripser
        from persim import wasserstein
    except ImportError:
        print("  [Warning] ripser/persim not installed. Skipping RTD computation.")
        return 0.0

    n = X.shape[0]

    # サブサンプリング
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        X = X[idx]
        Z = Z[idx]

    # 正規化（スケール不変にする）
    X_norm = X / (np.std(X) + 1e-8)
    Z_norm = Z / (np.std(Z) + 1e-8)

    try:
        # H0 と H1 のパーシステントホモロジーを計算
        dgms_X = ripser(X_norm, maxdim=1)['dgms']
        dgms_Z = ripser(Z_norm, maxdim=1)['dgms']

        # H1 のワッサースタイン距離
        dgm_X_h1 = dgms_X[1]
        dgm_Z_h1 = dgms_Z[1]

        if len(dgm_X_h1) == 0 and len(dgm_Z_h1) == 0:
            return 0.0

        # 無限大の点を除外
        dgm_X_h1 = dgm_X_h1[np.isfinite(dgm_X_h1[:, 1])] if len(dgm_X_h1) > 0 else dgm_X_h1
        dgm_Z_h1 = dgm_Z_h1[np.isfinite(dgm_Z_h1[:, 1])] if len(dgm_Z_h1) > 0 else dgm_Z_h1

        rtd = float(wasserstein(dgm_X_h1, dgm_Z_h1))
        return rtd

    except Exception as e:
        print(f"  [Warning] RTD computation failed: {e}")
        return 0.0
