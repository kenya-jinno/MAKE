"""L0-ARM 推定量の単体検証。

原著 Eq.(12) の推定量が、解析的に勾配の分かる問題で正しい値に収束するかを確認する。
実装の誤りと、手法そのものの性質を切り分けるために必要な検証である。

対象: L(gamma) = E_{g ~ Bern(sigma(gamma))} [ F(g) ],  F(g) = sum_j a_j g_j
解析解: dL/dgamma_j = a_j * sigma'(gamma_j) = a_j * sigma(gamma_j) (1 - sigma(gamma_j))

実行: python3 src/make52/test_arm_estimator.py
"""
import numpy as np
import torch


def arm_grad(gamma, F, n_samples, rng):
    """原著 Eq.(12) の ARM 推定量。

    grad ~= E_u[ ( F(1[u > sigma(-gamma)]) - F(1[u < sigma(gamma)]) ) * (u - 1/2) ]
    """
    m = len(gamma)
    sg = torch.sigmoid(gamma)
    sng = torch.sigmoid(-gamma)
    u = torch.rand(n_samples, m, generator=rng)
    g1 = (u < sg.unsqueeze(0)).float()      # 1[u < sigma(gamma)]
    g2 = (u > sng.unsqueeze(0)).float()     # 1[u > sigma(-gamma)]
    return ((F(g2) - F(g1)).unsqueeze(1) * (u - 0.5)).mean(0)


def main():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    m = 8
    a = torch.randn(m)
    gamma = torch.randn(m) * 1.5

    def F(g):                      # 標本ごとのスカラー値（バッチ平均しない）
        return (g * a.unsqueeze(0)).sum(1)

    exact = a * torch.sigmoid(gamma) * (1 - torch.sigmoid(gamma))
    print("L0-ARM 推定量の単体検証")
    print(f"  次元 {m}、F(g) = sum_j a_j g_j、解析勾配 = a_j sigma'(gamma_j)\n")
    print(f"  {'標本数':>9s} {'相関':>8s} {'相対誤差':>10s} {'符号一致':>9s}")
    ok = False
    for n in [100, 1000, 10000, 100000, 1000000]:
        est = arm_grad(gamma, F, n, rng)
        corr = float(np.corrcoef(est.numpy(), exact.numpy())[0, 1])
        rel = float((est - exact).norm() / exact.norm())
        sign = int((torch.sign(est) == torch.sign(exact)).sum())
        print(f"  {n:9d} {corr:8.4f} {rel:10.4f} {sign:6d}/{m}")
        if n >= 100000 and corr > 0.99 and rel < 0.1:
            ok = True
    print(f"\n  判定: {'推定量は正しい（大標本で解析解に収束）' if ok else '**推定量に誤りがある**'}")

    # バッチ平均した F を渡した場合（誤った使い方）との比較
    print("\n  参考: F をバッチ平均してから渡すと何が起きるか")
    def F_mean(g):
        return (g * a.unsqueeze(0)).sum(1).mean().expand(g.shape[0])
    est_bad = arm_grad(gamma, F_mean, 1000000, rng)
    corr_bad = float(np.corrcoef(est_bad.numpy(), exact.numpy())[0, 1])
    print(f"    相関 {corr_bad:.4f}（標本ごとの差が消えるため情報が失われる）")
    return ok


if __name__ == '__main__':
    main()
