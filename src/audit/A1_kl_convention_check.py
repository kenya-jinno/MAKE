import torch, torch.nn.functional as F
torch.manual_seed(0)
B, D = 128, 784
for m in [4, 8, 16, 32, 64]:
    x   = torch.rand(B, D)
    rec = torch.rand(B, D)
    mu  = torch.randn(B, m) * 0.5
    lv  = torch.randn(B, m) * 0.1

    # 現行実装
    kl_impl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    # 標準（潜在次元は和、バッチは平均）
    kl_std  = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
    print(f"m={m:3d}  KL_impl={kl_impl:.6f}  KL_std={kl_std:.6f}  "
          f"ratio=KL_std/KL_impl={kl_std/kl_impl:.2f}  (= m)")
