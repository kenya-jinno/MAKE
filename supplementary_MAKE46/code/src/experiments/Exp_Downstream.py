"""
実験 Exp-Downstream: ID 誘導型選定の実用価値検証
Review_codex35 Major 6 対応:
  - 潜在表現からの下流分類精度（ロジスティック回帰）を m スイープで測定
  - 「フル m グリッド探索」vs「ID 誘導候補範囲」の探索コスト・性能保持を定量化

Part A: 参照 AE チェックポイント（E4_extended, 300 ep, seed 42）を再利用
        m ∈ {1,2,4,8,12,16,20,32,64,128,256}
Part B: FC-VAE (β=4, hidden [512,256], 300 ep, 3 seeds) を再学習し，
        AU・MSE に加えて下流分類精度を測定
        m ∈ {4,8,12,16,20,32,64}（Exp-Mnist-Multi-300 と同一構成）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression

from src.models.ae import AutoEncoder
from src.models.vae import VAE, vae_loss
from src.metrics.structure import active_units
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate

SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
os.makedirs('results/tables', exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_mnist_choice(n_train=20000, n_test=3000):
    """E4_extended と同一のサブセット抽出（rng seed 42, choice）＋ラベル"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    rng = np.random.RandomState(SEED)
    tr_idx = rng.choice(len(train_ds), n_train, replace=False)
    te_idx = rng.choice(len(test_ds), n_test, replace=False)
    X_tr = np.stack([train_ds[i][0].numpy().flatten() for i in tr_idx]).astype(np.float32)
    X_te = np.stack([test_ds[i][0].numpy().flatten() for i in te_idx]).astype(np.float32)
    y_tr = np.array([int(train_ds[i][1]) for i in tr_idx])
    y_te = np.array([int(test_ds[i][1]) for i in te_idx])
    return X_tr, y_tr, X_te, y_te


def load_mnist_head(n_train=20000, n_test=3000):
    """Exp-Mnist-Multi-300 と同一のサブセット抽出（先頭 n サンプル）＋ラベル"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_tr = train_ds.data[:n_train].float().view(-1, 784).numpy() / 255.0
    X_te = test_ds.data[:n_test].float().view(-1, 784).numpy() / 255.0
    y_tr = train_ds.targets[:n_train].numpy()
    y_te = test_ds.targets[:n_test].numpy()
    return X_tr.astype(np.float32), y_tr, X_te.astype(np.float32), y_te


def linear_probe(Z_tr, y_tr, Z_te, y_te):
    """潜在表現上の多クラスロジスティック回帰（線形プローブ）"""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Z_tr, y_tr)
    return float(clf.score(Z_te, y_te))


def part_a():
    print("\n===== Part A: AE (E4_extended checkpoints, 300 ep, seed 42) =====")
    X_tr, y_tr, X_te, y_te = load_mnist_choice()
    m_list = [1, 2, 4, 8, 12, 16, 20, 32, 64, 128, 256]
    rows = []
    # 対照: 生ピクセル上の線形プローブ
    acc_raw = linear_probe(X_tr, y_tr, X_te, y_te)
    print(f"raw pixels (784): acc = {acc_raw:.4f}")
    for m in m_list:
        model = AutoEncoder(input_dim=784, latent_dim=m, hidden_dims=[256, 128])
        sd = torch.load(f'results/E4_extended_AE_m{m}.pth', map_location=device)
        model.load_state_dict(sd)
        model.to(device).eval()
        with torch.no_grad():
            Z_tr = model.encode(torch.tensor(X_tr).to(device)).cpu().numpy()
            Z_te_t = model.encode(torch.tensor(X_te).to(device))
            rec = model.decode(Z_te_t)
            mse = nn.functional.mse_loss(rec, torch.tensor(X_te).to(device)).item()
            Z_te = Z_te_t.cpu().numpy()
        acc = linear_probe(Z_tr, y_tr, Z_te, y_te)
        rows.append({'m': m, 'acc': acc, 'mse': float(mse)})
        print(f"m={m:4d}: acc = {acc:.4f}, mse = {mse:.4f}")
    return {'raw_acc': acc_raw, 'ae_probe': rows}


def train_vae_probe(m, X_tr, y_tr, X_te, y_te, epochs=300, beta=4.0, batch_size=128):
    vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=m)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr)),
                        batch_size=batch_size, shuffle=True)
    vae.to(device).train()
    for ep in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
    vae.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te).to(device)
        x_hat, mu_te, _ = vae(X_te_t)
        mse = nn.functional.mse_loss(x_hat, X_te_t).item()
        Z_te = mu_te.cpu().numpy()
        Z_tr = vae.encode(torch.tensor(X_tr).to(device))[0].cpu().numpy()
    au = int(active_units(Z_te, threshold=1e-2))
    acc = linear_probe(Z_tr, y_tr, Z_te, y_te)
    return {'m': m, 'mse': float(mse), 'au': au, 'acc': acc,
            'twonn': float(twonn_estimate(Z_te)),
            'mle': float(mle_estimate(Z_te, k=10))}


def part_b():
    print("\n===== Part B: FC-VAE (beta=4, 300 ep, 3 seeds) with linear probe =====")
    X_tr, y_tr, X_te, y_te = load_mnist_head()
    seeds = [42, 123, 777]
    m_list = [4, 8, 12, 16, 20, 32, 64]
    all_res = {}
    for seed in seeds:
        set_seed(seed)
        print(f"--- seed {seed} ---")
        res = []
        for m in m_list:
            r = train_vae_probe(m, X_tr, y_tr, X_te, y_te)
            res.append(r)
            print(f"m={m:3d}: acc={r['acc']:.4f}, au={r['au']}, "
                  f"mse={r['mse']:.4f}, twonn={r['twonn']:.2f}")
        all_res[str(seed)] = res
    return all_res


def main():
    out = {'seed_base': SEED}
    out['part_a'] = part_a()
    with open('results/tables/Exp_Downstream.json', 'w') as f:
        json.dump(out, f, indent=2)
    out['part_b'] = part_b()
    with open('results/tables/Exp_Downstream.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/tables/Exp_Downstream.json")


if __name__ == '__main__':
    main()
