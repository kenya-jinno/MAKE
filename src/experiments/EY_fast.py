"""
実験 EY (Fast version): クラス制限 MNIST 実験
目的: Manifold Entanglement 仮説の定量的検証
  - 1クラス（数字"1"のみ）: 単一多様体に近い状態
  - 2クラス（"0"と"1"）: 最小の Manifold Entanglement
  - 5クラス（"0"〜"4"）: 中程度の Manifold Entanglement
  - 全10クラス（参照）
  でパイプラインを実行し、AU_sat と d_ID の乖離が
  クラス数に依存するかを定量化する。

Fast version: 各設定後に部分的に保存, 少エポック数
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.vae import VAE, vae_loss
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import active_units

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def load_mnist_classes(classes: list, n_train_per_class: int = 1000,
                        n_test_per_class: int = 200):
    """指定クラスのみ MNIST をロード。"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)

    def filter_and_flatten(dataset, n_per_class):
        imgs, lbls = [], []
        for img, lbl in dataset:
            if lbl in classes:
                imgs.append(img.view(-1).numpy())
                lbls.append(lbl)
        imgs = np.array(imgs, dtype=np.float32)
        lbls = np.array(lbls)
        selected = []
        for c in classes:
            idx = np.where(lbls == c)[0]
            np.random.shuffle(idx)
            selected.append(idx[:n_per_class])
        sel_idx = np.concatenate(selected)
        np.random.shuffle(sel_idx)
        return imgs[sel_idx]

    X_train = filter_and_flatten(train_ds, n_train_per_class)
    X_test  = filter_and_flatten(test_ds,  n_test_per_class)
    print(f"  Classes {classes}: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test


def train_vae(vae, X_train, X_test, epochs=75, batch_size=128, beta=4.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    vae.to(device); vae.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
    vae.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = vae(X_te_t)
        mse = nn.functional.mse_loss(x_hat, X_te_t).item()
        Z = mu_te.cpu().numpy()
    au = active_units(Z, threshold=1e-2)
    return mse, au, Z


def run_sweep(classes, m_list, n_train_per_class, n_test_per_class, epochs=75, label=''):
    X_train, X_test = load_mnist_classes(classes, n_train_per_class, n_test_per_class)
    results = []
    for m in m_list:
        print(f"  [{label}] m={m} ...", end=" ", flush=True)
        vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=m)
        mse, au, Z = train_vae(vae, X_train, X_test, epochs=epochs, beta=4.0)
        id_2nn = twonn_estimate(Z) if len(Z) > 10 else float('nan')
        id_mle = mle_estimate(Z, k=10) if len(Z) > 10 else float('nan')
        results.append({'m': m, 'mse': mse, 'au': au, 'twonn': id_2nn, 'mle': id_mle})
        print(f"AU={au}, TwoNN={id_2nn:.2f}, MSE={mse:.4f}")
    return results


def main():
    # Key m values (skip very small which are clearly suboptimal for MNIST)
    m_list = [2, 4, 8, 12, 16, 20, 32, 64]
    epochs = 75

    all_results = {}

    configs = [
        {'classes': [1],              'label': '1class',  'n_train': 2000, 'n_test': 400},
        {'classes': [0, 1],           'label': '2class',  'n_train': 1000, 'n_test': 200},
        {'classes': list(range(5)),   'label': '5class',  'n_train':  500, 'n_test': 100},
        {'classes': list(range(10)),  'label': '10class', 'n_train':  500, 'n_test': 100},
    ]

    for cfg in configs:
        label = cfg['label']
        print(f"\n=== Classes: {cfg['classes']} ({label}) ===")
        res = run_sweep(
            cfg['classes'], m_list,
            n_train_per_class=cfg['n_train'],
            n_test_per_class=cfg['n_test'],
            epochs=epochs,
            label=label
        )
        all_results[label] = res
        # Save partial results after each config
        with open('results/tables/EY_class_restricted.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial results for {label}")

    # Summary
    print("\n=== Summary: AU_sat and TwoNN ID ===")
    print(f"{'Config':12s}  {'#classes':8s}  {'AU_sat':8s}  {'TwoNN_ID(m=64)':15s}  {'MLE_ID(m=64)':12s}")
    for label, res in all_results.items():
        ausat = max(r['au'] for r in res)
        r64 = next((r for r in res if r['m'] == 64), res[-1])
        n_cls = {'1class': 1, '2class': 2, '5class': 5, '10class': 10}[label]
        print(f"{label:12s}  {n_cls:8d}  {ausat:8d}  {r64['twonn']:15.2f}  {r64['mle']:12.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'1class': 'tab:blue', '2class': 'tab:orange',
              '5class': 'tab:purple', '10class': 'tab:red'}
    labels_nice = {'1class': '1 class (digit 1)', '2class': '2 classes (0,1)',
                   '5class': '5 classes (0-4)', '10class': '10 classes (all)'}

    for label, res in all_results.items():
        ms   = [r['m']    for r in res]
        aus  = [r['au']   for r in res]
        ids  = [r['twonn'] for r in res]
        mses = [r['mse']  for r in res]
        c = colors[label]
        axes[0].plot(ms, aus,  'o-', color=c, label=labels_nice[label])
        axes[1].plot(ms, ids,  's-', color=c)
        axes[2].plot(ms, mses, '^-', color=c)

    for ax, ylabel, title in zip(axes,
            ['AU (VAE, β=4)', 'TwoNN ID (VAE latent)', 'MSE (VAE)'],
            ['AU vs m by class count', 'TwoNN ID vs m', 'MSE vs m']):
        ax.set_xlabel('Bottleneck dim $m$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.4)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/figures/EY_class_restricted.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EY_class_restricted.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: results/figures/EY_class_restricted.{png,pdf}")


if __name__ == '__main__':
    main()
