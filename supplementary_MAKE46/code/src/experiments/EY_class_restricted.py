"""
実験 EY: クラス制限 MNIST 実験
目的: Manifold Entanglement 仮説の定量的検証
  - 1クラス（数字"1"のみ）: 単一多様体に近い状態
  - 2クラス（"0"と"1"）: 最小の Manifold Entanglement
  - 全10クラス（参照）
  でパイプラインを実行し、AU_sat と d_ID の乖離が
  クラス数に依存するかを定量化する。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder
from src.models.vae import VAE, vae_loss
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import active_units, centered_kernel_alignment

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def load_mnist_classes(classes: list, n_train_per_class: int = 2000,
                        n_test_per_class: int = 500):
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
        # 各クラスから n_per_class 枚をランダム選択
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


def train_model(model, X_train, X_test, epochs: int = 150,
                batch_size: int = 128, is_vae: bool = False, beta: float = 4.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_tr = torch.tensor(X_train)
    X_te = torch.tensor(X_test)
    loader = DataLoader(TensorDataset(X_tr), batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            if is_vae:
                x_hat, mu, logvar = model(x)
                loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            else:
                x_hat = model(x)
                loss = nn.functional.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
    # test MSE
    model.eval()
    with torch.no_grad():
        x_te = X_te.to(device)
        if is_vae:
            x_hat, mu, logvar = model(x_te)
        else:
            x_hat = model(x_te)
            mu = model.encode(x_te)
        mse = nn.functional.mse_loss(x_hat, x_te).item()
        Z = mu.cpu().numpy()
    return mse, Z


def run_sweep(classes: list, m_list: list, n_train_per_class: int,
              n_test_per_class: int, epochs: int = 150):
    X_train, X_test = load_mnist_classes(classes, n_train_per_class, n_test_per_class)
    N = X_train.shape[1]  # 784
    results_ae, results_vae = [], []

    for m in m_list:
        print(f"  m={m} ...", end=" ", flush=True)
        # AE
        ae = AutoEncoder(input_dim=N, hidden_dims=[512, 256], latent_dim=m)
        mse_ae, Z_ae = train_model(ae, X_train, X_test, epochs=epochs, is_vae=False)
        au_ae  = X_train.shape[0]  # AE always AU=m
        id_twonn_ae = twonn_estimate(Z_ae) if len(Z_ae) > 10 else float('nan')
        id_mle_ae   = mle_estimate(Z_ae, k=10) if len(Z_ae) > 10 else float('nan')
        results_ae.append({'m': m, 'mse': mse_ae, 'au': m,
                            'twonn': id_twonn_ae, 'mle': id_mle_ae})

        # VAE
        vae = VAE(input_dim=N, hidden_dims=[512, 256], latent_dim=m)
        mse_vae, Z_vae = train_model(vae, X_train, X_test, epochs=epochs,
                                      is_vae=True, beta=4.0)
        # compute AU
        vae.to(device); vae.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_test).to(device)
            mu_all, logvar_all = [], []
            bs = 512
            for i in range(0, len(X_test), bs):
                xb = X_te_t[i:i+bs]
                mu_b, lv_b = vae.encode(xb)
                mu_all.append(mu_b.cpu())
                logvar_all.append(lv_b.cpu())
            mu_all = torch.cat(mu_all, 0).numpy()
        au_vae = active_units(mu_all, threshold=1e-2)
        id_twonn_vae = twonn_estimate(mu_all) if len(mu_all) > 10 else float('nan')
        id_mle_vae   = mle_estimate(mu_all, k=10) if len(mu_all) > 10 else float('nan')
        results_vae.append({'m': m, 'mse': mse_vae, 'au': au_vae,
                             'twonn': id_twonn_vae, 'mle': id_mle_vae})
        print(f"AE mse={mse_ae:.4f} | VAE au={au_vae} id={id_twonn_vae:.2f}")

    return results_ae, results_vae


def main():
    m_list = [1, 2, 4, 8, 12, 16, 20, 32, 64]
    epochs = 150

    configs = [
        {'classes': [1],    'label': '1class',   'n_train': 4000, 'n_test': 1000},
        {'classes': [0, 1], 'label': '2class',   'n_train': 2000, 'n_test': 500},
        {'classes': list(range(10)), 'label': '10class', 'n_train': 2000, 'n_test': 500},
    ]

    all_results = {}
    for cfg in configs:
        label = cfg['label']
        print(f"\n=== Classes: {cfg['classes']} ({label}) ===")
        rae, rvae = run_sweep(
            cfg['classes'], m_list,
            n_train_per_class=cfg['n_train'],
            n_test_per_class=cfg['n_test'],
            epochs=epochs
        )
        all_results[label] = {'ae': rae, 'vae': rvae}

    # Save JSON
    with open('results/tables/EY_class_restricted.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'1class': 'tab:blue', '2class': 'tab:orange', '10class': 'tab:red'}
    labels_nice = {'1class': '1 class (digit 1)', '2class': '2 classes (0,1)',
                   '10class': '10 classes (all)'}

    for label, res in all_results.items():
        ms  = [r['m']    for r in res['vae']]
        aus = [r['au']   for r in res['vae']]
        ids = [r['twonn'] for r in res['vae']]
        mses= [r['mse']  for r in res['vae']]
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

    axes[0].legend()
    plt.tight_layout()
    plt.savefig('results/figures/EY_class_restricted.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EY_class_restricted.pdf', bbox_inches='tight')
    plt.close()

    # ---- Summary table: AU_sat and TwoNN ID at large m ----
    print("\n=== Summary: AU_sat and TwoNN ID at m=64 ===")
    print(f"{'Config':12s}  {'AU_sat':8s}  {'TwoNN_ID(m=64)':15s}  {'MLE_ID(m=64)':12s}")
    for label, res in all_results.items():
        vae_res = res['vae']
        ausat = max(r['au'] for r in vae_res)
        r64 = next((r for r in vae_res if r['m'] == 64), vae_res[-1])
        print(f"{label:12s}  {ausat:8d}  {r64['twonn']:15.2f}  {r64['mle']:12.2f}")


if __name__ == '__main__':
    main()
