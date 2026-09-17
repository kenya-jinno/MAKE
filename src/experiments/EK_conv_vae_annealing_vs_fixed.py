"""
実験 EK: MNIST Conv-VAE — KLアニーリング vs 固定β比較
目的: EH（固定β=4, 300ep）では AU≈m だったが，KLサイクリックアニーリングを
     導入することで Conv-VAE でも proper な AU 飽和が得られることを実証する．
     (査読コメント 3.3 への対応)
Architecture: Conv-VAE (EHと同一アーキテクチャ)
m: [4, 8, 12, 16, 20, 32, 64]
条件:
  - fixed:    β=4.0 固定 (EHと同設定)
  - annealed: β_max=4.0, cyclic annealing, n_cycles=4
epochs: 300  seeds: [42, 123, 777]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate
from src.metrics.structure import active_units

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


# ── Conv-VAE (EHと同一アーキテクチャ) ──────────────────────────────────────────
class ConvVAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28→14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_lv = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14→28
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, 64, 7, 7)
        return self.deconv(h).flatten(1)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvVAEEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=4.0):
    rec = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


def cyclic_beta(epoch, total_epochs, n_cycles=4, beta_max=4.0, ratio=0.5):
    """Cyclic annealing schedule"""
    cycle_len = total_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    return beta_max * min(1.0, t / ratio)


def load_mnist(n_train=20000, n_test=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)
    X_train = train_ds.data[:n_train].float().view(-1, 784) / 255.0
    X_test  = test_ds.data[:n_test].float().view(-1, 784) / 255.0
    return X_train.numpy(), X_test.numpy()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_conv_vae(m, X_train, X_test, epochs=300, beta_schedule='fixed',
                   beta_value=4.0, n_cycles=4, batch_size=128):
    model = ConvVAE(m).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        if beta_schedule == 'fixed':
            beta = beta_value
        else:  # cyclic
            beta = cyclic_beta(ep, epochs, n_cycles=n_cycles, beta_max=beta_value)
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test).to(device)
        x_hat, mu_te, _ = model(X_te)
        mse = F.mse_loss(x_hat, X_te).item()
        Z   = mu_te.cpu().numpy()
    au     = active_units(Z, threshold=1e-2)
    id_2nn = twonn_estimate(Z)
    id_mle = mle_estimate(Z, k=10)
    return {'m': m, 'mse': float(mse), 'au': int(au),
            'twonn': float(id_2nn), 'mle': float(id_mle)}


def compute_mknee(results, tau=0.25):
    ms  = [r['m']  for r in results]
    aus = [r['au'] for r in results]
    for i in range(1, len(ms)):
        dm  = ms[i] - ms[i-1]
        dau = aus[i] - aus[i-1]
        r = dau / dm if dm > 0 else 1.0
        if r < tau:
            return ms[i]
    return ms[-1]


def run_experiment(seeds, m_list, epochs, beta_schedule, label, result_key):
    X_train, X_test = load_mnist()
    print(f"Data: train={len(X_train)}, test={len(X_test)}")
    all_results = {}
    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Conv-VAE [{label}]  Seed {seed}  MNIST ===")
        res = []
        for m in m_list:
            print(f"  m={m} ...", end=" ", flush=True)
            r = train_conv_vae(m, X_train, X_test, epochs=epochs,
                               beta_schedule=beta_schedule, beta_value=4.0)
            res.append(r)
            mknee = compute_mknee(res, tau=0.25)
            print(f"AU={r['au']}, TwoNN={r['twonn']:.2f}, mknee={mknee}")
        all_results[str(seed)] = res
    return all_results


def main():
    seeds  = [42, 123, 777]
    m_list = [4, 8, 12, 16, 20, 32, 64]
    epochs = 300

    # ── Fixed beta (EH相当) ──────────────────────────────────────────────────
    print("=" * 60)
    print("CONDITION A: Fixed beta=4.0 (EH equivalent)")
    results_fixed = run_experiment(seeds, m_list, epochs, 'fixed', 'Fixed β=4.0', 'fixed')

    # ── Cyclic annealing ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONDITION B: Cyclic annealing (β_max=4.0, 4 cycles)")
    results_ann   = run_experiment(seeds, m_list, epochs, 'cyclic', 'Cyclic', 'annealed')

    combined = {'fixed': results_fixed, 'annealed': results_ann}
    with open('results/tables/EK_conv_vae_annealing.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary: Fixed vs Annealed ===")
    print(f"{'m':>4} | {'Fixed AU':>10} | {'Anneal AU':>10} | {'Fixed TwoNN':>12} | {'Anneal TwoNN':>12}")
    for i, m in enumerate(m_list):
        aus_f = [results_fixed[str(s)][i]['au'] for s in seeds]
        aus_a = [results_ann[str(s)][i]['au']   for s in seeds]
        t_f   = [results_fixed[str(s)][i]['twonn'] for s in seeds]
        t_a   = [results_ann[str(s)][i]['twonn']   for s in seeds]
        print(f"{m:4d} | {np.mean(aus_f):5.1f}±{np.std(aus_f):.1f}   | "
              f"{np.mean(aus_a):5.1f}±{np.std(aus_a):.1f}   | "
              f"{np.mean(t_f):6.2f}±{np.std(t_f):.2f}   | "
              f"{np.mean(t_a):6.2f}±{np.std(t_a):.2f}")

    mk_f = [compute_mknee(results_fixed[str(s)], tau=0.25) for s in seeds]
    mk_a = [compute_mknee(results_ann[str(s)],   tau=0.25) for s in seeds]
    print(f"\nFixed:    mknee={mk_f},  mean={np.mean(mk_f):.1f}±{np.std(mk_f):.1f}")
    print(f"Annealed: mknee={mk_a},  mean={np.mean(mk_a):.1f}±{np.std(mk_a):.1f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    all_aus_f = np.array([[results_fixed[str(s)][i]['au'] for i in range(len(m_list))] for s in seeds])
    all_aus_a = np.array([[results_ann[str(s)][i]['au']   for i in range(len(m_list))] for s in seeds])
    all_t_f   = np.array([[results_fixed[str(s)][i]['twonn'] for i in range(len(m_list))] for s in seeds])
    all_t_a   = np.array([[results_ann[str(s)][i]['twonn']   for i in range(len(m_list))] for s in seeds])

    mu_f = all_aus_f.mean(0); sd_f = all_aus_f.std(0)
    mu_a = all_aus_a.mean(0); sd_a = all_aus_a.std(0)
    mt_f = all_t_f.mean(0);   st_f = all_t_f.std(0)
    mt_a = all_t_a.mean(0);   st_a = all_t_a.std(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: per-seed AU (fixed)
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for seed, col in zip(seeds, colors):
        aus_s = [results_fixed[str(seed)][i]['au'] for i in range(len(m_list))]
        ax.plot(m_list, aus_s, 'o--', color=col, ms=4, alpha=0.7, label=f'seed={seed}')
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('Fixed β=4.0 (EH条件)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Middle: per-seed AU (annealed)
    ax = axes[1]
    for seed, col in zip(seeds, colors):
        aus_s = [results_ann[str(seed)][i]['au'] for i in range(len(m_list))]
        mk = compute_mknee(results_ann[str(seed)], tau=0.25)
        ax.plot(m_list, aus_s, 'o-', color=col, ms=4, label=f'seed={seed} ($m_{{\\mathrm{{knee}}}}={mk}$)')
        ax.axvline(mk, color=col, ls=':', lw=1, alpha=0.6)
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('KL Cyclic Annealing')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: mean comparison
    ax = axes[2]
    ax.fill_between(m_list, mu_f-sd_f, mu_f+sd_f, alpha=0.2, color='tab:orange')
    ax.plot(m_list, mu_f, 's--', color='tab:orange', ms=5, label='Fixed β=4 (mean±std)')
    ax.fill_between(m_list, mu_a-sd_a, mu_a+sd_a, alpha=0.2, color='tab:blue')
    ax.plot(m_list, mu_a, 'o-',  color='tab:blue',   ms=5, label='Cyclic Annealing (mean±std)')
    ax.plot(m_list, m_list, 'gray', ls='--', lw=1, alpha=0.4, label='AU=m')
    mk_a_mean = int(round(np.mean(mk_a)))
    ax.axvline(mk_a_mean, color='purple', ls='--', lw=1.5,
               label=f'Anneal $m_{{\\mathrm{{knee}}}}$={mk_a_mean}')
    ax.set_xlabel('Bottleneck dim $m$')
    ax.set_ylabel('Active Units (AU)')
    ax.set_title('Fixed β vs KL Annealing: AU comparison')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/EK_conv_vae_annealing.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/EK_conv_vae_annealing.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/EK_conv_vae_annealing.{png,pdf}")


if __name__ == '__main__':
    main()
