"""MAKE49 共通モジュール。

`MAKE49_実験設定表.md` / `config/experiment_config_MAKE49.json` の事前登録設定を実装する。
段階 1 の監査結果 A1・A4・A6 の修正をここに集約する。
"""
import hashlib, json, os, platform, random, subprocess, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, 'config', 'experiment_config_MAKE49.json')


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CFG = load_config()
SPLIT_SEED = CFG['data_splits']['split_seed']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── likelihood（設定表 §2.6）────────────────────────────────────────────
def likelihood_for(dataset):
    """データセットに対応する尤度を返す。dSprites のみ Bernoulli。"""
    return CFG['likelihood']['by_dataset'][dataset]


def reconstruction_term(x_hat, x, likelihood):
    """サンプルあたり、入力次元について和をとった再構成項。"""
    B = x.size(0)
    if likelihood == 'bernoulli':
        return F.binary_cross_entropy(x_hat.clamp(1e-6, 1 - 1e-6), x, reduction='sum') / B
    return F.mse_loss(x_hat, x, reduction='sum') / B


# ── A1 の修正: KL は潜在次元について和をとる ────────────────────────────────
def vae_loss(x_hat, x, mu, logvar, beta, likelihood='gaussian'):
    """損失規約 sum_recon_sum_kl_v1。

    再構成項: サンプルあたり、入力次元について和（尤度に応じて MSE か BCE）
    KL 項:   サンプルあたり、潜在次元について和（バッチのみ平均）
    これにより beta は m に依存しない（旧実装は KL を潜在次元でも平均していた）。

    beta の意味は尤度によって変わる。尤度をまたいで同じ beta を同じ強さと解釈しない。
    """
    B = x.size(0)
    rec = reconstruction_term(x_hat, x, likelihood)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return rec + beta * kld, rec, kld


def elbo_terms(x_hat, x, mu, logvar):
    """通常の ELBO（beta=1）の構成要素。beta 重みつき目的関数とは別に必ず記録する。"""
    B = x.size(0)
    rec = F.mse_loss(x_hat, x, reduction='sum') / B
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return -(rec + kld), rec, kld


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── データ分割（§4）: 固定乱数による抽出、インデックス保存 ────────────────
def draw_indices(n_total, n_draw, seed=SPLIT_SEED, offset=0):
    """先頭順ではなく固定乱数の置換から抽出する。offset で重複しない区間を取る。"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    return perm[offset:offset + n_draw]


def split_train_val(idx, ratio=0.9, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed + 1)
    p = rng.permutation(len(idx))
    k = int(round(len(idx) * ratio))
    return idx[p[:k]], idx[p[k:]]


def index_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
    return h.hexdigest()[:16]


# ── 出所情報（§7 / A5 の再発防止）────────────────────────────────────────
def provenance():
    try:
        drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        drv = None
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = None
    return {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'torch_version_cuda': torch.version.cuda,
        'cudnn_version': cudnn,
        'nvidia_driver_version': drv or None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'platform': platform.platform(),
        'code_hash': _code_hash(),
        'loss_convention_id': CFG['loss_convention']['id'],
    }


def _code_hash():
    h = hashlib.sha256()
    for p in sorted([__file__, CONFIG_PATH]):
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


# ── 学習（§1.6 の停止条件）──────────────────────────────────────────────
def train_vae(model, Xtr, Xval, beta, max_epochs, patience, batch_size=128, lr=1e-3,
              log_epochs=(), progress=None, early_stopping=True, likelihood='gaussian'):
    """事前登録の停止条件で学習する。

    - checkpoint は validation 目的関数（通常 ELBO）の最良点のみで選ぶ。test は使わない。
    - early stopping: patience epochs 更新なしで停止。
    - plateau: 直近 50 epochs の相対変化 < 1%。満たさなければ未 plateau。

    early_stopping=False は **E7a の軌跡ラン専用**（設定表 §1.6 の例外）。
    E7a は 5 vs 300 epochs を主要比較にするため、必ず 300 epochs まで回す必要がある。
    plateau は記録するが停止には使わない。選択を伴う実験（E1, E7b, +Q 対照）では True にする。
    """
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    Xtr_t = torch.as_tensor(Xtr)
    Xval_t = torch.as_tensor(Xval).to(DEVICE)

    best = {'val_obj': -float('inf'), 'epoch': -1, 'state': None}
    hist = {'epoch': [], 'val_elbo': [], 'val_rec': [], 'val_kl': [], 'train_rec': []}
    snapshots = {}
    t0 = time.time()
    steps = 0
    stopped_early = False

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_rec = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            xb = Xtr_t[perm[i:i + batch_size]].to(DEVICE)
            opt.zero_grad()
            xh, mu, lv = model(xb)
            loss, rec, _ = vae_loss(xh, xb, mu, lv, beta, likelihood)
            loss.backward()
            opt.step()
            ep_rec += rec.item(); nb += 1; steps += 1

        model.eval()
        with torch.no_grad():
            xh, mu, lv = model(Xval_t)
            elbo, vrec, vkl = elbo_terms(xh, Xval_t, mu, lv)
            val_obj = elbo.item()
        hist['epoch'].append(ep)
        hist['val_elbo'].append(val_obj)
        hist['val_rec'].append(vrec.item())
        hist['val_kl'].append(vkl.item())
        hist['train_rec'].append(ep_rec / nb)

        if val_obj > best['val_obj']:
            best = {'val_obj': val_obj, 'epoch': ep,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        if ep in log_epochs:
            snapshots[ep] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if progress and ep % progress == 0:
            print(f"    ep {ep:4d}/{max_epochs}  val_ELBO={val_obj:11.4f}  "
                  f"best@{best['epoch']}  {time.time()-t0:7.1f}s", flush=True)

        if early_stopping and ep - best['epoch'] >= patience:
            stopped_early = True
            break

    elapsed = time.time() - t0
    v = hist['val_elbo']
    if len(v) >= 50:
        window = v[-50:]
        denom = max(abs(np.mean(window)), 1e-12)
        plateau = (max(window) - min(window)) / denom < 0.01
    else:
        plateau = False

    model.load_state_dict(best['state'])
    return {
        'model': model,
        'history': hist,
        'snapshots': snapshots,
        'best_epoch': best['epoch'],
        'best_val_elbo': best['val_obj'],
        'epochs_run': len(v),
        'stopped_early': stopped_early,
        'early_stopping_enabled': bool(early_stopping),
        'likelihood': likelihood,
        'plateau': bool(plateau),
        'elapsed_seconds': elapsed,
        'update_steps': steps,
    }
