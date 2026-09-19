"""段階 4 の候補キャッシュ。

同じ (データ, m, seed, epochs) の学習は 1 回だけ行い、全方法で共有する。
**費用は方法ごとに独立に計上する**（方法が要求したランの実測時間を足す）。
FONDUE の memoisation（Algorithm 2）は 1 回の方法実行の中での重複要求を無料にする。

保存する量: validation / test の品質、AU、読み取り A/B/C、実時間、epoch 数、終了状態。
"""
import json, os, time

import numpy as np
import torch
import torch.nn.functional as F

from .common import CFG, DEVICE, provenance, set_seed, reconstruction_term
from .models import build, FCAE
from .readouts import all_readouts, readout_B_au

CACHE_DIR = os.path.join('results', 'make49', 'stage4_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_EPOCHS = CFG['budget_and_termination']['max_epochs']
PATIENCE = CFG['budget_and_termination']['early_stopping_patience_epochs']


def _key(dataset, m, seed, epochs, kind='vae', early_stopping=None, beta=None):
    es = '' if early_stopping is None else ('|es1' if early_stopping else '|es0')
    b = '' if beta is None else f'|b{beta:g}'
    return f'{kind}|{dataset}|m{m}|s{seed}|e{epochs}{es}{b}'


class CandidateCache:
    """学習結果の共有キャッシュ。方法はここから引くだけで、自分では学習しない。"""

    def __init__(self, dataset, data, X_est, likelihood, spec, path=None, beta=None):
        self.dataset, self.data, self.X_est = dataset, data, X_est
        self.likelihood, self.spec = likelihood, spec
        # β はキャッシュキーに入れる。入れないと梯子を回したとき誤った結果を返す。
        self.beta = CFG['beta']['primary'] if beta is None else float(beta)
        self.path = path or os.path.join(CACHE_DIR, f'{dataset}.json')
        self.store = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.store = json.load(f)

    def save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.store, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── 候補 VAE ────────────────────────────────────────────────────────
    def get_vae(self, m, seed, epochs=MAX_EPOCHS, early_stopping=True, verbose=False):
        k = _key(self.dataset, m, seed, epochs, early_stopping=early_stopping,
                 beta=self.beta)
        if k in self.store:
            return self.store[k]
        r = self._train_vae(m, seed, epochs, early_stopping, verbose)
        self.store[k] = r
        self.save()
        return r

    def _train_vae(self, m, seed, epochs, early_stopping, verbose):
        set_seed(seed)
        model = build(self.spec, m).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xtr = torch.as_tensor(self.data['X_train'])
        n = len(Xtr)
        best = {'elbo': -float('inf'), 'epoch': -1, 'state': None}
        t0, steps, hist = time.time(), 0, []
        curve_mse, curve_kl = [], []
        beta = self.beta
        stopped_early = False
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 128):
                xb = Xtr[perm[i:i + 128]].to(DEVICE); B = xb.size(0)
                opt.zero_grad()
                xh, mu, lv = model(xb)
                rec = reconstruction_term(xh, xb, self.likelihood)
                kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
                (rec + beta * kl).backward(); opt.step(); steps += 1
            v = self._eval(model, self.data['X_val'])
            hist.append(v['elbo'])
            curve_mse.append(v['mse']); curve_kl.append(v['kl'])
            if v['elbo'] > best['elbo']:
                best = {'elbo': v['elbo'], 'epoch': ep,
                        'state': {kk: x.detach().cpu().clone() for kk, x in model.state_dict().items()}}
            if early_stopping and ep - best['epoch'] >= PATIENCE:
                stopped_early = True
                break
        elapsed = time.time() - t0
        w = hist[-50:]
        plateau = bool(len(w) >= 50 and (max(w) - min(w)) / max(abs(np.mean(w)), 1e-12) < 0.01)
        model.load_state_dict(best['state'])
        val = self._eval(model, self.data['X_val'])
        test = self._eval(model, self.data['X_test'])
        ro = all_readouts(model, self.X_est, seed)
        probe = self._probe(model, seed) if epochs >= MAX_EPOCHS // 2 else None
        if verbose:
            print(f"      [train] m={m:3d} s={seed} ep={len(hist):3d} "
                  f"val_MSE={val['mse']:.5f} AU={ro['B']['au']:3d} {elapsed:6.1f}s", flush=True)
        return {'m': m, 'seed': seed, 'beta': beta,
                'epochs_requested': epochs, 'epochs_run': len(hist),
                'early_stopping': early_stopping, 'stopped_early': stopped_early,
                'best_epoch': best['epoch'], 'plateau': plateau,
                'elapsed_seconds': elapsed, 'update_steps': steps,
                'val': val, 'test': test, 'readouts': ro, 'probe': probe,
                # 方式2（設定表 §1.1）の「最終 20 epochs の D_val の傾き」に必要
                'curve_val_mse': curve_mse, 'curve_val_kl': curve_kl,
                'curve_val_elbo': hist}

    def _eval(self, model, X, batch=512):
        model.eval()
        mse = rec = kl = 0.0
        n = len(X)
        with torch.no_grad():
            for i in range(0, n, batch):
                xb = torch.as_tensor(X[i:i + batch]).to(DEVICE); B = xb.size(0)
                mu, lv = model.encode(xb)
                xh = model.decode(mu)
                mse += F.mse_loss(xh, xb, reduction='sum').item() / xb[0].numel()
                rec += reconstruction_term(xh, xb, self.likelihood).item() * B
                kl += (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()
        return {'mse': mse / n, 'rec': rec / n, 'kl': kl / n, 'elbo': -(rec / n + kl / n)}

    # ── 参照 AE ─────────────────────────────────────────────────────────
    def get_ae(self, m_ref, seed, epochs=100, verbose=False):
        k = _key(self.dataset, m_ref, seed, epochs, kind='ae')
        if k in self.store:
            return self.store[k]
        r = self._train_ae(m_ref, seed, epochs, verbose)
        self.store[k] = r
        self.save()
        return r

    def _train_ae(self, m_ref, seed, epochs, verbose):
        from .common import SPLIT_SEED
        from ..metrics.intrinsic_dim import twonn_estimate, mle_estimate
        set_seed(seed)
        D = int(np.prod(self.data['X_train'].shape[1:]))
        model = FCAE(D, m_ref).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xtr = torch.as_tensor(self.data['X_train'].reshape(len(self.data['X_train']), -1))
        n = len(Xtr); t0 = time.time()
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 128):
                xb = Xtr[perm[i:i + 128]].to(DEVICE)
                opt.zero_grad()
                xh, _ = model(xb)
                F.mse_loss(xh, xb, reduction='sum').div(xb.size(0)).backward()
                opt.step()
        elapsed = time.time() - t0
        model.eval()
        Xe = torch.as_tensor(self.X_est.reshape(len(self.X_est), -1))
        with torch.no_grad():
            zs = [model.encode(Xe[i:i + 512].to(DEVICE)).cpu().numpy() for i in range(0, len(Xe), 512)]
        Z = np.concatenate(zs)
        return {'m_ref': m_ref, 'seed': seed, 'epochs': epochs,
                'elapsed_seconds': elapsed,
                'twonn': float(twonn_estimate(Z)), 'mle': float(mle_estimate(Z, k=10))}

    # ── 下流 probe（B3 用）──────────────────────────────────────────────
    def _probe(self, model, seed):
        """固定した線形 probe。正則化は validation で選ぶ（設定表 §3）。

        費用は別フィールドに記録し、**probe を使う方法にだけ計上する**。
        """
        from sklearn.linear_model import LogisticRegression
        if 'y_train' not in self.data:
            return None
        t0 = time.time()
        Z = {k: self._latents(model, self.data[f'X_{k}']) for k in ('train', 'val', 'test')}
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(Z['train'], self.data['y_train'])
            acc_v = float(clf.score(Z['val'], self.data['y_val']))
            if best is None or acc_v > best['val_acc']:
                best = {'C': C, 'val_acc': acc_v,
                        'test_acc': float(clf.score(Z['test'], self.data['y_test']))}
        best['probe_seconds'] = time.time() - t0
        return best

    def _latents(self, model, X, batch=512):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(X[i:i + batch]).to(DEVICE)
                out.append(model.encode(xb)[0].cpu().numpy())
        return np.concatenate(out)
