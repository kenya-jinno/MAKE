"""実験成果物の保存と機能確認（MAKE53_revision_review.md §5.3）。

指摘: native マスク評価の記録に選択軸 ID・relevance・checkpoint・epoch 履歴が無く、
「どの軸を除去したモデルでその MSE になったか」を summary から独立に再評価できない。

本モジュールは、以後の run で最低限保存すべきものを一箇所に集約する。

保存するもの（§5.3 の列挙に対応）:
  - Full と Adapt 後の model state、モデル構成、実行コードの識別情報
  - 潜在軸ごとの relevance、prior 分散、順位、選択軸 ID、binary mask
  - split・学習 seed・評価 seed、完了 epoch 数、微調整中の validation 記録
  - Full / Mask / Adapt の評価結果と、評価関数・推論方式の識別情報

あわせて §5.3 が挙げる機能確認を実装する:
  - 全軸を残す mask なら Full と同じ出力になる
  - 選択数と mask 中の 1 の個数が一致する
  - decoder-only adaptation で encoder が変化しない
  - 省メモリ版 Jacobian が明示的計算と一致する
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / 'results/make53/checkpoints'
CKPT.mkdir(parents=True, exist_ok=True)


def code_fingerprint(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        fp = ROOT / p
        if fp.exists():
            h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def save_run(tag, model_full_state, model_adapt_state, spec, relevance, prior_var,
             order, selected_ids, mask, meta, evals):
    """1 run 分の成果物を保存し、参照可能な識別子を返す。"""
    d = CKPT / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model_full_state, d / 'model_full.pt')
    if model_adapt_state is not None:
        torch.save(model_adapt_state, d / 'model_adapt.pt')
    rec = {
        'tag': tag, 'model_spec': spec,
        'relevance': [float(x) for x in relevance],
        'prior_var': [float(x) for x in prior_var],
        'rank_order': [int(x) for x in order],
        'selected_axis_ids': [int(x) for x in selected_ids],
        'binary_mask': [int(x) for x in mask],
        'meta': meta, 'evaluations': evals,
        'files': {'model_full': 'model_full.pt',
                  'model_adapt': 'model_adapt.pt' if model_adapt_state is not None else None},
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    (d / 'record.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return str(d.relative_to(ROOT))


# ── §5.3 が挙げる機能確認 ────────────────────────────────────────────
def check_full_mask_identity(model, X, eval_masked, eval_plain, likelihood, m, tol=1e-9):
    """全軸を残す mask なら Full と同じ出力になるか。"""
    from src.make49.common import DEVICE
    ones = torch.ones(m, device=DEVICE)
    a = eval_plain(model, X, likelihood)['mse']
    b = eval_masked(model, X, likelihood, ones)['mse']
    return {'name': 'all-ones mask equals full', 'full': a, 'masked': b,
            'abs_diff': abs(a - b), 'pass': abs(a - b) <= tol}


def check_mask_count(selected_m, mask):
    n = int(np.asarray(mask).sum())
    return {'name': 'selected count equals ones in mask', 'selected_m': int(selected_m),
            'ones_in_mask': n, 'pass': int(selected_m) == n}


def check_encoder_unchanged(state_before, state_after, tol=0.0):
    """decoder-only 微調整で encoder が変化しないか。"""
    keys = [k for k in state_before if k.startswith(('enc', 'fc_mu', 'fc_lv'))]
    worst, worst_k = 0.0, None
    for k in keys:
        d = float((state_before[k].float() - state_after[k].float()).abs().max())
        if d > worst:
            worst, worst_k = d, k
    return {'name': 'encoder unchanged by decoder-only adaptation',
            'n_encoder_tensors': len(keys), 'max_abs_change': worst,
            'worst_key': worst_k, 'pass': worst <= tol}


def check_jacobian_paths(model, X, fast_fn, vjp_fn, n=25, batch=25, rtol=1e-4):
    """省メモリ版 Jacobian が明示的計算と一致するか。"""
    a = fast_fn(model, X, n_samples=n, batch=batch)
    b = vjp_fn(model, X, n_samples=n, batch=batch)
    rel = float((a - b).norm() / a.norm())
    corr = float(np.corrcoef(a.cpu().numpy(), b.cpu().numpy())[0, 1])
    return {'name': 'memory-efficient Jacobian matches explicit', 'relative_diff': rel,
            'correlation': corr, 'pass': rel <= rtol}


def report(checks, path=None):
    ok = all(c['pass'] for c in checks)
    print(f"{'機能確認':46s} {'結果':>6s}")
    for c in checks:
        print(f"  {c['name']:44s} {'OK' if c['pass'] else '**NG**':>6s}")
    if path:
        Path(path).write_text(json.dumps({'all_pass': ok, 'checks': checks},
                                         ensure_ascii=False, indent=1))
    return ok
