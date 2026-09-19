"""§5.4: 未マスク Full モデルへの 30-epoch 追加学習対照。

MAKE53_revision_review.md §5.4 の指摘:
  Adapt は追加計算を使い 18,000 例で学習する。未マスク Full に同じ条件で
  30 epochs 追加学習した対照がないため、改善がマスクによる損失の補償に
  特有なのか、単なる追加学習でも得られるのかを区別できない。

要求された 4 条件:
  Full            : マスク無し、追加学習無し
  Mask            : マスク有り、追加学習無し
  Full + 30 epochs: マスク無し、decoder のみ・同じ 18,000 例・同じ optimizer・同じ予算
  Mask + 30 epochs: マスク有り、同上（= 既存の Adapt）

二つの範囲に分けて実施する。

範囲 A（厳密な対応比較）:
  §5.2 のゼロ中心 ARD run は checkpoint と選択軸 ID を保存済み。
  同一 state から二つの追加学習を、追加学習直前に同一の seed を張って走らせる。
  マスク以外はすべて同一なので、両腕は完全に対応する。

範囲 B（掲載 Table 11 の被覆）:
  sample-centred の 12 件は checkpoint 未保存。学習は set_seed と
  cudnn.deterministic により決定的なので再学習でビット一致の state が戻る
  （実行時に記録済み Full MSE と照合して確認する）。
  relevance 再計算は不要なため、新しい Full+30ep 腕のみを追加し、
  既存記録の Mask / Adapt と対応づける。
  追加学習の RNG 流が既存記録と異なる点は、範囲 B 末尾の
  微調整 seed 感度測定で雑音下限を出して評価する。

実行: python3 src/make53/full_control.py [--scope both]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.make49.common import CFG, DEVICE, set_seed
from src.make49.datasets import LOADERS
from src.make49.models import build
from src.make52.geco_arm import provenance, SPECS, KW, eval_pass
from src.make52.ard_native_pruned import (train_ard, eval_masked, finetune_decoder,
                                          FINETUNE_EPOCHS)

OUT = ROOT / 'results/make53'
CKPT = OUT / 'results'  # placeholder, 実体は下で解決
CKPT = ROOT / 'results/make53/checkpoints'
DATASETS = ['MNIST', 'FashionMNIST', 'dSprites', 'CIFAR10']
SEEDS = [42, 123, 777]
FT_SEEDS = [1, 2, 3]          # 事前固定。微調整 RNG 感度の測定にのみ使う


def pct(new, base):
    return 100.0 * (new - base) / base


def run_finetune(model, state0, data, lik, mask, ft_seed):
    """state0 から復元し、同一 seed を張って decoder のみ 30 epochs 追加学習する。"""
    model.load_state_dict({k: v.to(DEVICE) for k, v in state0.items()})
    for p in model.parameters():
        p.requires_grad_(True)
    set_seed(ft_seed)
    sec = finetune_decoder(model, data, lik, mask, epochs=FINETUNE_EPOCHS)
    return sec


# --------------------------------------------------------------------------
def scope_a(results):
    """ゼロ中心 ARD run: 同一 state・同一 seed で Full+30ep と Mask+30ep を対比。"""
    print('=== 範囲 A: ゼロ中心 ARD（checkpoint 有り・両腕を同一流で再実行）\n', flush=True)
    rows = []
    for ds in ['MNIST', 'dSprites']:
        lik = CFG['likelihood']['by_dataset'][ds]
        anchor = CFG['candidate_grids'][ds][-1]
        data = LOADERS[ds](**KW[ds])
        for s in SEEDS:
            tag = f'ard_zero_{ds}_s{s}'
            rec = json.loads((CKPT / tag / 'record.json').read_text())
            state0 = torch.load(CKPT / tag / 'model_full.pt', map_location=DEVICE)
            mask_sel = torch.tensor(rec['binary_mask'], dtype=torch.float32, device=DEVICE)
            mask_all = torch.ones(anchor, device=DEVICE)

            model = build(SPECS[ds], anchor).to(DEVICE)
            model.load_state_dict({k: v.to(DEVICE) for k, v in state0.items()})
            full = eval_pass(model, data['X_val'], lik)
            full_te = eval_pass(model, data['X_test'], lik)
            msk = eval_masked(model, data['X_val'], lik, mask_sel)
            msk_te = eval_masked(model, data['X_test'], lik, mask_sel)

            t_f = run_finetune(model, state0, data, lik, mask_all, s)
            full30 = eval_pass(model, data['X_val'], lik)
            full30_te = eval_pass(model, data['X_test'], lik)

            t_m = run_finetune(model, state0, data, lik, mask_sel, s)
            mask30 = eval_masked(model, data['X_val'], lik, mask_sel)
            mask30_te = eval_masked(model, data['X_test'], lik, mask_sel)

            r = {'scope': 'A', 'variant': 'zero-centred', 'dataset': ds, 'seed': s,
                 'm_start': anchor, 'selected_m': int(sum(rec['binary_mask'])),
                 'full': full, 'full_test': full_te, 'mask': msk, 'mask_test': msk_te,
                 'full_plus30': full30, 'full_plus30_test': full30_te,
                 'mask_plus30': mask30, 'mask_plus30_test': mask30_te,
                 'finetune_seed': s, 'paired_streams': True,
                 'full_finetune_seconds': t_f, 'mask_finetune_seconds': t_m,
                 'delta_full_pct': pct(full30['mse'], full['mse']),
                 'delta_mask_pct': pct(mask30['mse'], msk['mse']),
                 'recorded_adapt_mse': rec['evaluations']['adapt']['mse']}
            r['mask_specific_pp'] = r['delta_mask_pct'] - r['delta_full_pct']
            rows.append(r); results.append(r)
            print(f"  {ds:<13s} seed {s:<4d} 選択 {r['selected_m']:2d}/{anchor}  "
                  f"Full {full['mse']:.5f} → +30ep {full30['mse']:.5f} ({r['delta_full_pct']:+.2f}%)   "
                  f"Mask {msk['mse']:.5f} → +30ep {mask30['mse']:.5f} ({r['delta_mask_pct']:+.2f}%)   "
                  f"マスク固有 {r['mask_specific_pp']:+.2f}pp", flush=True)
        print(flush=True)
    return rows


# --------------------------------------------------------------------------
def scope_b(results, datasets):
    """掲載 Table 11 の run: 決定的再学習で state を復元し Full+30ep を追加する。"""
    print('=== 範囲 B: sample-centred（決定的再学習・Full+30ep 腕を追加）\n', flush=True)
    rows = []
    for ds in datasets:
        lik = CFG['likelihood']['by_dataset'][ds]
        anchor = CFG['candidate_grids'][ds][-1]
        data = LOADERS[ds](**KW[ds])
        ref = {r['seed']: r for r in json.loads(
            (ROOT / f'results/make52/ard_pruned_{ds}.json').read_text())}
        for s in SEEDS:
            t0 = time.time()
            model, sigma_hat, Xa, tr_sec = train_ard(data, SPECS[ds], lik, anchor, s, 300)
            full = eval_pass(model, data['X_val'], lik)
            rec_full = ref[s]['unpruned']['mse']
            identical = (full['mse'] == rec_full)
            state0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            full_te = eval_pass(model, data['X_test'], lik)
            mask_all = torch.ones(anchor, device=DEVICE)

            t_f = run_finetune(model, state0, data, lik, mask_all, s)
            full30 = eval_pass(model, data['X_val'], lik)
            full30_te = eval_pass(model, data['X_test'], lik)

            d_full = pct(full30['mse'], full['mse'])
            d_mask = pct(ref[s]['pruned_finetuned']['mse'], ref[s]['pruned_masked']['mse'])
            r = {'scope': 'B', 'variant': 'sample-centred', 'dataset': ds, 'seed': s,
                 'm_start': anchor, 'selected_m': ref[s]['selected_m'],
                 'reproduction': {'recorded_full_mse': rec_full,
                                  'retrained_full_mse': full['mse'],
                                  'abs_diff': abs(full['mse'] - rec_full),
                                  'bitwise_identical': identical},
                 'full': full, 'full_test': full_te,
                 'mask': ref[s]['pruned_masked'], 'mask_test': ref[s]['pruned_masked_test'],
                 'full_plus30': full30, 'full_plus30_test': full30_te,
                 'mask_plus30': ref[s]['pruned_finetuned'],
                 'mask_plus30_test': ref[s]['pruned_finetuned_test'],
                 'finetune_seed': s, 'paired_streams': False,
                 'train_seconds': tr_sec, 'full_finetune_seconds': t_f,
                 'wall_seconds': time.time() - t0,
                 'delta_full_pct': d_full, 'delta_mask_pct': d_mask}
            r['mask_specific_pp'] = d_mask - d_full
            rows.append(r); results.append(r)
            print(f"  {ds:<13s} seed {s:<4d} 選択 {r['selected_m']:3d}/{anchor}  "
                  f"再現{'一致' if identical else '不一致'}  "
                  f"Full {full['mse']:.5f} → +30ep {full30['mse']:.5f} ({d_full:+.2f}%)   "
                  f"Mask → Adapt ({d_mask:+.2f}%)   "
                  f"マスク固有 {r['mask_specific_pp']:+.2f}pp", flush=True)
            json.dump(results, open(OUT / 'full_control.json', 'w'), ensure_ascii=False)
        print(flush=True)
    return rows


# --------------------------------------------------------------------------
def ft_sensitivity():
    """微調整 RNG 流の違いが生む雑音下限を MNIST で測る（範囲 B の対応づけの妥当性）。"""
    print('=== 微調整 RNG 感度（MNIST・雑音下限の測定）\n', flush=True)
    ds = 'MNIST'
    lik = CFG['likelihood']['by_dataset'][ds]
    anchor = CFG['candidate_grids'][ds][-1]
    data = LOADERS[ds](**KW[ds])
    mask_all = torch.ones(anchor, device=DEVICE)
    out = []
    for s in SEEDS:
        model, _, _, _ = train_ard(data, SPECS[ds], lik, anchor, s, 300)
        base = eval_pass(model, data['X_val'], lik)['mse']
        state0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        vals = []
        for fs in FT_SEEDS:
            run_finetune(model, state0, data, lik, mask_all, fs)
            vals.append(eval_pass(model, data['X_val'], lik)['mse'])
        deltas = [pct(v, base) for v in vals]
        out.append({'dataset': ds, 'seed': s, 'base_mse': base,
                    'finetune_seeds': FT_SEEDS, 'mse': vals, 'delta_pct': deltas,
                    'delta_sd_pp': float(np.std(deltas, ddof=1)),
                    'delta_range_pp': float(max(deltas) - min(deltas))})
        print(f"  seed {s:<4d} Full {base:.5f}  +30ep 変化 "
              f"{'  '.join(f'{d:+.3f}%' for d in deltas)}  "
              f"SD {out[-1]['delta_sd_pp']:.3f}pp  幅 {out[-1]['delta_range_pp']:.3f}pp", flush=True)
    print(flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='both', choices=['a', 'b', 'both'])
    ap.add_argument('--datasets', default=','.join(DATASETS))
    args = ap.parse_args()

    t_all = time.time()
    results, sens = [], []
    prov = provenance(); prov['estimator'] = 'ARD-VAE 未マスク 30-epoch 対照（MAKE53 §5.4）'
    print(f"§5.4 未マスク 30-epoch 対照  追加学習 {FINETUNE_EPOCHS} epochs  "
          f"装置 {provenance()['gpu']}\n", flush=True)
    if args.scope in ('a', 'both'):
        scope_a(results)
    if args.scope in ('b', 'both'):
        scope_b(results, args.datasets.split(','))
        sens = ft_sensitivity()

    # 範囲ごとに別ファイルへ保存し、統合ファイルは既存の他範囲を保ったまま更新する。
    # （--scope b を単独で再実行しても範囲 A の結果を失わないようにする）
    payload = {'runs': results, 'finetune_rng_sensitivity': sens,
               'provenance': prov, 'finetune_epochs': FINETUNE_EPOCHS,
               'wall_seconds': time.time() - t_all}
    for sc in set(r['scope'] for r in results):
        part = dict(payload, runs=[r for r in results if r['scope'] == sc])
        if sc != 'B':
            part['finetune_rng_sensitivity'] = []
        json.dump(part, open(OUT / f'full_control_scope{sc}.json', 'w'), ensure_ascii=False)
    merged, merged_sens = [], sens
    for sc in ('A', 'B'):
        f = OUT / f'full_control_scope{sc}.json'
        if f.exists():
            d = json.loads(f.read_text())
            merged += d['runs']
            merged_sens = merged_sens or d.get('finetune_rng_sensitivity', [])
    json.dump(dict(payload, runs=merged, finetune_rng_sensitivity=merged_sens),
              open(OUT / 'full_control.json', 'w'), ensure_ascii=False)
    print(f"書き出し: {OUT}/full_control.json  総実時間 {payload['wall_seconds']/60:.1f} 分")


if __name__ == '__main__':
    main()
