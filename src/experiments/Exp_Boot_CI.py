"""
実験 Exp-Boot-CI: Step 1（内在次元推定）の統計的頑健性検証
Review_codex35 Major 5 対応:
  (a) サブサンプリングに基づく 95% 信頼区間（kNN 系推定量では
      復元抽出ブートストラップが重複点により破綻するため，
      非復元サブサンプリングを用いる）
  (b) サブサンプルサイズ感度（n = 500〜3000）
  (c) 原理の異なる追加推定量（DANCo, ESS; scikit-dimension）との相互検証

再利用: E4_extended (Exp-Mnist-Base) の学習済み参照 AE チェックポイント
        results/E4_extended_AE_m{64,128,256}.pth（300 エポック，seed 42，
        hidden [256,128]）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.models.ae import AutoEncoder
from src.metrics.intrinsic_dim import twonn_estimate, mle_estimate

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('results/tables', exist_ok=True)


def load_mnist_subset(n_train: int = 20000, n_test: int = 3000):
    """E4_extended と同一のサブセット抽出（rng seed 42, choice）"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/.cache/datasets'),
        train=False, download=True, transform=transform)

    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(len(train_ds), min(n_train, len(train_ds)), replace=False)
    test_idx = rng.choice(len(test_ds), min(n_test, len(test_ds)), replace=False)

    X_test = np.stack([test_ds[i][0].numpy().flatten() for i in test_idx])
    return X_test.astype(np.float32)


def encode_latents(m_ref: int, X: np.ndarray) -> np.ndarray:
    model = AutoEncoder(input_dim=784, latent_dim=m_ref, hidden_dims=[256, 128])
    sd = torch.load(f'results/E4_extended_AE_m{m_ref}.pth', map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X).to(device)).cpu().numpy()
    return Z


def subsample_stats(Z: np.ndarray, n_sub: int, n_rep: int, rng: np.random.RandomState):
    """非復元サブサンプリングによる TwoNN/MLE の反復推定"""
    tw, ml = [], []
    for _ in range(n_rep):
        idx = rng.choice(len(Z), n_sub, replace=False)
        tw.append(twonn_estimate(Z[idx]))
        ml.append(mle_estimate(Z[idx], k=10))
    return np.array(tw), np.array(ml)


def main():
    X_test = load_mnist_subset()
    print(f"Test subset: {X_test.shape}")

    results = {'seed': SEED, 'n_test': len(X_test),
               'note': 'Subsampling without replacement (bootstrap with '
                       'replacement corrupts kNN-based estimators via '
                       'duplicated points).',
               'm_ref': {}}

    # skdim 追加推定量
    try:
        import skdim
        HAS_SKDIM = True
        print(f"skdim {skdim.__version__} available")
    except ImportError:
        HAS_SKDIM = False
        print("skdim not available; skipping DANCo/ESS")

    def extra_estimators(Z):
        out = {}
        if not HAS_SKDIM:
            return out
        import skdim
        try:
            d = skdim.id.DANCo().fit(Z).dimension_
            out['danco'] = float(d)
        except Exception as e:
            out['danco'] = f'error: {e}'
        try:
            d = skdim.id.ESS().fit(Z).dimension_
            out['ess'] = float(d)
        except Exception as e:
            out['ess'] = f'error: {e}'
        return out

    rng = np.random.RandomState(SEED)
    n_grid = [500, 1000, 1500, 2000, 2500]
    N_REP_SENS = 20      # サブサンプルサイズ感度の反復数
    N_REP_CI = 200       # CI 用サブサンプル反復数
    N_CI = 1500          # CI 用サブサンプルサイズ（n/2）

    # 生入力空間（対照）
    print("\n=== Raw input space (784-dim) ===")
    raw = {'twonn_full': float(twonn_estimate(X_test)),
           'mle_full': float(mle_estimate(X_test, k=10))}
    raw.update(extra_estimators(X_test))
    print(raw)
    results['raw_input'] = raw

    for m_ref in [64, 128, 256]:
        print(f"\n=== m_ref = {m_ref} ===")
        Z = encode_latents(m_ref, X_test)
        entry = {}

        # 全サンプル点推定（論文既報値との照合用）
        entry['twonn_full'] = float(twonn_estimate(Z))
        entry['mle_full'] = float(mle_estimate(Z, k=10))
        print(f"  full-sample: TwoNN={entry['twonn_full']:.2f}, "
              f"MLE={entry['mle_full']:.2f}")

        # (a) サブサンプリング 95% CI（n = N_CI, N_REP_CI 反復）
        tw, ml = subsample_stats(Z, N_CI, N_REP_CI, rng)
        entry['ci'] = {
            'n_sub': N_CI, 'n_rep': N_REP_CI,
            'twonn_mean': float(tw.mean()), 'twonn_std': float(tw.std()),
            'twonn_ci95': [float(np.percentile(tw, 2.5)),
                           float(np.percentile(tw, 97.5))],
            'mle_mean': float(ml.mean()), 'mle_std': float(ml.std()),
            'mle_ci95': [float(np.percentile(ml, 2.5)),
                         float(np.percentile(ml, 97.5))],
        }
        print(f"  CI(n={N_CI}): TwoNN {entry['ci']['twonn_mean']:.2f} "
              f"[{entry['ci']['twonn_ci95'][0]:.2f}, {entry['ci']['twonn_ci95'][1]:.2f}], "
              f"MLE {entry['ci']['mle_mean']:.2f} "
              f"[{entry['ci']['mle_ci95'][0]:.2f}, {entry['ci']['mle_ci95'][1]:.2f}]")

        # (b) サブサンプルサイズ感度
        entry['subsample_sensitivity'] = []
        for n_sub in n_grid:
            tw, ml = subsample_stats(Z, n_sub, N_REP_SENS, rng)
            row = {'n_sub': n_sub,
                   'twonn_mean': float(tw.mean()), 'twonn_std': float(tw.std()),
                   'mle_mean': float(ml.mean()), 'mle_std': float(ml.std())}
            entry['subsample_sensitivity'].append(row)
            print(f"  n={n_sub}: TwoNN {row['twonn_mean']:.2f}±{row['twonn_std']:.2f}, "
                  f"MLE {row['mle_mean']:.2f}±{row['mle_std']:.2f}")

        # (c) 追加推定量
        entry['extra'] = extra_estimators(Z)
        print(f"  extra: {entry['extra']}")

        results['m_ref'][str(m_ref)] = entry

    with open('results/tables/Exp_Boot_CI.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/tables/Exp_Boot_CI.json")


if __name__ == '__main__':
    main()
