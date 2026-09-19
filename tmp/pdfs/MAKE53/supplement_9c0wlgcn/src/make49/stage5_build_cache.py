"""段階 5: E4/E3/E5 に必要な候補キャッシュを作り直す。

段階 4 のキャッシュには次が無かったため、保存内容を拡張して再構築する。
  - 次元別の μ 分散（E4 の δ 感度・V1/V2 再評価を**再学習なし**で行うため）
  - validation 曲線（設定表 §1.1 方式 2 の「最終 20 epochs の傾き」特徴量のため）

同じ seed・同じ設定なので学習結果自体は段階 4 と同一であり、
段階 4 の結論は変わらない。保存する統計量が増えるだけである。

CIFAR-10 は E5（自然画像での全手順）のために新規に作る。

実行: プロジェクト直下で python3 src/make49/stage5_build_cache.py [--datasets MNIST,dSprites,CIFAR10]
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.make49.common import CFG, SPLIT_SEED
from src.make49.datasets import LOADERS
from src.make49.cache import CandidateCache

SPECS = {'MNIST': {'kind': 'fc', 'input_dim': 784},
         'FashionMNIST': {'kind': 'fc', 'input_dim': 784},
         'dSprites': {'kind': 'conv', 'in_ch': 1, 'size': 64},
         'CIFAR10': {'kind': 'conv', 'in_ch': 3, 'size': 32}}
KW = {'MNIST': {'flatten': True}, 'FashionMNIST': {'flatten': True},
      'dSprites': {}, 'CIFAR10': {}}
SEEDS = CFG['data_splits']['training_seeds']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='MNIST,dSprites,CIFAR10,FashionMNIST')
    ap.add_argument('--betas', default=None,
                    help='カンマ区切り。既定は設定表の主設定のみ')
    args = ap.parse_args()

    betas = ([float(b) for b in args.betas.split(',')] if args.betas
             else [CFG['beta']['primary']])
    for beta in betas:
      print(f"\n########## beta = {beta} ##########", flush=True)
      for ds in args.datasets.split(','):
        grid = CFG['candidate_grids'][ds]
        lik = CFG['likelihood']['by_dataset'][ds]
        data = LOADERS[ds](**KW[ds])
        rng = np.random.RandomState(SPLIT_SEED + 3)
        n = min(CFG['reference_and_id']['estimation_sample']['n'], len(data['X_train']))
        X_est = data['X_train'][rng.choice(len(data['X_train']), n, replace=False)]
        cache = CandidateCache(ds, data, X_est, lik, SPECS[ds], beta=beta)
        total = len(grid) * len(SEEDS)
        print(f"\n=== {ds}  beta={beta}  尤度={lik}  グリッド {len(grid)} 点 × "
              f"{len(SEEDS)} seeds = {total} ラン ===", flush=True)
        t0 = time.time()
        k = 0
        for m in grid:
            for s in SEEDS:
                k += 1
                r = cache.get_vae(m, s)
                print(f"  [{k:3d}/{total}] b={beta:g} m={m:3d} s={s:<6d} "
                      f"val_MSE={r['val']['mse']:.5f} "
                      f"AU={r['readouts']['B']['au']:3d} ep={r['epochs_run']:3d} "
                      f"累計 {(time.time()-t0)/60:.0f}分", flush=True)
        print(f"=== {ds} beta={beta} 完了 {(time.time()-t0)/60:.1f} 分 ===", flush=True)
    print("\n全データ完了")


if __name__ == '__main__':
    main()
