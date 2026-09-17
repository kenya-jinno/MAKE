"""MAKE49 データ読み込み。設定表 §4 の分割規則に従う。

- 部分集合は固定乱数 seed 20260912 による無作為抽出（先頭順を使わない）
- 公式 train を 90:10 で train/val に分割
- test は最終評価のみ
- 抽出インデックスを保存する
"""
import os

import numpy as np
import torchvision

from .common import CFG, SPLIT_SEED, draw_indices, split_train_val, index_hash

CACHE = os.path.expanduser('~/.cache/datasets')


def _pack(Xpool, tr_idx, val_idx, Xtest_pool, te_idx, shape_note, ypool=None, ytest=None):
    out = {
        'X_train': Xpool[tr_idx].astype(np.float32),
        'X_val': Xpool[val_idx].astype(np.float32),
        'X_test': Xtest_pool[te_idx].astype(np.float32),
        'split': {
            'train_idx': tr_idx.tolist(), 'val_idx': val_idx.tolist(), 'test_idx': te_idx.tolist(),
            'index_hash': index_hash(tr_idx, val_idx, te_idx),
            'split_seed': SPLIT_SEED, 'shape_note': shape_note,
        },
    }
    if ypool is not None:
        out['y_train'] = ypool[tr_idx]
        out['y_val'] = ypool[val_idx]
        out['y_test'] = ytest[te_idx]
    return out


def load_mnist(flatten=True):
    sz = CFG['data_splits']['subset_sizes']['MNIST']
    tr = torchvision.datasets.MNIST(root=CACHE, train=True, download=False)
    te = torchvision.datasets.MNIST(root=CACHE, train=False, download=False)
    pool = (tr.data.numpy().astype(np.float32) / 255.0)
    tpool = (te.data.numpy().astype(np.float32) / 255.0)
    if flatten:
        pool = pool.reshape(len(pool), -1); tpool = tpool.reshape(len(tpool), -1)
    else:
        pool = pool[:, None]; tpool = tpool[:, None]
    idx = draw_indices(len(pool), sz['N_from_official_train'])
    tr_i, va_i = split_train_val(idx)
    te_i = draw_indices(len(tpool), sz['test'], seed=SPLIT_SEED + 2)
    return _pack(pool, tr_i, va_i, tpool, te_i,
                 'MNIST 784 flat' if flatten else 'MNIST 1x28x28',
                 ypool=tr.targets.numpy(), ytest=te.targets.numpy())


def load_fashion(flatten=True):
    sz = CFG['data_splits']['subset_sizes']['FashionMNIST']
    tr = torchvision.datasets.FashionMNIST(root=CACHE, train=True, download=False)
    te = torchvision.datasets.FashionMNIST(root=CACHE, train=False, download=False)
    pool = (tr.data.numpy().astype(np.float32) / 255.0)
    tpool = (te.data.numpy().astype(np.float32) / 255.0)
    if flatten:
        pool = pool.reshape(len(pool), -1); tpool = tpool.reshape(len(tpool), -1)
    else:
        pool = pool[:, None]; tpool = tpool[:, None]
    idx = draw_indices(len(pool), sz['N_from_official_train'])
    tr_i, va_i = split_train_val(idx)
    te_i = draw_indices(len(tpool), sz['test'], seed=SPLIT_SEED + 2)
    return _pack(pool, tr_i, va_i, tpool, te_i,
                 'Fashion-MNIST 784 flat' if flatten else 'Fashion-MNIST 1x28x28',
                 ypool=tr.targets.numpy(), ytest=te.targets.numpy())


def load_dsprites():
    """dSprites 64x64 二値。公式の train/test 分割が無いため、
    同一の固定置換から重複しない 2 区間を取る。"""
    sz = CFG['data_splits']['subset_sizes']['dSprites']
    d = np.load(os.path.join(CACHE, 'dsprites.npz'), allow_pickle=True, encoding='latin1')
    imgs = d['imgs']                      # (737280, 64, 64) uint8 {0,1}
    n_all = len(imgs)
    idx = draw_indices(n_all, sz['N'])                                  # 学習用プール
    te_i_global = draw_indices(n_all, sz['test'], offset=sz['N'])       # 重複しない区間
    tr_i, va_i = split_train_val(idx)
    # 保存インデックスと行順を一致させるため、読み出し前にソートして確定させる
    tr_i = np.sort(tr_i); va_i = np.sort(va_i); te_i_global = np.sort(te_i_global)

    def take(gidx):
        # 必要な行だけ読み出す（全 737280 枚を float 化しない）
        return imgs[gidx][:, None].astype(np.float32)

    Xtr = take(tr_i); Xva = take(va_i); Xte = take(te_i_global)
    lat = d['latents_classes']          # [色, 形, 大きさ, 向き, posX, posY]
    ysh = lat[:, 1].astype(np.int64)    # 形 3 クラスを下流ラベルに使う
    return {
        'X_train': Xtr, 'X_val': Xva, 'X_test': Xte,
        'y_train': ysh[tr_i], 'y_val': ysh[va_i], 'y_test': ysh[te_i_global],
        'split': {'train_idx': tr_i.tolist(), 'val_idx': va_i.tolist(),
                  'test_idx': te_i_global.tolist(),
                  'index_hash': index_hash(tr_i, va_i, te_i_global),
                  'split_seed': SPLIT_SEED, 'shape_note': 'dSprites 1x64x64 binary'},
    }


def load_cifar10():
    sz = CFG['data_splits']['subset_sizes']['CIFAR10']
    tr = torchvision.datasets.CIFAR10(root=CACHE, train=True, download=False)
    te = torchvision.datasets.CIFAR10(root=CACHE, train=False, download=False)
    pool = tr.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    tpool = te.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    idx = draw_indices(len(pool), sz['N_from_official_train'])
    tr_i, va_i = split_train_val(idx)
    te_i = draw_indices(len(tpool), sz['test'], seed=SPLIT_SEED + 2)
    return _pack(pool, tr_i, va_i, tpool, te_i, 'CIFAR-10 3x32x32',
                 ypool=np.array(tr.targets), ytest=np.array(te.targets))


LOADERS = {'MNIST': load_mnist, 'FashionMNIST': load_fashion,
           'dSprites': load_dsprites, 'CIFAR10': load_cifar10}
