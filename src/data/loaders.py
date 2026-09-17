"""
実データセットのローダーモジュール
MNIST, Fashion-MNIST, CIFAR-10 を返す。
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_mnist(batch_size: int = 64, normalize: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    MNIST データセットを返す。

    Args:
        batch_size: バッチサイズ
        normalize: 正規化するか（[-1, 1]へのスケーリング）
    Returns:
        train_loader: 訓練データローダー
        test_loader: テストデータローダー
        input_dim: 入力次元 = 784
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 784


def get_fashion_mnist(batch_size: int = 64, normalize: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    Fashion-MNIST データセットを返す。

    Args:
        batch_size: バッチサイズ
        normalize: 正規化するか
    Returns:
        train_loader: 訓練データローダー
        test_loader: テストデータローダー
        input_dim: 入力次元 = 784
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 784


def get_cifar10(batch_size: int = 64, normalize: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    CIFAR-10 データセットを返す（フラット化 = 3072次元）。

    Args:
        batch_size: バッチサイズ
        normalize: 正規化するか
    Returns:
        train_loader: 訓練データローダー
        test_loader: テストデータローダー
        input_dim: 入力次元 = 3072
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ]
    if normalize:
        transform_list.insert(1, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 3072
