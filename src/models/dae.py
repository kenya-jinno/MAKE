"""
Denoising AutoEncoder (DAE) モデル
訓練時に入力にノイズを加え、クリーンな入力を再構成する。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List


class DenoisingAE(nn.Module):
    """
    Denoising AutoEncoder。
    訓練時に入力にガウスノイズを加えてロバストな表現を学習する。

    Args:
        input_dim: 入力次元
        latent_dim: ボトルネック次元
        hidden_dims: エンコーダの隠れ層次元
        noise_std: 訓練時に加えるノイズの標準偏差
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        noise_std: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.noise_std = noise_std

        # エンコーダ構築
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # デコーダ構築
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """入力をエンコードして潜在表現を返す。"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在表現をデコードして再構成を返す。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        順伝播。

        Args:
            x: 入力テンソル
            add_noise: True なら訓練用ノイズを加える
        Returns:
            recon: 再構成（クリーンな入力を目標）
        """
        if add_noise and self.training:
            x_noisy = x + torch.randn_like(x) * self.noise_std
        else:
            x_noisy = x
        return self.decode(self.encode(x_noisy))


def train_dae(
    model: DenoisingAE,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> List[float]:
    """
    Denoising AutoEncoder を訓練する。

    Args:
        model: DenoisingAE モデル
        train_loader: 訓練データローダー
        epochs: エポック数
        lr: 学習率
        device: 計算デバイス
    Returns:
        losses: エポックごとの平均訓練損失
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device).float()

            optimizer.zero_grad()
            recon = model(x, add_noise=True)  # ノイズ付き入力から再構成
            loss = criterion(recon, x)  # クリーン入力と比較
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  DAE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return losses
