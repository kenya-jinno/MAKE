"""
Variational AutoEncoder (VAE) モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple


class VAE(nn.Module):
    """
    Variational AutoEncoder。

    Args:
        input_dim: 入力次元
        latent_dim: 潜在空間次元
        hidden_dims: エンコーダの隠れ層次元
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # エンコーダ（共有部分）
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder_shared = nn.Sequential(*encoder_layers)

        # mu と logvar の線形層
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # デコーダ
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力をエンコードして mu と logvar を返す。

        Returns:
            mu: 平均ベクトル
            logvar: 対数分散ベクトル
        """
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        再パラメータ化トリック: z = mu + eps * std

        Args:
            mu: 平均
            logvar: 対数分散
        Returns:
            z: サンプリングされた潜在変数
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数をデコードして再構成を返す。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播。

        Returns:
            recon: 再構成
            mu: 平均
            logvar: 対数分散
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """
    VAE 損失関数: 再構成損失 + beta * KL 乖離。

    Args:
        recon: 再構成
        x: 元データ
        mu: 平均
        logvar: 対数分散
        beta: KL 項の重み（beta-VAE）
    Returns:
        total_loss: 合計損失
    """
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = 'cpu'
) -> List[float]:
    """
    VAE を訓練する。

    Args:
        model: VAE モデル
        train_loader: 訓練データローダー
        epochs: エポック数
        lr: 学習率
        beta: KL 項の重み
        device: 計算デバイス
    Returns:
        losses: エポックごとの平均訓練損失
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  VAE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return losses
