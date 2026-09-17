"""
Isometric AutoEncoder モデル
デコーダのヤコビアン J_g^T J_g ≈ I を正則化する。
対応定理: 定理3（プルバック計量）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional


class IsometricAE(nn.Module):
    """
    等長 AutoEncoder。
    損失: L = L_rec + lambda_iso * ||J_g(z)^T J_g(z) - I||_F^2

    Args:
        input_dim: 入力次元
        latent_dim: ボトルネック次元
        hidden_dims: エンコーダの隠れ層次元（デコーダは逆順）
        lambda_iso: 等長正則化の重み
        activation: 活性化関数クラス
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        lambda_iso: float = 1e-2,
        activation: type = nn.ReLU
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.lambda_iso = lambda_iso
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # エンコーダ構築
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(activation())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # デコーダ構築（エンコーダの逆）
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(activation())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """入力をエンコードして潜在表現を返す。"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在表現をデコードして再構成を返す。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """順伝播: エンコード → デコード。"""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def isometry_loss(self, z_batch: torch.Tensor) -> torch.Tensor:
        """
        デコーダの等長損失: ||J_g(z)^T J_g(z) - I||_F^2 のバッチ平均。

        Args:
            z_batch: 潜在コードのバッチ (batch_size, latent_dim)
        Returns:
            loss: 等長損失のスカラー
        """
        batch_size = z_batch.shape[0]
        total_loss = torch.tensor(0.0, device=z_batch.device)

        for i in range(batch_size):
            z_i = z_batch[i].detach().requires_grad_(True)
            z_input = z_i.unsqueeze(0)
            output = self.decoder(z_input).squeeze(0)  # (input_dim,)
            ambient_dim = output.shape[0]
            latent_dim = z_i.shape[0]

            # ヤコビアン J: (ambient_dim, latent_dim)
            J = torch.zeros(ambient_dim, latent_dim, device=z_batch.device)
            for k in range(ambient_dim):
                grad = torch.autograd.grad(
                    output[k], z_i,
                    create_graph=True, retain_graph=True
                )[0]
                J[k] = grad

            # G = J^T J - I (latent_dim, latent_dim)
            G = J.T @ J
            I = torch.eye(latent_dim, device=z_batch.device)
            diff = G - I
            total_loss = total_loss + (diff ** 2).sum()

        return total_loss / batch_size


def train_isometric_ae(
    model: IsometricAE,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu',
    iso_batch_size: int = 8
) -> List[float]:
    """
    Isometric AutoEncoder を訓練する。

    Args:
        model: IsometricAE モデル
        train_loader: 訓練データローダー
        epochs: エポック数
        lr: 学習率
        device: 計算デバイス
        iso_batch_size: 等長損失計算に使うバッチサイズ（計算コスト削減）
    Returns:
        losses: エポックごとの平均訓練損失
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device).float()

            optimizer.zero_grad()
            recon, z = model(x)
            rec_loss = criterion(recon, x)

            # 等長損失は小さいバッチで計算（コスト削減）
            sub_size = min(iso_batch_size, z.shape[0])
            z_sub = z[:sub_size]
            iso_loss = model.isometry_loss(z_sub)

            loss = rec_loss + model.lambda_iso * iso_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  IsometricAE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return losses
