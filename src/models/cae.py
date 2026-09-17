"""
Contractive AutoEncoder (CAE) モデル
エンコーダのヤコビアンにフロベニウスノルムペナルティを加える。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List


class ContractiveAE(nn.Module):
    """
    Contractive AutoEncoder。
    損失: L = L_rec + lambda_c * ||J_f(x)||_F^2

    Args:
        input_dim: 入力次元
        latent_dim: ボトルネック次元
        hidden_dims: エンコーダの隠れ層次元
        lambda_c: 収縮ペナルティの重み
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        lambda_c: float = 1e-4
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.lambda_c = lambda_c

        # エンコーダ構築
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.Tanh())  # CAE は Tanh を使う
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)

        # デコーダ構築
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.Tanh())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """入力をエンコードして潜在表現を返す。"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在表現をデコードして再構成を返す。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播: エンコード → デコード。"""
        return self.decode(self.encode(x))

    def contractive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        エンコーダのヤコビアンのフロベニウスノルム^2 を計算する。

        Args:
            x: 入力 (batch_size, input_dim)
        Returns:
            jac_loss: ヤコビアンノルムの平均
        """
        x = x.requires_grad_(True)
        z = self.encode(x)  # (batch_size, latent_dim)

        jac_loss = torch.tensor(0.0, device=x.device)
        for i in range(z.shape[1]):
            grad = torch.autograd.grad(
                z[:, i].sum(), x,
                create_graph=True, retain_graph=True
            )[0]
            jac_loss = jac_loss + (grad ** 2).sum(dim=1).mean()

        return jac_loss


def train_cae(
    model: ContractiveAE,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> List[float]:
    """
    Contractive AutoEncoder を訓練する。

    Args:
        model: ContractiveAE モデル
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
            x.requires_grad_(True)

            optimizer.zero_grad()
            recon = model(x)
            rec_loss = criterion(recon, x.detach())
            jac_loss = model.contractive_loss(x)
            loss = rec_loss + model.lambda_c * jac_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  CAE Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return losses
