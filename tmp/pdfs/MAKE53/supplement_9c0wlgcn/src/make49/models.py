"""MAKE49 モデル定義。既存の Conv アーキテクチャを踏襲し、64x64 用を追加する。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCVAE(nn.Module):
    """MNIST/Fashion 用 全結合 VAE（既存 src/models/vae.py と同じ構成、hidden=[512,256]）。"""

    def __init__(self, input_dim, latent_dim, hidden=(512, 256)):
        super().__init__()
        enc, d = [], input_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.enc = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_lv = nn.Linear(d, latent_dim)
        dec, d = [], latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]; d = h
        dec += [nn.Linear(d, input_dim)]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        h = self.enc(x); return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.decode(z), mu, lv


class ConvVAE(nn.Module):
    """32x32 と 64x64 の両対応 Conv-VAE。32x32 は既存 EI/EH と同じ 3 段構成。"""

    def __init__(self, latent_dim, in_ch, size):
        super().__init__()
        assert size in (32, 64)
        chans = [32, 64, 128] if size == 32 else [32, 64, 128, 256]
        self.size, self.in_ch = size, in_ch
        enc, d = [], in_ch
        for c in chans:
            enc += [nn.Conv2d(d, c, 4, stride=2, padding=1), nn.ReLU()]; d = c
        self.enc = nn.Sequential(*enc)
        self.feat = (d, size // (2 ** len(chans)), size // (2 ** len(chans)))
        flat = d * self.feat[1] * self.feat[2]
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_lv = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        dec = []
        rev = list(reversed(chans))
        for i, c in enumerate(rev):
            nxt = rev[i + 1] if i + 1 < len(rev) else in_ch
            dec += [nn.ConvTranspose2d(c, nxt, 4, stride=2, padding=1)]
            dec += [nn.ReLU()] if i + 1 < len(rev) else [nn.Sigmoid()]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        h = self.enc(x).flatten(1); return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, *self.feat)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.decode(z), mu, lv


def build(spec, latent_dim):
    if spec['kind'] == 'fc':
        return FCVAE(spec['input_dim'], latent_dim)
    return ConvVAE(latent_dim, spec['in_ch'], spec['size'])


class FCAE(nn.Module):
    """参照 AE（設定表 §5.3）。VAE の候補とは別系統。"""

    def __init__(self, input_dim, latent_dim, hidden=(512, 256)):
        super().__init__()
        enc, d = [], input_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]; d = h
        enc += [nn.Linear(d, latent_dim)]
        self.enc = nn.Sequential(*enc)
        dec, d = [], latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]; d = h
        dec += [nn.Linear(d, input_dim)]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z
