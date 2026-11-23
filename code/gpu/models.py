"""Model registry: MIXCNN variants, CNN baselines, TCN and a small Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

import common  # noqa: F401  (sets sys.path for variants/baselines)
from variants import build as build_mixcnn
from baselines import DRSN1D, ResNet1D, WDCNN

MIXCNN = ['base', 'noPad', 'nores', 'soft', 'multi', 'soft_multi']
BASELINES = ['wdcnn', 'resnet1d', 'drsn1d', 'tcn', 'transformer']
ALL_MODELS = MIXCNN + BASELINES


class _TCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        z = self.drop(torch.relu(self.bn1(self.conv1(x))[..., :x.shape[-1]]))
        z = self.drop(torch.relu(self.bn2(self.conv2(z))[..., :x.shape[-1]]))
        return torch.relu(x + z)


class TCN1D(nn.Module):
    def __init__(self, n_classes=10, ch=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[_TCNBlock(ch, d) for d in dilations])
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        z = self.blocks(self.stem(x))
        return self.head(z.mean(dim=2))


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(ff, d_model))

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class Transformer1D(nn.Module):
    """Patch transformer: 1024 samples -> 64 tokens of 16 samples."""

    def __init__(self, n_classes=10, d_model=64, heads=4, layers=4,
                 patch=16):
        super().__init__()
        self.patch = nn.Conv1d(1, d_model, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(
            *[_TransformerBlock(d_model, heads, d_model * 2)
              for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        z = self.patch(x).transpose(1, 2)
        z = z + self.pos[:, :z.shape[1]]
        z = self.norm(self.blocks(z))
        return self.head(z.mean(dim=1))


def build(name: str, n_classes: int = 10) -> nn.Module:
    if name in MIXCNN:
        return build_mixcnn(name, n_cls=n_classes)
    if name == 'wdcnn':
        return WDCNN(n_classes)
    if name == 'resnet1d':
        return ResNet1D(n_classes)
    if name == 'drsn1d':
        return DRSN1D(n_classes)
    if name == 'tcn':
        return TCN1D(n_classes)
    if name == 'transformer':
        return Transformer1D(n_classes)
    raise ValueError(f'unknown model: {name}')


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
