"""Variants of MIXCNN for ablation / graft study.

Every variant shares the same outer skeleton as the original MIXCNN.py so that
only the studied part changes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftThreshold(nn.Module):
    """Learnable soft-thresholding (DRSN-style shrinkage, TII 2020)."""

    def __init__(self, ch, init=0.01):
        super().__init__()
        self.tau = nn.Parameter(torch.full((1, ch, 1), float(init)))

    def forward(self, x):
        return torch.sign(x) * F.relu(x.abs() - self.tau.abs())


class Mixconv(nn.Module):
    def __init__(self, dim_in=128, channal=128, kersize=64, residual=True,
                 soft=False, multi=False):
        super().__init__()
        self.residual = residual
        self.multi = multi
        if multi:
            # true multi-scale: two parallel depthwise branches, same total width
            half = dim_in // 2
            # grouped depthwise: two channels per group, one output per group,
            # so the two branches keep the total width at dim_in
            self.dw_s = nn.Conv1d(dim_in, half, kernel_size=8, padding="same",
                                  groups=half, bias=False)
            self.dw_l = nn.Conv1d(dim_in, half, kernel_size=kersize, padding="same",
                                  groups=half, bias=False)
        else:
            self.dw = nn.Conv1d(dim_in, dim_in, kernel_size=kersize, padding="same",
                                groups=dim_in, bias=False)
        self.soft = SoftThreshold(channal) if soft else None
        self.bn_2 = nn.BatchNorm1d(channal)
        self.conv_1 = nn.Conv1d(channal, channal, kernel_size=1)
        self.bn_3 = nn.BatchNorm1d(channal)

    def forward(self, x):
        x1 = x
        if self.multi:
            z = torch.cat([self.dw_s(x), self.dw_l(x)], dim=1)
        else:
            z = self.dw(x)
        if self.soft is not None:
            z = self.soft(z)
        z = F.relu(z)
        z = self.bn_2(z)
        if self.residual:
            z = z + x1
        z = self.conv_1(z)
        z = F.relu(z)
        z = self.bn_3(z)
        return z


class Net(nn.Module):
    """MIXCNN with switchable parts.

    variant keys:
      residual  : keep the identity skip inside Mixconv (original: True)
      soft      : insert learnable soft-threshold on the depthwise output
      multi     : replace one k=64 depthwise with k=8 + k=64 branches
      pad       : restore the 1024-length zero padding after the stem (original: True)
    """

    def __init__(self, n_cls=10, residual=True, soft=False, multi=False, pad=True):
        super().__init__()
        self.pad = pad
        self.conv_1 = nn.Conv1d(1, 128, kernel_size=32, stride=4)
        self.bn_1 = nn.BatchNorm1d(128)
        self.mix = nn.ModuleList([
            Mixconv(dim_in=128, channal=128, kersize=64, residual=residual,
                    soft=soft, multi=multi) for _ in range(3)
        ])
        self.bn_2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        target = x.shape[-1]
        x = self.conv_1(x)
        if self.pad:
            total = target - x.shape[-1]
            if total > 0:
                left = total // 2
                x = F.pad(x, (left, total - left), "constant", 0)
        x = self.bn_1(x)
        x = F.relu(x)
        for m in self.mix:
            x = m(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build(name, n_cls=10):
    cfg = dict(base=dict(), nores=dict(residual=False), noPad=dict(pad=False),
               soft=dict(soft=True), multi=dict(multi=True, pad=False),
               soft_multi=dict(soft=True, multi=True, pad=False),
               multi_pad=dict(multi=True, pad=True))
    return Net(n_cls=n_cls, **cfg[name])
