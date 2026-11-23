"""
model.py —— 轻量选择性状态空间（SSM）诊断器 + 三个对照基线

设计约束（对应论文的“部署预算”）：参数量 < 50k，INT8 后 flash < 100 kB，
序列长度 L=61 时不使用 O(L^2) 注意力。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 基础组件 ----------------
class SelectiveSSM(nn.Module):
    """单方向选择性扫描（Mamba 的核心递推），d_state 为状态维度。
    h_t = exp(dt*A) * h_{t-1} + ((exp(dt*A)-1)/A) * dt * B_t * x_t
    y_t = C_t · h_t + D * x_t
    """

    def __init__(self, d_model, d_state=8, d_conv=3):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv // 2, groups=d_model)
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        A = torch.arange(1, d_state + 1).float().repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)   # 小步长起步

    def forward(self, x):                                  # x: B,L,D
        B, L, D = x.shape
        u = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        bc = self.x_proj(u)                                # B,L,2N+1
        dt = F.softplus(self.dt_proj(bc[..., :1]))         # B,L,D
        Bm = bc[..., 1:1 + self.d_state]
        Cm = bc[..., 1 + self.d_state:]
        A = -torch.exp(self.A_log)                         # D,N
        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)                # B,D,1
            dA = torch.exp(dt_t * A.unsqueeze(0))           # B,D,N
            dB = (dA - 1.0) / A.unsqueeze(0) * dt_t         # 精确 ZOH 离散化
            h = dA * h + dB * Bm[:, t, None, :] * u[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * u[:, t])
        y = torch.stack(ys, dim=1)
        return self.out_proj(y)


class SSMBlock(nn.Module):
    """双向 SSM + 门控 + 残差。"""

    def __init__(self, d_model, d_state=8, bidirectional=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state)
        self.bwd = SelectiveSSM(d_model, d_state) if bidirectional else None
        self.gate = nn.Linear(d_model, d_model)
        self.bi = bidirectional

    def forward(self, x):
        h = self.norm(x)
        y = self.fwd(h)
        if self.bi:
            y = 0.5 * (y + self.bwd(torch.flip(h, [1])).flip(1))
        return x + y * torch.sigmoid(self.gate(h))


class TinySSMClassifier(nn.Module):
    """本文模型：响应域 tokenizer + 2 层双向 SSM + 池化头。"""

    def __init__(self, n_classes=6, d_model=32, d_state=8, n_layers=2):
        super().__init__()
        self.inp = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([SSMBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, n_classes)

    def forward(self, x):                                  # B,L,3
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


# ---------------- 对照基线 ----------------
class CNN1D(nn.Module):
    def __init__(self, n_classes=6, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, n_classes))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class GRUClassifier(nn.Module):
    def __init__(self, n_classes=6, d=32):
        super().__init__()
        self.rnn = nn.GRU(3, d, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * d, n_classes)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class TransformerClassifier(nn.Module):
    def __init__(self, n_classes=6, d=32, nhead=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(3, d)
        enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.enc(self.inp(x)).mean(1))


# ---------------- 量化感知训练（QAT）的假量化算子 ----------------
class FakeQuant(nn.Module):
    """对称 8-bit 假量化，反向用直通估计（STE）。"""

    def __init__(self, bits=8, per_channel=False):
        super().__init__()
        self.bits, self.per_channel = bits, per_channel
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        qmax = 2 ** (self.bits - 1) - 1
        s = self.scale.clamp(min=1e-8)
        xq = torch.clamp(torch.round(x / s), -qmax, qmax) * s
        return x + (xq - x).detach()                       # STE


def enable_qat(model, bits=8):
    """把模型中所有 Conv/Linear 权重包一层假量化（最简版 QAT）。"""
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            m.register_forward_pre_hook(lambda mod, inp: _quantize_weight(mod, bits))
    return model


@torch.no_grad()
def _quantize_weight(mod, bits):
    qmax = 2 ** (bits - 1) - 1
    w = mod.weight
    s = w.abs().max() / qmax + 1e-8
    mod.weight.copy_(torch.clamp(torch.round(w / s), -qmax, qmax) * s)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def quantize_ptq(model, bits=8):
    """训练后量化（PTQ）：权重对称 8-bit 舍入，用于快速评估部署精度损失。
    真正的 QAT 需要在训练中插入假量化算子，见论文实验计划（待实现）。"""
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            qmax = 2 ** (bits - 1) - 1
            w = m.weight
            s = w.abs().max() / qmax + 1e-8
            w.copy_(torch.clamp(torch.round(w / s), -qmax, qmax) * s)
    return model
