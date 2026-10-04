"""第14–15章才引入的架構分支；初始模型不需要讀這個檔案。"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = (x.float().square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * scale.to(x.dtype) * self.weight


def rope(x, positions, base=10000):
    """將成對的 feature 旋轉；位置不會把向量長度改掉。"""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE 需要偶數 feature 維度")
    frequency = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    if positions.ndim == 1:
        angle = positions[:, None] * frequency[None, :]
        angle = angle[None, None]
    else:
        angle = (positions[:, :, None] * frequency)[..., :][:, None]
    cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)


class DenseFFN(nn.Module):
    def __init__(self, width, hidden=None, activation="gelu"):
        super().__init__()
        hidden = hidden or width * 4
        self.up, self.down = nn.Linear(width, hidden), nn.Linear(hidden, width)
        self.activation = activation
        if activation not in ("gelu", "relu2", "swiglu"):
            raise ValueError("activation 必須是 gelu/relu2/swiglu")
        self.gate = nn.Linear(width, hidden) if activation == "swiglu" else None

    def forward(self, x):
        h = self.up(x)
        if self.activation == "gelu":
            h = F.gelu(h)
        elif self.activation == "relu2":
            h = F.relu(h).square()
        else:
            h = F.silu(self.gate(x)) * h
        return self.down(h)


class MoEFFN(nn.Module):
    """可讀的 dropless dispatch；小矩陣與 Python loop 未必比 Dense 快。"""

    def __init__(self, width, experts=4, top_k=2, hidden=None):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k 必須介於 1 與 expert 數量之間")
        self.top_k = top_k
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([DenseFFN(width, hidden) for _ in range(experts)])

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probabilities = self.router(flat).softmax(-1)
        weights, chosen = probabilities.topk(self.top_k, dim=-1)
        # top-1 不重新除以自己：否則 gate=1，任務 loss 無法給 router 梯度。
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            token_rows, slots = torch.where(chosen == i)
            if len(token_rows):
                contribution = expert(flat[token_rows]) * weights[token_rows, slots, None]
                output.index_add_(0, token_rows, contribution)
        load = F.one_hot(chosen, len(self.experts)).float().mean((0, 1))
        importance = probabilities.mean(0)
        auxiliary = len(self.experts) * (load.detach() * importance).sum()
        return output.view(shape), auxiliary, chosen
