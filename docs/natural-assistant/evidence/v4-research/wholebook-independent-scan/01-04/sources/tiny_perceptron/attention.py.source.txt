"""手寫 attention，先保留完整 [B,H,T,T] 矩陣以便觀察。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def attention_mask(query_positions, key_positions, valid=None, segments=None):
    """True 代表可以看：因果、PAD、文件邊界是三個獨立條件。"""
    allowed = key_positions[None, :] <= query_positions[:, None]
    allowed = allowed[None, None]
    if valid is not None:
        allowed = allowed & valid[:, None, None, :]
    if segments is not None:
        allowed = allowed & (segments[:, None, :, None] == segments[:, None, None, :])
    return allowed


def manual_attention(q, k, v, allowed):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float("-inf"))
    # 全 PAD 的 query 沒有可見 key；softmax(-inf,...) 會 NaN，要明確改成零。
    empty = ~allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(empty, 0.0)
    weights = scores.softmax(dim=-1).masked_fill(empty, 0.0)
    return weights @ v, weights


class CausalAttention(nn.Module):
    def __init__(self, width, heads=1, kv_heads=None, backend="manual", rotary=False):
        super().__init__()
        kv_heads = kv_heads or heads
        if width % heads or heads % kv_heads:
            raise ValueError("width 必須能被 heads 整除，heads 必須能被 kv_heads 整除")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        if rotary and self.head_dim % 2:
            raise ValueError("RoPE 的 head_dim 必須是偶數")
        if backend not in ("manual", "sdpa"):
            raise ValueError("attention backend 必須是 manual 或 sdpa")
        self.backend, self.rotary = backend, rotary
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, valid=None, positions=None, segments=None, cache=None):
        from tiny_perceptron.modern import rope

        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        offset = 0 if cache is None else cache[0].shape[2]
        pos = torch.arange(offset, offset + t, device=x.device) if positions is None else positions
        if self.rotary:
            q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            if valid is not None or segments is not None:
                raise ValueError("最小 cache 路線只接受無 padding、無 packing 的單長度 batch")
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        new_cache = (k, v)  # 儲存未擴張的 KV heads，GQA 才能縮小 cache。
        keys = torch.arange(k.shape[2], device=x.device)
        queries = torch.arange(offset, offset + t, device=x.device)
        allowed = attention_mask(queries, keys, valid, segments)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        if self.backend == "manual":
            out, _ = manual_attention(q, k, v, allowed)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out), new_cache
