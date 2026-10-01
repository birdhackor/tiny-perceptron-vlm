"""CPU能讀懂的分塊attention數值版；不是FlashAttention的GPU kernel。"""

import math

import torch


def online_attention(q, k, v, block_size=2):
    """無因果限制的單head矩陣，用online softmax得到同一加權平均。"""
    if block_size < 1:
        raise ValueError("block_size必須為正")
    maximum = torch.full((q.shape[0],), float("-inf"), device=q.device, dtype=q.dtype)
    denominator = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    numerator = torch.zeros(q.shape[0], v.shape[-1], device=q.device, dtype=q.dtype)
    for start in range(0, k.shape[0], block_size):
        scores = q @ k[start : start + block_size].T / math.sqrt(q.shape[-1])
        next_maximum = torch.maximum(maximum, scores.amax(-1))
        correction = (maximum - next_maximum).exp()
        weights = (scores - next_maximum[:, None]).exp()
        numerator = numerator * correction[:, None] + weights @ v[start : start + block_size]
        denominator = denominator * correction + weights.sum(-1)
        maximum = next_maximum
    return numerator / denominator[:, None]
