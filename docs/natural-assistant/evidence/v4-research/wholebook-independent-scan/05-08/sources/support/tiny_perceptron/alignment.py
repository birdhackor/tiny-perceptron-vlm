"""LoRA、DPO、蒸餾：各自的學習訊號分開實作。"""

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2):
        super().__init__()
        if rank < 1:
            raise ValueError("rank 必須為正")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * (self.alpha / self.rank)

    def merged_weight(self):
        return self.base.weight + self.b @ self.a * (self.alpha / self.rank)


def sequence_log_probability(logits, labels):
    valid = labels != IGNORE
    safe_labels = labels.masked_fill(~valid, 0)
    selected = logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    if not valid.any(-1).all():
        raise ValueError("偏好回答不能没有有效 token")
    return (selected * valid).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("DPO beta 必須為正")
    relative_margin = policy_chosen - policy_rejected - (reference_chosen - reference_rejected).detach()
    return -F.logsigmoid(beta * relative_margin).mean()


def distillation_kl(student_logits, teacher_logits, labels, temperature=2.0):
    """KL(teacher || student)，逐 token 加總詞表，再平均有效位置，含 T²。"""
    if student_logits.shape != teacher_logits.shape or temperature <= 0:
        raise ValueError("教師／學生的詞表與序列 shape 必須相同，temperature 必須為正")
    valid = labels != IGNORE
    if not valid.any():
        raise ValueError("蒸餾不能没有有效回答位置")
    p = (teacher_logits.detach().float() / temperature).softmax(-1)
    log_q = (student_logits.float() / temperature).log_softmax(-1)
    per_token = F.kl_div(log_q, p, reduction="none").sum(-1)
    return per_token[valid].mean() * temperature**2


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必須介於 0 與 1")
    return (1 - alpha) * masked_loss(student_logits, labels) + alpha * distillation_kl(
        student_logits, teacher_logits, labels, temperature
    )


def reliability_bins(confidence, correct, bins=5):
    result = []
    for i in range(bins):
        left, right = i / bins, (i + 1) / bins
        selected = (confidence >= left) & (confidence <= right if i == bins - 1 else confidence < right)
        if selected.any():
            result.append(
                {
                    "left": left,
                    "right": right,
                    "count": int(selected.sum()),
                    "confidence": float(confidence[selected].mean()),
                    "accuracy": float(correct[selected].float().mean()),
                }
            )
    return result
