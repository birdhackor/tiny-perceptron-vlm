"""第1–2章的兩個起點：只看上一字的表，或同時攤開幾張字卡。"""

import torch
from torch import nn


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, contexts):
        return self.table(contexts[:, -1])


class ContextMLP(nn.Module):
    def __init__(self, vocab_size, context=3, width=16):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, width)
        self.hidden = nn.Linear(context * width, width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, contexts):
        if contexts.shape[1] != self.context:
            raise ValueError("字卡數與模型的固定窗口不一致")
        features = self.embedding(contexts).flatten(1)
        return self.output(torch.tanh(self.hidden(features)))
