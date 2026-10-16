"""直接用PyTorch實作共享循環block及與候選順序無關的共享scorer。"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import CANDIDATES, MEANINGS, QUESTIONS, SPLITS, render


class Tokenizer:
    """公開有限字元表，沒有從測試資料fit詞彙。"""

    def __init__(self, characters=None):
        if characters is None:
            examples = [
                render(
                    {"task": "boxes", "a": 0, "b": 7, "events": [("A", 6), ("B", 0), ("swap", 0)], "query": "B"},
                    shifted=shift,
                )
                for shift in (False, True)
            ]
            examples += [render({"task": "sum", "x": 0, "y": 6, "z": 3}, shifted=shift) for shift in (False, True)]
            examples += [
                render({"task": task, "application": "1234567890", "device": device}, shifted=shift)
                for task in ("card", "address", "app", "none")
                for device in ("手機", "平板", "電腦")
                for shift in (False, True)
            ]
            characters = sorted(
                set("".join(examples + list(MEANINGS.values()) + list(QUESTIONS.values())) + "?0123456789PQ")
            )
        self.characters = characters
        self.ids = {character: i + 3 for i, character in enumerate(characters)}

    def __len__(self):
        return len(self.characters) + 3

    def encode(self, state, question=None, candidate=None):
        parts = [state] if question is None else [state, QUESTIONS[question], MEANINGS[candidate]]
        ids = [1]
        for i, part in enumerate(parts):
            if i:
                ids.append(2)
            try:
                ids.extend(self.ids[c] for c in part)
            except KeyError as error:
                raise ValueError("超出公開有限字元表：" + repr(error.args[0])) from None
        return ids

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {"version": 1, "pad": 0, "cls": 1, "sep": 2, "characters": self.characters},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_bytes())
        if value.get("version") != 1 or (value.get("pad"), value.get("cls"), value.get("sep")) != (0, 1, 2):
            raise ValueError("不支援的tokenizer契約")
        return cls(value["characters"])


@dataclass
class Config:
    vocab_size: int
    width: int = 128
    heads: int = 4
    ff: int = 256
    max_length: int = 128
    c_layers: int = 2
    input_injection: float = 0.5

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def padded(sequences, max_length, device="cpu"):
    length = max(map(len, sequences))
    if length > max_length:
        raise ValueError(f"輸入有{length} tokens，超過公開上限{max_length}")
    result = torch.zeros((len(sequences), length), dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        result[i, : len(sequence)] = torch.tensor(sequence, device=device)
    return result


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff1 = nn.Linear(config.width, config.ff)
        self.ff2 = nn.Linear(config.ff, config.width)

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.norm1(hidden)).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0)
        hidden = hidden + self.out(values.transpose(1, 2).reshape(batch, length, width))
        return hidden + self.ff2(F.gelu(self.ff1(self.norm2(hidden))))


class LoopedSolver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.width)
        self.shared = Block(config)
        self.norm = nn.LayerNorm(config.width)
        self.answer = nn.Linear(config.width, 19)

    def forward(self, tokens, loops=3, all_depths=False, diagnostics=False):
        if type(loops) is not int or not 1 <= loops <= 6:
            raise ValueError("公開loop範圍是1–6")
        if tokens.shape[1] > self.config.max_length:
            raise ValueError("輸入超過max_length")
        original = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        hidden = original
        logits, norms = [], []
        for _ in range(loops):
            hidden = self.shared(hidden + self.config.input_injection * original, tokens != 0)
            logits.append(self.answer(self.norm(hidden[:, 0])))
            if diagnostics:
                norms.append(float(hidden.detach().float()[tokens != 0].square().mean().sqrt().item()))
        result = torch.stack(logits, dim=1) if all_depths else logits[-1]
        return (result, norms) if diagnostics else result


class CandidateScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.c_layers))
        self.norm = nn.LayerNorm(config.width)
        self.score = nn.Linear(config.width, 1)

    def forward(self, tokens):
        """每列一個(state,question,candidate)，沒有候選清單的位置。"""
        if tokens.shape[1] > self.config.max_length:
            raise ValueError("輸入超過max_length")
        hidden = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            hidden = block(hidden, tokens != 0)
        return self.score(self.norm(hidden[:, 0])).squeeze(-1)


def validate_candidates(question, candidates):
    if question not in CANDIDATES:
        raise ValueError("未知typed question")
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("至少兩個不重複的候選意義")
    if not set(candidates) <= set(CANDIDATES[question]):
        raise ValueError("候選不屬於這個typed question")
    if question == "choice" and "none" not in candidates:
        raise ValueError("Choice需保留皆不適合")


def candidate_batch(rows, tokenizer, config, device="cpu"):
    sequences, slices, labels = [], [], []
    max_candidates = max(len(row["candidates"]) for row in rows)
    for row in rows:
        validate_candidates(row["question"], row["candidates"])
        start = len(sequences)
        sequences.extend(tokenizer.encode(row["state"], row["question"], candidate) for candidate in row["candidates"])
        slices.append((start, len(sequences)))
        if "gold" in row:
            labels.append(row["candidates"].index(row["gold"]))
    return (
        padded(sequences, config.max_length, device),
        slices,
        max_candidates,
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def grouped_scores(scores, slices, max_candidates):
    rows = []
    for start, end in slices:
        rows.append(F.pad(scores[start:end], (0, max_candidates - (end - start)), value=float("-inf")))
    return torch.stack(rows)


def audit_tokens(b, c, tokenizer):
    """查實際token序列；即使tokenizer錯把兩字合併，也不能靠text查重蒙混。"""
    result = {}
    for product, rows in (("b", b), ("c", c)):
        observed = defaultdict(lambda: {"splits": set(), "labels": set()})
        for row in rows:
            if row["split"] not in SPLITS:
                continue
            if product == "b":
                key = tuple(tokenizer.encode(row["state"]))
            else:
                key = tuple(
                    tuple(tokenizer.encode(row["state"], row["question"], candidate))
                    for candidate in sorted(row["candidates"])
                )
            observed[key]["splits"].add(row["split"])
            observed[key]["labels"].add(row["gold"])
        conflicts = sum(len(value["labels"]) > 1 for value in observed.values())
        shared = sum(len(value["splits"]) > 1 for value in observed.values())
        if conflicts or (product == "b" and shared):
            raise ValueError("實際token序列有標籤衝突或B跨split重複")
        result[product] = {
            "unique_token_observations": len(observed),
            "cross_split_shared_token_observations": shared,
            "conflicting_token_labels": conflicts,
        }
    return result
