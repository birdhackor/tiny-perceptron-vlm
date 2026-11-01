"""詞彙＋中文字 tokenizer、共享 state memory 與獨立候選交叉注意力。"""

import json
import re
from dataclasses import asdict, dataclass

import torch
from torch import nn

PAD, CLS, SEP, BYTE = 0, 1, 2, 3
PATTERN = re.compile(r"[A-Za-z]+|[0-9]|[\u3400-\u9fff]|[^\s]", re.UNICODE)
KINDS = {"choice": 0, "info": 1, "rubric": 2}


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = dict(vocabulary)
        self.size = 260 + len(self.vocabulary)

    @classmethod
    def fit(cls, texts):
        words = sorted({word for text in texts for word in PATTERN.findall(text)})
        return cls({word: 260 + i for i, word in enumerate(words)})

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("輸入須為非空文字")
        text.encode("utf-8", errors="strict")
        result = []
        for word in PATTERN.findall(text):
            if word in self.vocabulary:
                result.append(self.vocabulary[word])
            else:
                result.extend((BYTE, *(4 + byte for byte in word.encode("utf-8"))))
        return result

    def save(self, path):
        path.write_text(json.dumps(self.vocabulary, ensure_ascii=False, sort_keys=True) + "\n")


def encode_example(example, tokenizer):
    state = [CLS, *tokenizer.encode(example.state)]
    question = tokenizer.encode(example.question)
    branches = [[CLS, *question, SEP, *tokenizer.encode(description)] for _, description in example.candidates]
    if len(state) > 384 or max(map(len, branches)) > 192 or not 1 <= len(branches) <= 7:
        raise ValueError("輸入超出 state384／branch192 tokens／候選1–7 的合約")
    ids = [key for key, _ in example.candidates] + ["none"]
    if len(set(ids)) != len(ids):
        raise ValueError("候選 ID 須唯一且不可使用 none")
    return {
        "state": state,
        "branches": branches,
        "kind": KINDS[example.kind],
        "label": ids.index(example.gold) if example.gold else -100,
    }


def pad(rows, device):
    result = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, device=device)
    return result


def collate(rows, device="cpu"):
    count = max(len(row["branches"]) for row in rows)
    branches = [branch for row in rows for branch in (row["branches"] + [[CLS]] * (count - len(row["branches"])))]
    return {
        "state": pad([row["state"] for row in rows], device),
        "branches": pad(branches, device),
        "counts": torch.tensor([len(row["branches"]) for row in rows], device=device),
        "kinds": torch.tensor([row["kind"] for row in rows], device=device),
        "labels": torch.tensor([row["label"] for row in rows], device=device),
        "max_candidates": count,
    }


@dataclass(frozen=True)
class Config:
    vocabulary_size: int
    width: int = 96
    heads: int = 4
    state_layers: int = 2
    branch_layers: int = 1
    readers: int = 2


class Encoder(nn.Module):
    def __init__(self, config, embedding, length, layers):
        super().__init__()
        self.embedding = embedding
        self.positions = nn.Embedding(length, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 3 * config.width, dropout=0, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, tokens):
        values = self.embedding(tokens) + self.positions(torch.arange(tokens.shape[1], device=tokens.device))
        return self.norm(self.layers(values, src_key_padding_mask=tokens.eq(PAD)))


class DecisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.width, padding_idx=PAD)
        self.state_encoder = Encoder(config, self.embedding, 384, config.state_layers)
        self.branch_encoder = Encoder(config, self.embedding, 192, config.branch_layers)
        self.kinds = nn.Embedding(3, config.width)
        layer = nn.TransformerDecoderLayer(
            config.width, config.heads, config.width * 3, dropout=0, batch_first=True, norm_first=True
        )
        self.readers = nn.TransformerDecoder(layer, config.readers, norm=nn.LayerNorm(config.width))
        self.score = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(), nn.Linear(config.width, 1))
        self.none = nn.Sequential(nn.Linear(2 * config.width + 3, config.width), nn.GELU(), nn.Linear(config.width, 1))
        # 三個三分類抽取頭只在訓練提供輔助監督；推論不使用其輸出或真實欄位。
        self.fact_head = nn.Linear(config.width, 9)

    def forward(self, batch, memory=None, return_auxiliary=False):
        states, branches = batch["state"], batch["branches"]
        b, count = len(states), batch["max_candidates"]
        if memory is None:
            memory = self.state_encoder(states)
        query = self.branch_encoder(branches) + self.kinds(batch["kinds"].repeat_interleave(count))[:, None]
        values = self.readers(
            query,
            memory.repeat_interleave(count, 0),
            tgt_key_padding_mask=branches.eq(PAD),
            memory_key_padding_mask=states.eq(PAD).repeat_interleave(count, 0),
        )[:, 0]
        hidden = values.reshape(b, count, -1)
        scores = self.score(hidden).squeeze(-1)
        valid = torch.arange(count, device=states.device)[None] < batch["counts"][:, None]
        mean = (hidden * valid[..., None]).sum(1) / batch["counts"][:, None]
        maximum = hidden.masked_fill(~valid[..., None], -torch.inf).max(1).values
        mean_score = (scores * valid).sum(1) / batch["counts"]
        max_score = scores.masked_fill(~valid, -torch.inf).max(1).values
        summary = torch.cat(
            (mean, maximum, mean_score[:, None], max_score[:, None], batch["counts"][:, None].float() / 7), -1
        )
        none = self.none(summary)
        logits = torch.full((b, count + 1), -torch.inf, device=states.device)
        logits[:, :count] = scores.masked_fill(~valid, -torch.inf)
        # rubric 欄位缺漏也必須 none，不能捏造等級；info 本身總能回答 yes/no。
        none = none.masked_fill((batch["kinds"] == KINDS["info"])[:, None], -torch.inf)
        logits.scatter_(1, batch["counts"][:, None], none)
        return (logits, self.fact_head(mean).reshape(b, 3, 3)) if return_auxiliary else logits

    def metadata(self):
        return asdict(self.config)
