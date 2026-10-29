"""C-v2：UTF-8 state-only memory、獨立候選分支、對稱 none 與明確快取身分。"""

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PAD, CLS, SEP = 256, 257, 258
PREPROCESSING = "strict-utf8-bytes-0-255-right-pad256-v1"
TYPES = {"choice": 0, "info": 1, "rubric": 2}


def text_bytes(text, limit, name):
    if type(text) is not str or not text.strip():
        raise ValueError(f"{name} 必須是非空文字")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{name} 含非法 Unicode surrogate") from error
    if len(raw) > limit:
        raise ValueError(f"{name} 超過 {limit} bytes")
    return raw


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str


@dataclass(frozen=True)
class Question:
    state: str
    kind: str
    text: str
    candidates: tuple[Candidate, ...]
    gold: str | None = None

    def validate(self):
        text_bytes(self.state, 512, "state")
        text_bytes(self.text, 128, "question")
        if self.kind not in TYPES:
            raise ValueError("僅支援 choice/info/rubric")
        if not 1 <= len(self.candidates) <= 7:
            raise ValueError("實質候選 1–7，加 none 最多 8")
        ids, descriptions = [], []
        for candidate in self.candidates:
            text_bytes(candidate.id, 64, "stable ID")
            text_bytes(candidate.description, 128, "candidate description")
            if candidate.id == "none":
                raise ValueError("none 為系統候選，不能當成實質候選")
            ids.append(candidate.id)
            descriptions.append(candidate.description)
        if len(set(ids)) != len(ids) or len(set(descriptions)) != len(descriptions):
            raise ValueError("候選 ID／完整描述不得重複；語義等價項也須由資料作者先去重")
        if self.kind == "info" and set(ids) != {"yes", "no"}:
            raise ValueError("info 固定 yes/no")
        if self.kind == "rubric" and set(ids) != {"0", "1", "2"}:
            raise ValueError("rubric 固定 0/1/2")
        if self.gold is not None and self.gold not in {*ids, "none"}:
            raise ValueError("gold 不在實質候選／none")
        if self.kind != "choice" and self.gold == "none":
            raise ValueError("info/rubric 不另加 none 類別")
        return self


def pad_bytes(rows, device):
    width = max(map(len, rows))
    tokens = torch.full((len(rows), width), PAD, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, device=device)
    return tokens


def collate_questions(questions, device="cpu"):
    questions = [question.validate() for question in questions]
    count = max(len(question.candidates) for question in questions)
    states, branches, kinds, counts, labels = [], [], [], [], []
    for question in questions:
        states.append([CLS, *text_bytes(question.state, 512, "state")])
        q = list(text_bytes(question.text, 128, "question"))
        kinds.append(TYPES[question.kind])
        counts.append(len(question.candidates))
        ids = [candidate.id for candidate in question.candidates] + (["none"] if question.kind == "choice" else [])
        labels.append(ids.index(question.gold) if question.gold is not None else -100)
        for i in range(count):
            description = question.candidates[i].description if i < len(question.candidates) else "padding"
            branches.append([CLS, *q, SEP, *text_bytes(description, 128, "description")])
    return {
        "state": pad_bytes(states, device),
        "branches": pad_bytes(branches, device),
        "kinds": torch.tensor(kinds, device=device),
        "counts": torch.tensor(counts, device=device),
        "labels": torch.tensor(labels, device=device),
        "max_candidates": count,
    }


@dataclass(frozen=True)
class ChoiceConfig:
    width: int = 64
    heads: int = 4
    state_layers: int = 2
    branch_layers: int = 1
    dropout: float = 0.0
    state_limit: int = 512
    question_limit: int = 128
    candidate_limit: int = 128


@dataclass(frozen=True)
class StateMemory:
    key: str
    values: torch.Tensor
    padding: torch.Tensor


class ByteEncoder(nn.Module):
    def __init__(self, config, length, layers):
        super().__init__()
        self.bytes = nn.Embedding(259, config.width, padding_idx=PAD)
        self.positions = nn.Embedding(length, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, tokens):
        length = tokens.shape[1]
        embedded = self.bytes(tokens) + self.positions(torch.arange(length, device=tokens.device))
        return self.norm(self.layers(embedded, src_key_padding_mask=tokens.eq(PAD)))


class ChoiceModel(nn.Module):
    def __init__(self, config=ChoiceConfig()):
        super().__init__()
        if (
            config.dropout != 0
            or config.state_limit != 512
            or config.question_limit != 128
            or config.candidate_limit != 128
        ):
            raise ValueError("C-v2 固定 byte limits 且 dropout=0")
        self.config = config
        width = config.width
        self.state_encoder = ByteEncoder(config, 513, config.state_layers)
        self.branch_encoder = ByteEncoder(config, 258, config.branch_layers)
        self.types = nn.Embedding(3, width)
        self.readers = nn.ModuleList(
            nn.MultiheadAttention(width, config.heads, dropout=0.0, batch_first=True) for _ in range(2)
        )
        self.reader_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.score = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.none = nn.Sequential(nn.Linear(width * 2 + 3, width), nn.GELU(), nn.Linear(width, 1))
        self._cache = {}

    def encode_state(self, state):
        return self.state_encoder(state)

    def forward(self, batch, recompute=False, memory=None):
        states, branch_tokens = batch["state"], batch["branches"]
        b, candidates = len(states), batch["max_candidates"]
        branches = (
            self.branch_encoder(branch_tokens)[:, :1]
            + self.types(batch["kinds"].repeat_interleave(candidates))[:, None]
        )
        if memory is not None:
            if self.training or recompute or b != 1:
                raise ValueError("外部 state cache 僅供單題 frozen eval")
            state_text = bytes(states[0, 1:][states[0, 1:] != PAD].tolist()).decode("utf-8", errors="strict")
            if memory.key != self.memory_key(state_text):
                raise ValueError("state cache 身分與本題／encoder 不符")
            state_values, padding = memory.values, memory.padding
            expanded = state_values.expand(candidates, -1, -1)
            pad = padding.expand(candidates, -1)
        elif recompute:
            # 真正重算 state-only encoder；不是 clone 已算 M 的假控制。
            expanded = self.encode_state(states.repeat_interleave(candidates, dim=0))
            pad = states.eq(PAD).repeat_interleave(candidates, dim=0)
        else:
            expanded = self.encode_state(states).repeat_interleave(candidates, dim=0)
            pad = states.eq(PAD).repeat_interleave(candidates, dim=0)
        for reader, norm in zip(self.readers, self.reader_norms):
            branches = norm(
                branches + reader(branches, expanded, expanded, key_padding_mask=pad, need_weights=False)[0]
            )
        hidden = branches[:, 0].reshape(b, candidates, -1)
        logits = self.score(hidden).squeeze(-1)
        valid = torch.arange(candidates, device=states.device)[None] < batch["counts"][:, None]
        mean = (hidden * valid[..., None]).sum(1) / batch["counts"][:, None]
        maximum = hidden.masked_fill(~valid[..., None], -torch.inf).max(1).values
        score_mean = (logits * valid).sum(1) / batch["counts"]
        score_max = logits.masked_fill(~valid, -torch.inf).max(1).values
        summary = torch.cat(
            (mean, maximum, score_mean[:, None], score_max[:, None], batch["counts"][:, None].float() / 7), dim=-1
        )
        none = self.none(summary).squeeze(-1)
        # 每題 none 放在自己的 count 位置，padding logits 保持 -inf。
        result = torch.full((b, candidates + 1), -torch.inf, device=states.device, dtype=logits.dtype)
        result[:, :candidates] = logits.masked_fill(~valid, -torch.inf)
        none = none.masked_fill(batch["kinds"] != TYPES["choice"], -torch.inf)
        result.scatter_(1, batch["counts"][:, None], none[:, None])
        return {"logits": result, "candidate_hidden": hidden, "memory": expanded}

    def memory_key(self, state):
        raw = text_bytes(state, 512, "state")
        digest = hashlib.sha256(raw)
        metadata = {
            "config": self.config.__dict__,
            "preprocessing": PREPROCESSING,
            "pad": PAD,
            "positions": "learned absolute, state starts0",
            "dtype": str(next(self.parameters()).dtype),
            "device": str(next(self.parameters()).device),
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for name, tensor in self.state_encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def state_memory(self, state):
        if self.training or next(self.parameters()).dtype != torch.float32:
            raise ValueError("快取只允許 frozen fp32 eval")
        key = self.memory_key(state)
        if key not in self._cache:
            device = next(self.parameters()).device
            tokens = pad_bytes([[CLS, *text_bytes(state, 512, "state")]], device)
            if len(self._cache) >= 64:
                self._cache.clear()
            self._cache[key] = StateMemory(key, self.encode_state(tokens), tokens.eq(PAD))
        cached = self._cache[key]
        # 呼叫者持有自己的副本；不能經公開 tensor 意外污染內部快取。
        return StateMemory(cached.key, cached.values.clone(), cached.padding.clone())

    @torch.no_grad()
    def predict(self, question, temperature=1.0, use_cache=True, memory=None):
        question.validate()
        if self.training or temperature <= 0:
            raise ValueError("推論須 eval 且 temperature > 0")
        if memory is not None and memory.key != self.memory_key(question.state):
            raise ValueError("state／encoder／tokenizer／dtype/device 已變，舊 cache 失效")
        if use_cache and memory is None:
            memory = self.state_memory(question.state)
        batch = collate_questions([question], next(self.parameters()).device)
        result = self(batch, memory=memory)
        ids = [candidate.id for candidate in question.candidates] + (["none"] if question.kind == "choice" else [])
        logits = result["logits"][0, : len(ids)]
        # raw logits 決平手，再按 stable ID；不讓 softmax 浮點捨入改答案。
        best = min(range(len(ids)), key=lambda i: (-float(logits[i]), ids[i]))
        probabilities = F.softmax(logits / temperature, dim=-1)
        return {
            "answer": ids[best],
            "ids": ids,
            "logits": logits.cpu().tolist(),
            "probabilities": probabilities.cpu().tolist(),
            "p_max": float(probabilities.max()),
            "candidate_hidden": result["candidate_hidden"][0].cpu(),
        }
