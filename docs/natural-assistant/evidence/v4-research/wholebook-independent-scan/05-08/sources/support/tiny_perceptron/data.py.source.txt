"""文字、對話與 batch：先看一筆，再合成多筆。"""

import hashlib
import json
import random
from pathlib import Path

import torch

IGNORE = -100
SPECIALS = ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>")


class ByteTokenizer:
    """不學切詞的可還原起點；每個 UTF-8 byte 對應一個 ID。"""

    vocab_size = 264
    pad_id, bos_id, eos_id, user_id, assistant_id, image_id, audio_id, system_id = range(8)

    def encode(self, text):
        return [b + len(SPECIALS) for b in text.encode("utf-8")]

    def decode(self, ids):
        raw = bytes(i - len(SPECIALS) for i in ids if i >= len(SPECIALS))
        return raw.decode("utf-8", errors="replace")

    def state(self):
        return {"type": "byte", "specials": list(SPECIALS)}


class CharTokenizer:
    """早期字元模型；未見字元用獨立 UNK，沒有無損還原保證。"""

    def __init__(self, text):
        self.chars = ["<unk>"] + sorted(set(text))
        self.to_id = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def shifted(ids):
    """只 shift 一次：輸入第 t 格猜原序列的第 t+1 格。"""
    if len(ids) < 2:
        raise ValueError("至少需要兩個 token 才有下一字目標")
    ids = torch.tensor(ids, dtype=torch.long)
    return ids[:-1], ids[1:]


def render_chat(messages, tokenizer=None):
    """邊界由角色欄位建立；使用者文字不會被解析為特殊 token。"""
    tok = tokenizer or ByteTokenizer()
    ids, targets = [tok.bos_id], [IGNORE]
    roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
    for message in messages:
        role = message["role"]
        if role not in roles or not isinstance(message["content"], str):
            raise ValueError("role 必須是 user/assistant/system，content 必須是文字")
        content = tok.encode(message["content"]) + [tok.eos_id]
        ids += [roles[role]] + content
        targets += [IGNORE] + (content if role == "assistant" else [IGNORE] * len(content))
    if not any(t != IGNORE for t in targets):
        raise ValueError("這筆對話没有可監督的 assistant 答案")
    return torch.tensor(ids[:-1]), torch.tensor(targets[1:])


def pad_batch(examples, pad_id=0, max_length=None):
    """右側 padding；attention 的有效位置和 loss 的有效位置各自保留。"""
    if not examples:
        raise ValueError("batch 不能是空的")
    length = max(len(x) for x, _ in examples)
    if max_length is not None:
        length = min(length, max_length)
    x = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    valid = torch.zeros_like(x, dtype=torch.bool)
    for row, (inputs, labels) in enumerate(examples):
        n = min(length, len(inputs))
        x[row, :n], y[row, :n], valid[row, :n] = inputs[:n], labels[:n], True
        if not (y[row] != IGNORE).any():
            raise ValueError(f"第 {row} 筆截斷後没有有效答案；增加 max_length 或移除這筆")
    return x, y, valid


def split_documents(documents, seed=42):
    """先去完全重複，再依完整文件切分；真實語料還要處理近重複家族。"""
    docs = list(dict.fromkeys(documents))
    random.Random(seed).shuffle(docs)
    n = len(docs)
    if n < 3:
        raise ValueError("至少三份不同文件，才能示範 train/validation/test")
    a, b = max(1, int(n * 0.8)), max(2, int(n * 0.9))
    b = min(b, n - 1)
    a = min(a, b - 1)
    return {"train": docs[:a], "validation": docs[a:b], "test": docs[b:]}


def fingerprint(text):
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def toy_documents():
    """課程自建的規則小世界；先學介面，不冒充自然語言能力。"""
    return [f"顏色={color}；形狀={shape}。" for color in ("紅", "綠", "藍", "黃") for shape in ("圓", "方", "三角")]


def toy_conversations():
    return [
        [{"role": "user", "content": f"{a}+{b}=?"}, {"role": "assistant", "content": str(a + b)}]
        for a in range(4)
        for b in range(4)
    ]
