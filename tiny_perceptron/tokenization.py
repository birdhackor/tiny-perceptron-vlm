"""推論時配對已訓練的 tokenizer；角色標記只能由程式插入。"""

import hashlib
import json
from pathlib import Path

from tiny_perceptron.data import SPECIALS, ByteTokenizer

ROLE_NAMES = ("pad", "bos", "eos", "user", "assistant", "image", "audio", "system")


class ByteLevelBPE:
    """與課程 BPE 實驗相同：移除 added-token 解析，保留原本詞表與 merges。"""

    def __init__(self, path):
        from tokenizers import Tokenizer, pre_tokenizers

        definition = json.loads(Path(path).read_text(encoding="utf-8"))
        if definition.get("model", {}).get("type") != "BPE":
            raise ValueError("這個入口只支援課程的 ByteLevel BPE JSON")
        if (definition.get("pre_tokenizer") or {}).get("type") != "ByteLevel" or (definition.get("decoder") or {}).get(
            "type"
        ) != "ByteLevel":
            raise ValueError("BPE 需要 ByteLevel 前處理與 decoder，不能猜測其他 tokenizer 的規則")
        if definition.get("normalizer") is not None or definition["pre_tokenizer"].get("add_prefix_space"):
            raise ValueError("課程 BPE 不改寫文字或自動補空格；請使用訓練時的 JSON")
        self.tokenizer = Tokenizer.from_str(json.dumps(definition))
        self.vocab_size = self.tokenizer.get_vocab_size()
        vocabulary = self.tokenizer.get_vocab()
        if set(vocabulary.values()) != set(range(self.vocab_size)):
            raise ValueError("tokenizer ID 必須連續且唯一")
        for index, (name, spelling) in enumerate(zip(ROLE_NAMES, SPECIALS, strict=True)):
            if self.tokenizer.token_to_id(spelling) != index:
                raise ValueError("課程特殊 ID 必須依序固定為 0 到 7")
            setattr(self, f"{name}_id", index)
        self.special_ids = set(range(len(SPECIALS)))
        if any(
            self.tokenizer.token_to_id(char) is None or self.tokenizer.token_to_id(char) < 8
            for char in pre_tokenizers.ByteLevel.alphabet()
        ):
            raise ValueError("BPE 缺少完整 256-byte 字母表，無法保證未見字與 emoji 的還原")
        definition["added_tokens"] = []
        self.content = Tokenizer.from_str(json.dumps(definition))
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def encode(self, text):
        ids = self.content.encode(text, add_special_tokens=False).ids
        if any(value in self.special_ids for value in ids):
            raise ValueError("普通文字被編成控制 ID；此 tokenizer 不符合課程的內容／角色邊界")
        return ids

    def decode(self, ids):
        return self.content.decode(
            [int(value) for value in ids if int(value) not in self.special_ids], skip_special_tokens=False
        )


def load_tokenizer(path, vocab_size, checkpoint=None):
    """有 hash／特殊 ID 綁定時檢查它們；沒有 BPE 檔就不猜未知詞表。"""
    checkpoint = checkpoint or {}
    metadata = checkpoint.get("metadata", {})
    binding = checkpoint.get("tokenizer", {})
    if not isinstance(binding, dict):
        raise ValueError("checkpoint tokenizer 設定必須是 dict")
    if path is None:
        if vocab_size != ByteTokenizer.vocab_size:
            raise ValueError("這個模型不是 264 個 byte ID；請用 --tokenizer 指定訓練時的 BPE JSON")
        tokenizer = ByteTokenizer()
        digest = None
        kind = "byte"
    else:
        definition = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if definition.get("type") == "byte":
            if definition.get("specials") != list(SPECIALS):
                raise ValueError("byte tokenizer 的特殊標記與課程不相容")
            tokenizer, kind = ByteTokenizer(), "byte"
        else:
            tokenizer, kind = ByteLevelBPE(path), "bpe"
    if tokenizer.vocab_size != vocab_size or ("vocab_size" in binding and binding["vocab_size"] != vocab_size):
        raise ValueError("tokenizer 詞表大小與 checkpoint config 不相容")
    declared = binding.get("type", "unspecified")
    if declared not in ("unspecified", "byte", "bpe") or (declared != "unspecified" and declared != kind):
        raise ValueError("checkpoint 與 tokenizer 類型不相容")
    trained = metadata.get("tokenizer")
    if isinstance(trained, str) and (
        (trained.startswith("bpe") and kind != "bpe") or (trained.startswith("byte") and kind != "byte")
    ):
        raise ValueError("tokenizer 與訓練 metadata 指定的類型不相容")
    for expected in (
        binding.get("sha256"),
        binding.get("tokenizer_sha256"),
        metadata.get("tokenizer_sha256"),
        metadata.get("tokenizer_file_sha256"),
        checkpoint.get("tokenizer_sha256"),
    ):
        if expected is not None and expected != digest:
            raise ValueError("tokenizer SHA-256 與 checkpoint 不相容；請提供同一份 --tokenizer 檔")
    if "tokenizer_vocab_size" in metadata and metadata["tokenizer_vocab_size"] != vocab_size:
        raise ValueError("訓練 metadata 的 tokenizer 詞表大小不相容")
    for source in (binding, metadata, checkpoint):
        if "specials" in source and source["specials"] != list(SPECIALS):
            raise ValueError("checkpoint 特殊標記與 tokenizer 不相容")
        for field in ("special_ids", "special_token_ids"):
            if field not in source:
                continue
            expected = source[field]
            actual = dict(zip(SPECIALS, range(8), strict=True)) if isinstance(expected, dict) else list(range(8))
            if isinstance(expected, dict) and set(expected) == set(ROLE_NAMES):
                actual = dict(zip(ROLE_NAMES, range(8), strict=True))
            if expected != actual:
                raise ValueError("checkpoint 特殊 ID 與 tokenizer 不相容")
    return tokenizer


def generation_report(tokenizer, ids):
    """非法控制 ID 留在可見答案與報告，不能被 decoder 丟掉後看似成功。"""
    ids = [int(value) for value in ids]
    if any(value < 0 or value >= tokenizer.vocab_size for value in ids):
        raise ValueError("生成 ID 超出 tokenizer 詞表")
    pieces, content, invalid = [], [], []
    for position, value in enumerate(ids):
        if value >= len(SPECIALS):
            content.append(value)
            continue
        if content:
            pieces.append(tokenizer.decode(content))
            content = []
        if value != tokenizer.eos_id:
            pieces.append(SPECIALS[value])
            invalid.append({"position": position, "id": value, "token": SPECIALS[value]})
    if content:
        pieces.append(tokenizer.decode(content))
    eos = tokenizer.eos_id in ids
    return {
        "answer": "".join(pieces),
        "generated_ids": ids,
        "eos": eos,
        "invalid_special_tokens": invalid,
        "valid_answer_tokens": not invalid,
        "generation_status": "invalid_special_tokens" if invalid else "eos" if eos else "token_or_context_limit",
    }
