"""Train-side character vocabulary and program-inserted roles for the finite assistant."""

import hashlib
import json
from pathlib import Path

SPECIAL_NAMES = ("pad", "bos", "eos", "user", "assistant", "system", "tool", "image", "ocr", "audio", "unk")
SPECIALS = tuple(f"<{name}>" for name in SPECIAL_NAMES)
OCR_CHARACTERS = "大小上下左右開關入出人口"


class CharacterTokenizer:
    """Vocabulary is a sorted train-only alphabet; literal role spellings stay ordinary text."""

    def __init__(self, characters):
        characters = list(characters)
        if any(not isinstance(char, str) or len(char) != 1 for char in characters):
            raise ValueError("The vocabulary must contain individual Unicode characters")
        if characters != sorted(set(characters)):
            raise ValueError("Character vocabulary must be sorted and unique")
        self.characters = characters
        self.vocab_size = len(SPECIALS) + len(characters)
        self.special_ids = set(range(len(SPECIALS)))
        self.control_ids = set(range(len(SPECIALS) - 1))
        self.character_ids = {char: i + len(SPECIALS) for i, char in enumerate(characters)}
        for index, name in enumerate(SPECIAL_NAMES):
            setattr(self, f"{name}_id", index)

    @classmethod
    def build(cls, texts, required_chars=OCR_CHARACTERS + "0123456789"):
        alphabet = set(required_chars)
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Tokenizer construction accepts train-side strings only")
            alphabet.update(text)
        return cls(sorted(alphabet))

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer content must be a string")
        return [self.character_ids.get(char, self.unk_id) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for token in ids:
            token = int(token)
            if not 0 <= token < self.vocab_size:
                raise ValueError("Token ID outside this tokenizer's vocabulary")
            if token == self.unk_id:
                pieces.append("�")
            elif token in self.control_ids:
                if not skip_special_tokens:
                    pieces.append(SPECIALS[token])
            else:
                pieces.append(self.characters[token - len(SPECIALS)])
        return "".join(pieces)

    def to_dict(self):
        return {"type": "selftrained_char_v1", "specials": list(SPECIALS), "characters": self.characters.copy()}

    @classmethod
    def from_dict(cls, definition):
        if definition.get("type") != "selftrained_char_v1" or definition.get("specials") != list(SPECIALS):
            raise ValueError("Tokenizer type or fixed role IDs differ from the selftrained contract")
        return cls(definition["characters"])

    @property
    def sha256(self):
        content = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def generation_report(tokenizer, ids):
    """Do not silently discard generated role markers and present them as valid answers."""
    ids = [int(token) for token in ids]
    invalid = [
        {"position": i, "id": token, "token": SPECIALS[token]}
        for i, token in enumerate(ids)
        if token in tokenizer.special_ids and token != tokenizer.eos_id
    ]
    answer_ids = ids[: ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids
    return {
        "answer": tokenizer.decode(answer_ids, skip_special_tokens=False),
        "generated_ids": ids,
        "eos": tokenizer.eos_id in ids,
        "invalid_special_tokens": invalid,
        "valid_answer_tokens": not invalid,
    }
