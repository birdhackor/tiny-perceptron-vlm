"""核對分段 BPE、空白、文件標記及 UTF-8 的具體行為。"""

import codecs
import json
import platform
import sys

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    ["hello hello hello"],
    trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]"], show_progress=False),
)
state = json.loads(tok.to_str())
vocab = dict(sorted(tok.get_vocab().items(), key=lambda item: item[1]))
samples = {}
for text in ["hello", "hel", "lo", "hello ", "hello hello", "hello\t\nhello", "x", "貓", "!"]:
    encoded = tok.encode(text)
    samples[text] = {"tokens": encoded.tokens, "ids": encoded.ids, "decoded": tok.decode(encoded.ids)}
assert list(vocab.items()) == [
    ("[UNK]", 0), ("e", 1), ("h", 2), ("l", 3), ("o", 4),
    ("el", 5), ("hel", 6), ("lo", 7), ("hello", 8),
]
assert samples["hello"]["ids"] == [8]
assert samples["hel"]["ids"] + samples["lo"]["ids"] == [6, 7]
assert samples["hello hello"]["ids"] == samples["hello "]["ids"] + samples["hello"]["ids"] == [8, 8]
assert samples["x"]["tokens"] == samples["貓"]["tokens"] == samples["!"]["tokens"] == ["[UNK]"]
assert samples["hello "]["ids"] == samples["hello"]["ids"]


def encode_stream(chunks):
    pending = ""
    ids = []
    for chunk in chunks:
        pending += chunk
        cut = max((i + 1 for i, char in enumerate(pending) if char.isspace()), default=0)
        if cut:
            ids.extend(tok.encode(pending[:cut]).ids)
            pending = pending[cut:]
    ids.extend(tok.encode(pending).ids)
    return ids


stream_checks = []
for text in ["hello hello", "hello\t\nhello", "hellohello", " hello  hello "]:
    for cut in range(len(text) + 1):
        actual = encode_stream([text[:cut], text[cut:]])
        assert actual == tok.encode(text).ids
        stream_checks.append({"text": text, "cut": cut, "ids": actual})

byte_tok = Tokenizer(models.BPE())
byte_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
byte_tok.decoder = decoders.ByteLevel()
byte_tok.train_from_iterator(
    ["貓看狗。", "a cat sees a dog."],
    trainers.BpeTrainer(
        vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ),
)
byte_samples = {}
for text in ["hello", "hello ", "hello hello", " 小鳥🙂\nnew "]:
    encoded = byte_tok.encode(text)
    decoded = byte_tok.decode(encoded.ids)
    assert decoded == text
    byte_samples[text] = {"tokens": encoded.tokens, "ids": encoded.ids, "decoded": decoded}
assert byte_samples["hello"]["ids"] != byte_samples["hello "]["ids"]
byte_splits = byte_tok.pre_tokenizer.pre_tokenize_str("hello hello ")
assert [part for part, _ in byte_splits] == ["hello", "Ġhello", "Ġ"]

docs = ["hel", "lo"]
separate = [tok.encode(doc).ids for doc in docs]
plain_concat = tok.encode("".join(docs)).ids
assert separate == [[6], [7]] and plain_concat == [8]
markers = Tokenizer(models.BPE(unk_token="[UNK]"))
markers.pre_tokenizer = pre_tokenizers.Whitespace()
markers.train_from_iterator(
    ["hello hello hello"],
    trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]", "<bos>", "<eos>"], show_progress=False),
)
marked = [
    [markers.token_to_id("<bos>")] + markers.encode(doc).ids + [markers.token_to_id("<eos>")]
    for doc in docs
]
assert all(ids[0] == markers.token_to_id("<bos>") and ids[-1] == markers.token_to_id("<eos>") for ids in marked)

utf8_checks = []
text = "貓🙂"
raw = text.encode("utf-8")
for cut in range(len(raw) + 1):
    decoder = codecs.getincrementaldecoder("utf-8")()
    first = decoder.decode(raw[:cut])
    buffered = list(decoder.getstate()[0])
    second = decoder.decode(raw[cut:], final=True)
    replaced = raw[:cut].decode("utf-8", errors="replace") + raw[cut:].decode("utf-8", errors="replace")
    assert first + second == text
    if cut not in (0, 3, 7):
        assert replaced != text
    utf8_checks.append({"cut": cut, "first": first, "buffered": buffered, "second": second, "replaced": replaced})

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-9/probe.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "tokenizers": tokenizers.__version__, "device": "cpu"},
    "vocab": vocab,
    "merges": state["model"]["merges"],
    "normalizer": state["normalizer"],
    "pre_tokenizer": state["pre_tokenizer"],
    "byte_fallback": state["model"]["byte_fallback"],
    "samples": samples,
    "safe_stream_checks": stream_checks,
    "bytelevel_splits": byte_splits,
    "bytelevel_samples": byte_samples,
    "document_boundary": {"documents": docs, "separate_ids": separate, "plain_concatenation_ids": plain_concat, "marked_ids": marked},
    "utf8_bytes": list(raw),
    "utf8_checks": utf8_checks,
    "result": "全部斷言通過",
}, ensure_ascii=False, indent=2))
