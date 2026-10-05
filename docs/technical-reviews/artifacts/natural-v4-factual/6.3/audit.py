"""Independent, bounded CPU audit of the current 6.3 text and original run record."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import sys
from collections import Counter

import torch
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
TEMP = ROOT / "outputs/natural-v4/factual-research/6.3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def first_rows(path, count):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(next(handle)) for _ in range(count)]


def make(corpus, budget=280):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=budget, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    ))
    return tok


record = json.loads((ROOT / "docs/course-experiments/results/tokenizer.json").read_text())
report = {
    "environment": {"python": sys.version, "torch": torch.__version__,
                    "tokenizers": tokenizers.__version__, "device": "cpu",
                    "cuda_available": torch.cuda.is_available(), "platform": platform.platform()},
    "scope": "No GPU training or benchmark replication; original recorded artifacts and bounded CPU behavior only.",
}
chapter = (ROOT / "course/chapters/06.md").read_text()
section = re.search(r"^## 6\.3\s.*?(?=^## |\Z)", chapter, re.M | re.S).group()
report["source_sha256"] = sha(section.encode())
assert report["source_sha256"] == "24b4be7c602de361dd439ec7a60000fecc53af6c039db22a5a123a32db677eb5"
code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
buf = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(buf):
    exec(compile(code, "course/chapters/06.md#6.3", "exec"), namespace)
demo = namespace["tok"]
report["exact_demo"] = {"stdout": buf.getvalue(), "vocabulary_size": demo.get_vocab_size(),
                        "alphabet_size": len(pre_tokenizers.ByteLevel.alphabet()),
                        "normalizer": json.loads(demo.to_str())["normalizer"]}
assert len(set(pre_tokenizers.ByteLevel.alphabet())) == 256
assert all(demo.token_to_id(c) is not None for c in pre_tokenizers.ByteLevel.alphabet())
checks = []
for text in ["小鳥🙂new", "未見字🦊", " 小鳥🙂\nnew ", "Ａ e\u0301\t\x00", "", "𐀀\U0010ffff"]:
    encoded = demo.encode(text)
    restored = demo.decode(encoded.ids)
    assert restored == text
    checks.append({"text": text, "ids": encoded.ids, "tokens": encoded.tokens, "restored": restored})
report["demo_roundtrips"] = checks
demo.save(str(TEMP / "demo-tokenizer.json"))
loaded = Tokenizer.from_file(str(TEMP / "demo-tokenizer.json"))
assert json.loads(loaded.to_str()) == json.loads(demo.to_str())
assert all(loaded.decode(loaded.encode(x["text"]).ids) == x["text"] for x in checks)
report["serialization"] = {"definition_equal": True, "roundtrips": len(checks),
                           "saved_sha256": sha((TEMP / "demo-tokenizer.json").read_bytes())}
report["budget_probes"] = {"target_280_single_a_actual": make(["a"]).get_vocab_size(),
                          "target_100_full_alphabet_actual": make(["a"], 100).get_vocab_size()}
left, right = make(["abababab"], 257), make(["cdcdcdcd"], 257)
assert left.get_vocab_size() == right.get_vocab_size() == 257
assert left.id_to_token(256) != right.id_to_token(256)
report["same_size_id_difference"] = {"vocabularies": [257, 257],
                                    "token_at_256": [left.id_to_token(256), right.id_to_token(256)]}

report["original_record"] = {"path": "docs/course-experiments/results/tokenizer.json",
    "sha256": sha((ROOT / "docs/course-experiments/results/tokenizer.json").read_bytes()),
    "revision": record["revision"], "seed": record["seed"], "device": record["device"],
    "python": record["python_version"], "torch": record["torch_version"],
    "run_id": record["modal"]["run_id"]}
report["code_checksums"] = {}
for rel in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py"]:
    digest = sha((ROOT / rel).read_bytes())
    assert digest == record["code_sha256"][rel]
    report["code_checksums"][rel] = {"sha256": digest, "equals_original_run": True}

raw = []
report["source_assets"] = {}
for name, count, asset in [("tinystories-train-512.jsonl", 96, "tinystories"),
                            ("chinese-classical-train-365.jsonl", 96, "chinese-poetry")]:
    rel = "data/training/text-initial/" + name
    path = ROOT / rel
    original = next(x for a in record["assets"] if a["id"] == asset
                    for x in a["files"] if x["path"] == "text-initial/" + name)
    digest = sha(path.read_bytes())
    assert digest == original["sha256"]
    rows = first_rows(path, count)
    assert all(sha(row["text"].encode()) == row["text_sha256"] for row in rows)
    if asset == "chinese-poetry":
        assert all(row["text"] == "\n".join([row["title"], row["author"], *row["paragraphs"]]) for row in rows)
    report["source_assets"][rel] = {"sha256": digest, "equals_original_run": True,
        "selected": len(rows), "languages": dict(Counter(row["language"] for row in rows)),
        "complete_text_hashes_match": True, "poem_title_author_headers_checked": asset == "chinese-poetry"}
    raw.extend(rows)

# Independently express the generator's normalized complete-text identity and seeded split.
groups = {}
for row in raw:
    normalized = " ".join(row["text"].split())
    family = sha(normalized.encode())
    assert family not in groups  # The chosen first 96 + 96 have no removed duplicates.
    groups[family] = row
keys = sorted(groups)
random.Random(42).shuffle(keys)
a, b = int(len(keys) * 0.8), int(len(keys) * 0.9)
assert (len(raw), len(groups), a, b - a, len(keys) - b) == (192, 192, 153, 19, 20)
report["split_derivation"] = {"seed": 42, "selected_complete_documents": 192,
    "deduplicated_families": 192, "floor_192_times_0_8": a, "floor_192_times_0_9": b,
    "counts": {"train": a, "validation": b - a, "test": len(keys) - b},
    "method": "Complete text normalized by whitespace for family SHA; sorted keys shuffled with Python Random(42); 80/90 percent integer cutpoints before byte cropping."}
stored_parts = {}
report["stored_splits"] = {}
for split, selected in [("train", keys[:a]), ("validation", keys[a:b]), ("test", keys[b:])]:
    rel = "outputs/text-behavior-interface-check/tokenizer/data/" + split + ".jsonl"
    data = (ROOT / rel).read_bytes()
    digest = sha(data)
    assert digest == record["results"]["data"][split]["sha256"]
    rows = [json.loads(x) for x in data.splitlines()]
    assert [row["family"] for row in rows] == selected
    discarded = Counter()
    for saved, family in zip(rows, selected, strict=True):
        original = groups[family]
        raw_bytes = original["text"].encode()
        expected_text = raw_bytes[:256].decode("utf-8", errors="ignore")
        expected = {**original, "family": family, "text": expected_text,
                    "complete_text_sha256": sha(raw_bytes)}
        assert saved == expected
        prefix_bytes = saved["text"].encode()
        assert raw_bytes.startswith(prefix_bytes) and len(prefix_bytes) <= 256
        prefix_bytes.decode("utf-8", errors="strict")
        discarded[len(raw_bytes[:256]) - len(prefix_bytes)] += 1
    stored_parts[split] = rows
    report["stored_splits"][split] = {"path": rel, "sha256": digest, "equals_original_run": True,
        "records": len(rows), "families": len(set(selected)), "all_exact_generator_rows_match": True,
        "raw_utf8_bytes": sum(len(row["text"].encode()) for row in rows),
        "maximum_excerpt_bytes": max(len(row["text"].encode()) for row in rows),
        "incomplete_suffix_bytes_removed_histogram": dict(discarded),
        "languages": dict(Counter(row["language"] for row in rows))}
sets = [set(row["family"] for row in rows) for rows in stored_parts.values()]
assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
report["cross_split_family_overlap"] = 0

path = ROOT / "checkpoints/course/tokenizer/tokenizer-bpe512.json"
expected_sha = next(x["sha256"] for x in record["artifacts"] if x["path"] == "tokenizer-bpe512.json")
assert sha(path.read_bytes()) == expected_sha
definition = json.loads(path.read_text())
original_tok = Tokenizer.from_file(str(path))
specials = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>"]
assert original_tok.get_vocab_size() == 512
assert [original_tok.token_to_id(x) for x in specials] == list(range(8))
assert len(definition["model"]["merges"]) == 512 - 256 - 8
assert definition["normalizer"] is None
assert definition["pre_tokenizer"]["type"] == definition["decoder"]["type"] == "ByteLevel"
assert definition["pre_tokenizer"]["add_prefix_space"] is False
definition["added_tokens"] = []  # Same content/role boundary used by original experiment _BPE.
content = Tokenizer.from_str(json.dumps(definition))
roundtrips = []
for original in record["results"]["roundtrip"]:
    ids = content.encode(original["text"], add_special_tokens=False).ids
    restored = content.decode(ids, skip_special_tokens=False)
    assert ids == original["ids"] and restored == original["text"] == original["restored"]
    assert not set(ids) & set(range(8))
    roundtrips.append({"text": original["text"], "ids": ids, "restored": restored,
                      "same_as_original_record": True, "contains_control_id": False})
assert all(content.decode(content.encode(row["text"]).ids) == row["text"]
           for rows in stored_parts.values() for row in rows)
report["original_tokenizer"] = {"path": str(path.relative_to(ROOT)), "sha256": expected_sha,
    "equals_original_run": True, "vocab_size": 512, "byte_alphabet_size": 256,
    "special_ids": dict(zip(specials, range(8))), "merge_count": 248,
    "record_roundtrips": roundtrips, "all_stored_excerpt_roundtrips": 192,
    "training_input_audit": "Source generator passes only parts['train'] (153 excerpts) to train_from_iterator; validation/test remain excluded."}
report["result"] = "All assertions passed. Original run input identity, generator split/crop, saved tokenizer identity and exact recorded roundtrips verified."
(OUT / "cpu-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
