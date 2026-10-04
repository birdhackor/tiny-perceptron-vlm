"""Independent CPU tokenizer checks; no model inference, updates, or finaltest access."""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from scripts.course_experiments.common import split_records
from tiny_perceptron.data import SPECIALS
from tiny_perceptron.tokenization import ByteLevelBPE, load_tokenizer

guard_events = []


def guard(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto", "subprocess.Popen", "os.system"}:
        guard_events.append(event)
        raise PermissionError(event)
    if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(arguments[0])).resolve()
        if "finaltest" in str(path).lower() or path.name == "test.jsonl":
            guard_events.append("forbidden held-out read")
            raise PermissionError("No finaltest or test.jsonl access in this probe")
        flags = arguments[2]
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if not path.is_relative_to(HERE):
                guard_events.append("outside-artifact write")
                raise PermissionError("Writes must remain inside the replacement review artifact directory")


sys.addaudithook(guard)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


section = (HERE / "6.3-current-raw.md").read_bytes()
fence = re.search(rb"```python\n(.*?)\n```", section, re.S).group(1) + b"\n"
(HERE / "original-fence.py").write_bytes(fence)


def execute(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "6.3-original-fence", "exec"), namespace)
    return namespace, stdout.getvalue()


demo, original_stdout = execute(fence)
exercise_code = fence.replace('text = "小鳥🙂new"'.encode(), 'text = "未見字🦊"'.encode())
assert exercise_code != fence
(HERE / "exercise-fence.py").write_bytes(exercise_code)
exercise, exercise_stdout = execute(exercise_code)
tok = demo["tok"]
alphabet = set(pre_tokenizers.ByteLevel.alphabet())
definition = json.loads(tok.to_str())
assert len(alphabet) == 256 and alphabet <= tok.get_vocab().keys()
assert tok.get_vocab_size() == 277 and len(definition["model"]["merges"]) == 21
assert definition["normalizer"] is None and definition["pre_tokenizer"]["add_prefix_space"] is False
assert definition["truncation"] is None and definition["post_processor"] is None
saved = HERE / "demo-saved.json"
tok.save(str(saved))
restored_tok = Tokenizer.from_file(str(saved))
demo_roundtrips = []
for text in ["小鳥🙂new", "未見字🦊", " 小鳥🙂\nnew ", "Ａ A é e\u0301", "<user>只是原文"]:
    enc = tok.encode(text)
    restored = tok.decode(enc.ids)
    assert restored == text and restored_tok.encode(text).ids == enc.ids
    demo_roundtrips.append({"text": text, "ids": enc.ids, "tokens": enc.tokens, "restored": restored, "same": True})


def train_toy(size, first="貓看狗。"):
    candidate = Tokenizer(models.BPE())
    candidate.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    candidate.decoder = decoders.ByteLevel()
    candidate.train_from_iterator(
        [first, "a cat sees a dog."],
        trainers.BpeTrainer(vocab_size=size, initial_alphabet=list(alphabet), show_progress=False),
    )
    return candidate


budgets = {}
for target in [10, 280, 1000]:
    candidate = train_toy(target)
    budgets[str(target)] = {
        "actual_vocabulary": candidate.get_vocab_size(),
        "merges": len(json.loads(candidate.to_str())["model"]["merges"]),
        "contains_all_bytes": alphabet <= candidate.get_vocab().keys(),
    }
assert budgets["10"]["actual_vocabulary"] == 256
assert budgets["1000"]["actual_vocabulary"] == 277
alternate = train_toy(280, first="貓看兔。")
assert alternate.get_vocab_size() == tok.get_vocab_size()
changed_ids = [i for i in range(tok.get_vocab_size()) if tok.id_to_token(i) != alternate.id_to_token(i)]
assert changed_ids

formal_path = ROOT / "docs/course-experiments/results/tokenizer.json"
formal = json.loads(formal_path.read_bytes())
assert formal["revision"] == "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3"
assert formal["evidence_status"] == "complete_run" and formal["step_scale"] == 1.0
frozen_path = ROOT / "checkpoints/course/tokenizer/tokenizer-bpe512.json"
frozen_sha = sha(frozen_path)
assert frozen_sha == next(a["sha256"] for a in formal["artifacts"] if a["path"] == "tokenizer-bpe512.json")
frozen = ByteLevelBPE(frozen_path)
frozen_definition = json.loads(frozen_path.read_bytes())
assert frozen.vocab_size == 512 and frozen.special_ids == set(range(8))
assert alphabet <= frozen.tokenizer.get_vocab().keys()
assert len(frozen_definition["model"]["merges"]) == 248
frozen_roundtrips = []
for ref in formal["results"]["roundtrip"]:
    ids = frozen.encode(ref["text"])
    restored = frozen.decode(ids)
    assert ids == ref["ids"] and restored == ref["text"]
    assert not set(ids) & frozen.special_ids
    frozen_roundtrips.append({"text": ref["text"], "ids": ids, "restored": restored, "same": True, "matches_formal_ids": True})
binding = {"tokenizer": {"vocab_size": 512, "sha256": frozen_sha, "specials": list(SPECIALS)}}
loaded = load_tokenizer(frozen_path, 512, binding)
assert loaded.encode("未見字🦊 new") == frozen_roundtrips[0]["ids"]
rejections = {}
for name, size, checkpoint in [("wrong_size", 513, {}), ("same_size_wrong_hash", 512, {"metadata": {"tokenizer_sha256": "0" * 64}})]:
    try:
        load_tokenizer(frozen_path, size, checkpoint)
    except ValueError as exc:
        rejections[name] = str(exc)
    else:
        raise AssertionError(name)

# Original inputs are designated training assets. Do not open any saved test.jsonl.
raw = []
source_inputs = {}
for basename in ["tinystories-train-512.jsonl", "chinese-classical-train-365.jsonl"]:
    path = ROOT / "data/training/text-initial" / basename
    rows = [json.loads(line) for line in path.read_text().splitlines()][:96]
    assert len(rows) == 96
    raw += rows
    source_inputs[basename] = {"sha256": sha(path), "selected_rows": len(rows)}
unique, seen = [], set()
for row in raw:
    key = " ".join(row["text"].split())
    if key not in seen:
        unique.append({**row, "family": hashlib.sha256(key.encode()).hexdigest()})
        seen.add(key)
assert len(unique) == 192
complete = split_records(unique, seed=42)
split_counts = {name: len(rows) for name, rows in complete.items()}
assert split_counts == {"train": 153, "validation": 19, "test": 20}
families = {name: {row["family"] for row in rows} for name, rows in complete.items()}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
split_checks = {}
for name in ["train", "validation"]:
    path = ROOT / "outputs/text-behavior-interface-check/tokenizer/data" / f"{name}.jsonl"
    actual_rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = formal["results"]["data"][name]
    assert sha(path) == expected["sha256"] and len(actual_rows) == expected["records"]
    transformed = [
        {**row, "text": row["text"].encode()[:256].decode("utf-8", errors="ignore"),
         "complete_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
        for row in complete[name]
    ]
    assert actual_rows == transformed
    assert all(len(row["text"].encode()) <= 256 for row in actual_rows)
    split_checks[name] = {"records": len(actual_rows), "sha256": sha(path), "maximum_bytes": max(len(r["text"].encode()) for r in actual_rows), "matches_reconstructed_full_family_prefixes": True}

before = [("abab", "abac")]
pairs = Counter(pair for row in before[0] for pair in zip(row, row[1:]))
assert pairs == Counter({("a", "b"): 3, ("b", "a"): 2, ("a", "c"): 1})
bpb = {}
for name, run in formal["results"]["runs"].items():
    aggregate = run["after"]["test"]
    value = aggregate["mean_token_nll"] * aggregate["effective_tokens"] / (aggregate["raw_utf8_bytes"] * math.log(2))
    assert abs(value - aggregate["bpb_including_eos_boundary_targets"]) < 1e-12
    bpb[name] = {"mean_token_nll": aggregate["mean_token_nll"], "targets": aggregate["effective_tokens"], "raw_bytes": aggregate["raw_utf8_bytes"], "recomputed_bpb": value}

result = {
    "reviewer_task": "/root/natural_dependency_factual_6_3",
    "environment": {"python": sys.version, "tokenizers": tokenizers.__version__, "torch_package": importlib.metadata.version("torch"), "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
    "original": {"fence_sha256": hashlib.sha256(fence).hexdigest(), "stdout": original_stdout, "assert_passed": True, "permanent_save_in_original": False},
    "exercise": {"stdout": exercise_stdout, "assert_passed": True, "tokens": len(exercise["encoded"].ids)},
    "demo": {"vocabulary": tok.get_vocab_size(), "merges": len(definition["model"]["merges"]), "alphabet_count": len(alphabet), "normalizer": definition["normalizer"], "pre_tokenizer": definition["pre_tokenizer"], "roundtrips": demo_roundtrips, "save_reload_ids_equal": True},
    "budgets": budgets,
    "same_size_changed_meanings": {"vocabulary": alternate.get_vocab_size(), "changed_ids": changed_ids},
    "formal": {"path": str(formal_path.relative_to(ROOT)), "sha256": sha(formal_path), "revision": formal["revision"], "source_inputs": source_inputs, "selected_source_counts": dict(Counter(row["source"] for row in raw)), "unique_families": len(unique), "split_counts": split_counts, "split_checks": split_checks, "test_metadata_only": formal["results"]["data"]["test"], "test_jsonl_read": False, "tokenizer_sha256": frozen_sha, "vocabulary": frozen.vocab_size, "merges": len(frozen_definition["model"]["merges"]), "special_ids": sorted(frozen.special_ids), "roundtrips": frozen_roundtrips, "binding_success": True, "binding_rejections": rejections},
    "prerequisite_math": {"pairs": {"".join(k): v for k, v in pairs.items()}, "bpb_from_published_aggregates": bpb},
    "guard_events": guard_events,
    "scope": "Only tiny tokenizer training and deterministic encoding/reloading/binding; no language model creation, inference, gradient updates, GPU, natural-assistant finaltest, or saved course test.jsonl access. Published historical aggregate statistics are read without rerunning the experiment.",
}
assert not guard_events
(HERE / "recheck-execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
