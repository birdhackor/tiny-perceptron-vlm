"""Recompute fixed project records; never run the recorded GPU training."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import hashlib
import json
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate(entry, records):
    assert len(entry["samples"]) == len(records) == entry["examples"]
    correct = eos = target_tokens = 0
    for item, row in zip(entry["samples"], records, strict=True):
        assert item["family"] == row["family"] and item["target"] == row["answer"]
        ids = item["generated_ids"]
        raw = ids[:ids.index(2)] if 2 in ids else ids
        target = [byte + 8 for byte in row["answer"].encode("utf-8")]
        match = raw == target
        assert match == item["exact_match"]
        assert (2 in ids) == item["eos"]
        correct += match
        eos += 2 in ids
        target_tokens += len(target) + 1
    assert correct == entry["correct"]
    assert correct / len(records) == entry["exact_match"]
    assert eos / len(records) == entry["eos_rate"]
    assert target_tokens == entry["effective_tokens"]
    return {"correct": correct, "examples": len(records), "eos": eos,
            "percent": 100 * correct / len(records), "target_tokens": target_tokens}


summaries = {}
for filename in ["real_modal.json", "audio.json"]:
    path = ROOT / "docs/course-experiments/results" / filename
    original = json.loads(path.read_bytes())
    result = original["results"]["fsdd"] if filename == "real_modal.json" else original["results"]
    assert original["seed"] == result["data"]["seed"] == 42
    inspected_code = ["scripts/course_experiments/modalities.py", "tiny_perceptron/data.py", "tiny_perceptron/multimodal.py"]
    code_hashes = {name: digest(ROOT / name) for name in inspected_code}
    for name, sha in code_hashes.items():
        assert sha == original["code_sha256"][name]
    rows = {}
    for side, entry in result["data"]["splits"].items():
        rows[side] = entry["records"]
        assert len(rows[side]) == entry["count"]
        assert record_hash(rows[side]) == entry["sha256"]
    families = {side: {r["family"] for r in values} for side, values in rows.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not families[a] & families[b]
    if filename == "real_modal.json":
        speakers = {side: sorted({r["family"].split("_")[1] for r in values}) for side, values in rows.items()}
        assert speakers == {"train": ["jackson"], "validation": ["nicolas"], "test": ["theo"]}
        for values in rows.values():
            assert Counter(r["answer"] for r in values) == Counter({str(n): 2 for n in range(10)})
            assert {r["question"] for r in values} == {"digit?"}
        assert len(result["resampling"]) == 60
        for conversion in result["resampling"]:
            assert conversion["source_rate"] == 8000 and conversion["target_rate"] == 16000
            assert conversion["samples_after"] == conversion["samples_before"] * 2
            assert Fraction(conversion["samples_before"], 8000) == Fraction(conversion["samples_after"], 16000)
        first = result["resampling"][0]
        assert first["source"] == "0_jackson_5.wav"
        assert (first["samples_before"], first["samples_after"]) == (4591, 9182)
        assert Fraction(4591, 8000) == Fraction("0.573875")
        extras = {"speakers": speakers, "digits_per_speaker": 10, "versions_per_digit": 2,
                  "resampling_rows": 60, "figure_duration_seconds": float(Fraction(4591, 8000))}
    else:
        for values in rows.values():
            assert {r["question"] for r in values} == {"pitch?"}
            for row in values:
                assert row["answer"] == ("high" if row["frequency"] > 300 else "low")
        extras = {"boundary_hz": 300, "questions": ["pitch?"], "answers": ["low", "high"]}

    history = result["training"]["history"]
    assert result["training"]["steps"] == len(history)
    independently_totalled = 0
    for step, entry in enumerate(history):
        rng = random.Random(42 + step)
        expected_tokens = sum(len(rng.choice(rows["train"])["answer"].encode("utf-8")) + 1 for _ in range(4))
        assert entry["step"] == step + 1 and entry["effective_targets"] == expected_tokens
        independently_totalled += expected_tokens
    assert independently_totalled == result["training"]["effective_targets"]
    summaries[filename] = {
        "original_path": str(path.relative_to(ROOT)), "original_sha256": digest(path),
        "original_run": {k: original[k] for k in ["revision", "seed", "device", "python_version", "torch_version", "gpu", "timing_scope"]},
        "split_counts": {side: len(values) for side, values in rows.items()},
        "training_updates": len(history), "effective_answer_targets": independently_totalled,
        "training_timing_scope": result["training"]["timing_scope"],
        "validation": evaluate(result["validation"], rows["validation"]),
        "test": evaluate(result["test"], rows["test"]),
        "scope": result["scope"], "source_code_sha256": code_hashes, **extras,
    }
assert summaries["real_modal.json"]["validation"]["correct"] == 5
assert summaries["real_modal.json"]["test"]["correct"] == 3
assert summaries["real_modal.json"]["effective_answer_targets"] == 2000
assert summaries["audio.json"]["test"]["correct"] == 11
assert summaries["audio.json"]["effective_answer_targets"] == 5409
print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "method": "Independent CPU recount of original manifests, raw generated token IDs, seeded batch target counts, rates and units; no model inference or training.",
    "results": summaries,
    "limitation": "Original GPU results are inspected and re-counted, not independently regenerated or benchmark-replicated. Source/derived audio payloads are not re-read by this record audit.",
}, indent=2, ensure_ascii=False))
