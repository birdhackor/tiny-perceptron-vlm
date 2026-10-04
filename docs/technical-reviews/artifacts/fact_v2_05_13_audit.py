"""Bounded CPU verification; no training or shared-environment mutation."""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import shutil
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import _nll, records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_05_13_"
torch.set_num_threads(2)
body = dict(sections(ROOT / "course/chapters/05.md"))["5.13"]
(OUT / (PREFIX + "section.txt")).write_text(body, encoding="utf-8")
code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
(OUT / (PREFIX + "snippet.py")).write_text(code, encoding="utf-8")
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, "course/chapters/05.md#5.13", "exec"), {})
plan = stdout.getvalue()
assert plan.count("'有效答案預算': 3200") == 4
assert plan.count("'品質': '尚待訓練與量測'") == 4

parameter_counts = {}
for width in (8, 16, 32):
    model = TinyLM(ModelConfig(width=width))
    actual = model.description()["parameters"]
    fixed = (2 * 264 + 128 + 2) * width
    block = 12 * width**2 + 9 * width
    assert actual == fixed + block
    parameter_counts[width] = {"fixed": fixed, "block": block, "total": actual,
                               "named_parameters": {n: p.numel() for n, p in model.named_parameters()}}
pool = [
    {"text": f"object={index};color={color};shape={shape}.", "family": str(index)}
    for index in range(80)
    for color, shape in [("red" if index % 2 else "blue", "circle" if index % 3 else "square")]
]
parts = split_records(pool, seed=42)
raw = json.loads((OUT / (PREFIX + "text-foundation-raw.json")).read_text())
manifest = raw["results"]["scaling_data"]
split_summary = {}
for split, rows in parts.items():
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == manifest[split]["sha256"]
    (OUT / (PREFIX + f"{split}.jsonl")).write_bytes(data)
    examples = text_examples(rows)
    x, y, valid = pad_batch(examples)
    count = int((y != IGNORE).sum())
    assert count == sum(len(row["text"].encode()) + 1 for row in rows)
    assert x.dtype == torch.long and y.dtype == torch.long and valid.dtype == torch.bool
    assert torch.all(y[~valid] == IGNORE)
    split_summary[split] = {"records": len(rows), "families": [r["family"] for r in rows],
                            "sha256": digest, "effective_targets": count,
                            "raw_utf8_bytes": sum(len(r["text"].encode()) for r in rows),
                            "padded_shape": list(x.shape), "ignored_padding": int((y == IGNORE).sum())}
assert not (set(split_summary["train"]["families"]) & set(split_summary["validation"]["families"]))
assert not (set(split_summary["train"]["families"]) & set(split_summary["test"]["families"]))
assert not (set(split_summary["validation"]["families"]) & set(split_summary["test"]["families"]))
sampling = {}
for size in (16, 64):
    rows = parts["train"][:size]
    examples = text_examples(rows)
    sampler = random.Random(42)
    total, selected = 0, []
    for _ in range(150):
        batch = sampler.choices(range(len(examples)), k=4)
        selected.extend(batch)
        x, y, valid = pad_batch([examples[i] for i in batch])
        total += int((y != IGNORE).sum())
    assert total == (20765 if size == 16 else 20623)
    sampling[size] = {"records_sha256": records_sha256(rows), "steps": 150,
                      "batch_size": 4, "draws_with_replacement": len(selected), "effective_targets": total,
                      "targets_per_record": [len(y) for _, y in examples],
                      "draw_counts": [selected.count(i) for i in range(size)]}

checkpoints = ROOT / "outputs/reviewer-tools/tool-choice-dep-r4-download/text_foundation"
export = json.loads((checkpoints / "export-manifest.json").read_text())
download = json.loads((checkpoints / "download-manifest.json").read_text())
for name in ("export-manifest.json", "download-manifest.json"):
    shutil.copyfile(checkpoints / name, OUT / (PREFIX + name))
evaluations = {}
for name, original in raw["results"]["scaling"].items():
    path = checkpoints / f"{name}.pt"
    entry = next(item for item in export["files"] if item["output"] == path.name)
    original_artifact = next(item for item in raw["artifacts"] if item["path"] == path.name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"]
    assert entry["source_sha256"] == original_artifact["sha256"]
    model, payload = load_checkpoint(path, device="cpu")
    size = original["unique_training_records"]
    assert payload["metadata"]["revision"] == raw["revision"]
    assert model.config.layers == 1 and model.config.width in (16, 32)
    assert model.description()["parameters"] == original["parameters"]
    assert records_sha256(parts["train"][:size]) == original["training"]["records_sha256"]
    values = {}
    for split in ("train", "validation", "test"):
        rows = parts[split][:size] if split == "train" else parts[split]
        observed = _nll(model, text_examples(rows, max_length=model.config.max_length))
        expected = original["training"]["final_loss"] if split == "train" else original["evaluation"][split]["nll"]
        difference = observed["nll"] - expected
        assert abs(difference) < 1e-6
        if split != "train":
            report = original["evaluation"][split]
            assert report["nll"] == report["nll_sum"] / report["effective_tokens"]
            assert observed["effective_tokens"] == report["effective_tokens"] == split_summary[split]["effective_targets"]
        values[split] = {"cpu": observed, "original_cuda_nll": expected, "difference": difference,
                         "five_decimal_table": f"{expected:.5f}"}
    evaluations[name] = {"checkpoint": str(path.relative_to(ROOT)), "sha256": digest,
                          "original_full_checkpoint_sha256": entry["source_sha256"], "config": payload["config"],
                          "original_training": original["training"], "recomputed": values}
code_versions = {}
for path in ("tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py",
             "tiny_perceptron/modern.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"):
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    code_versions[path] = {"current_sha256": digest, "formal_sha256": raw["code_sha256"][path],
                           "identical": digest == raw["code_sha256"][path]}
print(json.dumps({
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_05_13_audit.py",
    "result": "All assertions passed; no training performed.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                    "threads": str(torch.get_num_threads()), "dtype": "float32", "platform": platform.platform()},
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(), "snippet_stdout": plan,
    "parameter_counts": parameter_counts, "linear_flops": {"8x8": 2 * 8 * 8, "16x16": 2 * 16 * 16},
    "exercise": {"3200_div_4_div_4": 3200 // (4 * 4), "100_steps_4x4": 100 * 4 * 4},
    "splits": split_summary, "sampling": sampling, "evaluations": evaluations, "code_versions": code_versions,
    "original_environment": {k: raw[k] for k in ("python_version", "torch_version", "device", "gpu", "seed", "step_scale")},
    "measurement_scope": "CPU evaluation of exact exported weights and replay of data sampling only. Original quality from one completed L4 run, seed 42; no GPU timing or training reproduction claim.",
    "checkpoint_download": {"repo": download["repo"], "revision": download["revision"]},
}, ensure_ascii=False, indent=2))
