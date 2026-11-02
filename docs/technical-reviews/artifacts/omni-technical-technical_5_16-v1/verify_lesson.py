"""5.16 的短 CPU 核對；只執行教材例子、資料抽樣與計數，不訓練模型。"""

import contextlib
import hashlib
import inspect
import io
import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.course_experiments.common import Context, fit_lm, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum, masked_loss

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("verification-result.json")
torch.set_num_threads(1)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    text = raw.decode("utf-8")
    begin = re.search(r"^## " + re.escape(identifier) + r"\s", text, re.M)
    following = re.search(r"^## ", text[begin.end() :], re.M)
    end = begin.end() + following.start() if following else len(text)
    return text[begin.start() : end]


def run_blocks(text):
    namespace, output = {}, io.StringIO()
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
    with contextlib.redirect_stdout(output):
        for block in blocks:
            exec(compile(block, "<exact-current-lesson-block>", "exec"), namespace)
    return namespace, output.getvalue(), [digest(b.encode()) for b in blocks]


freeze_path = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "5.16")
raw = (ROOT / page["snapshot"]).read_bytes()
assert digest(raw) == page["source_sha256"]
assert section((ROOT / page["source"]).read_bytes(), "5.16").encode() == raw
assert page["figures_sha256"] == {} and not re.search(r"!\[[^\]]*\]\(", raw.decode())

result = {
    "reviewer_task": "/root/omni_technical_coordinator/technical_5_16",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
    },
    "source": {
        "freeze": freeze_path,
        "freeze_sha256": digest((ROOT / freeze_path).read_bytes()),
        "snapshot": page["snapshot"],
        "source_sha256": digest(raw),
        "current_section_equals_snapshot": True,
        "figures_sha256": page["figures_sha256"],
    },
    "scope": "No model forward, optimizer, checkpoint load, training, GPU, network or paid job; original 5.2/7.6 tensor examples do backward only on toy logits.",
}

prerequisites = {}
for identifier, path in [
    ("5.2", "course/chapters/05.md"),
    ("5.12", "course/chapters/05.md"),
    ("7.6", "course/chapters/07.md"),
    ("7.16", "course/chapters/07.md"),
    ("T.4", "course/training.md"),
]:
    current = section((ROOT / path).read_bytes(), identifier)
    frozen = next((p for p in freeze["pages"] if p["page_id"] == identifier), None)
    if identifier == "T.4":
        frozen = next(p for p in freeze["pages"] if p["page_id"] == "training")
        frozen_text = section((ROOT / frozen["snapshot"]).read_bytes(), identifier)
    else:
        frozen_text = (ROOT / frozen["snapshot"]).read_bytes().decode()
    assert current == frozen_text
    prerequisites[identifier] = {
        "source": path + "#" + identifier,
        "section_sha256": digest(current.encode()),
        "freeze_snapshot": frozen["snapshot"],
        "current_equals_frozen_section": True,
        "figure_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", current),
    }
result["current_prerequisites"] = prerequisites

result["inline_program"] = {}
for name, text, expected in [
    ("base", raw.decode(), {"短答": 90, "長答": 200}),
    ("long_length_1", raw.decode().replace('"長答": 20', '"長答": 1'), {"短答": 90, "長答": 10}),
    ("long_count_5", raw.decode().replace('"長答": 10', '"長答": 5'), {"短答": 90, "長答": 100}),
]:
    values, stdout, hashes = run_blocks(text)
    assert values["tokens"] == expected
    result["inline_program"][name] = {
        "tokens": values["tokens"], "total": values["total"], "ratios": values["ratios"],
        "stdout": stdout, "block_sha256": hashes,
    }
result["target_90_10"] = {"short_examples_per_long": 9 * 20, "long_token_fraction": 20 / (180 + 20)}
assert result["target_90_10"]["long_token_fraction"] == 0.1

result["prerequisite_code"] = {}
for identifier, path in [("5.2", "course/chapters/05.md"), ("7.6", "course/chapters/07.md")]:
    values, stdout, hashes = run_blocks(section((ROOT / path).read_bytes(), identifier))
    result["prerequisite_code"][identifier] = {"stdout": stdout, "block_sha256": hashes}

messages = [
    [{"role": "user", "content": "a much longer question"}, {"role": "assistant", "content": "x"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "long"}],
]
examples = [render_chat(m) for m in messages]
x, y, valid = pad_batch(examples)
effective = (y != IGNORE).sum(dim=1).tolist()
assert effective == [2, 5] and bool((y[~valid] == IGNORE).all())
logits = torch.zeros(*y.shape, ByteTokenizer.vocab_size, dtype=torch.float64)
summed, count = loss_sum(logits, y)
mean = masked_loss(logits, y)
assert int(count) == 7 and torch.allclose(mean, torch.log(torch.tensor(264.0, dtype=torch.float64)))
result["real_label_count"] = {
    "per_record": effective, "includes": "assistant UTF-8 bytes plus EOS; no prompt or padding labels",
    "padded_shape": list(x.shape), "valid_input_positions": int(valid.sum()),
    "effective_targets": int(count), "sum_loss": float(summed), "mean_loss": float(mean),
    "dtype": str(logits.dtype), "padding_labels_all_ignored": True,
}
result["transparent_loss_examples"] = {
    "same_90_200_token_counts_different_loss_share": {"short_nll": 90.0, "long_nll": 2.0, "long_loss_share": 2 / 92},
    "source_weight_changes_score_not_count": {"targets_before_and_after": 290, "unweighted_sum": 92.0, "long_weight_2_sum": 94.0},
    "aggregate_can_hide_minority_decline": {"counts": [90, 10], "means_before": [1.0, 1.0], "means_after": [0.5, 2.0], "overall_before": 1.0, "overall_after": (90 * 0.5 + 10 * 2.0) / 100},
}

original_path = "docs/course-experiments/results/sft_ablation.json"
original_raw = (ROOT / original_path).read_bytes()
original = json.loads(original_raw)
result["original_experiment"] = {
    "path": original_path, "sha256": digest(original_raw),
    **{k: original[k] for k in ["revision", "seed", "device", "gpu", "torch_version", "python_version", "step_scale", "evidence_status", "status"]},
}
assert original["seed"] == 42 and original["step_scale"] == 1.0
result["code_matches_recorded_experiment"] = {}
for path in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]:
    actual = digest((ROOT / path).read_bytes())
    assert actual == original["code_sha256"][path]
    result["code_matches_recorded_experiment"][path] = actual

attributes = split_records(generate_records("attributes-sft"), seed=42)
arithmetic = split_records(arithmetic_records(), seed=42)
result["regenerated_training_data"] = {}
for name, splits in [("attributes", attributes), ("arithmetic", arithmetic)]:
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits["train"]).encode()
    original_data = original["results"]["data"][name]["train"]
    assert digest(encoded) == original_data["sha256"]
    result["regenerated_training_data"][name] = {"records": len(splits["train"]), "sha256": digest(encoded)}

result["sampling_only_replay"] = {}
batch_size = inspect.signature(fit_lm).parameters["batch_size"].default
assert batch_size == 16
for name, rows in [("b-only", arithmetic["train"]), ("replay", attributes["train"] + arithmetic["train"])]:
    recorded = original["results"]["runs"][name]["training"]
    assert records_sha256(rows) == recorded["records_sha256"] and recorded["steps"] == 500
    prepared = text_examples(rows, mode="sft", max_length=128)
    lengths = [int((labels != IGNORE).sum()) for _, labels in prepared]
    source_by_id = {id(pair): ("A" if name == "replay" and i < len(attributes["train"]) else "B") for i, pair in enumerate(prepared)}
    sampler, targets, source_records, checkpoints = random.Random(42), 0, Counter(), {}
    for step in range(500):
        batch = sampler.choices(prepared, k=batch_size)
        _, labels, _ = pad_batch(batch)
        targets += int((labels != IGNORE).sum())
        source_records.update(source_by_id[id(pair)] for pair in batch)
        if (step + 1) % 125 == 0:
            checkpoints[str(step + 1)] = targets
    assert targets == recorded["effective_tokens"]
    result["sampling_only_replay"][name] = {
        "steps": 500, "batch_size": batch_size, "sampled_records": 500 * batch_size,
        "source_sampled_records": dict(source_records), "effective_tokens": targets,
        "matches_original": True, "cumulative_tokens": checkpoints,
        "population_record_digest": records_sha256(rows),
        "population_effective_length_histogram": dict(Counter(lengths)),
        "max_input_length": max(len(inputs) for inputs, _ in prepared),
    }
result["experiment_token_ratio"] = 36384 / 18453

result["recipe_interfaces"] = {}
for identifier in ["sft", "sft_ablation"]:
    spec = experiment_spec(identifier)
    command = [sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", identifier]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    result["recipe_interfaces"][identifier] = {
        "module": spec["module"], "function": spec["function"], "dependencies": spec["dependencies"],
        "assets": spec["assets"], "command": command, "exit_code": proc.returncode,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
missing = Context("cpu", ROOT / "unused-review-output", Path("/nonexistent-5-16-review-dependencies"), ROOT / "assets/training")
try:
    missing.dependency("sft")
except FileNotFoundError as error:
    result["recipe_interfaces"]["missing_sft_dependency"] = str(error)
else:
    raise AssertionError("Missing sft/model.pt must fail before training")

result["completed_at"] = datetime.now(timezone.utc).isoformat()
with OUT.open("x", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
