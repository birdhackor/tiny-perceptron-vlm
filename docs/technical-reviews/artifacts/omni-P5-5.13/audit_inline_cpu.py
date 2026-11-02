"""Independent 5.13 audit: model construction and pure preserved-result recomputation.

No model forward, gradient, optimizer update, checkpoint loading, training, or
external service is invoked. Run from the repository root with the project CPU
environment; the JSON output is the portable execution record.
"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


freeze_path = Path("docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json")
freeze = json.loads(freeze_path.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "5.13")
snapshot = Path(page["snapshot"]).read_bytes()
assert sha(snapshot) == page["source_sha256"]
body = snapshot.decode("utf-8")
live = Path(page["source"]).read_bytes().decode("utf-8")
live_section = re.search(r"^## 5\.13 .*?(?=^## |\Z)", live, re.M | re.S).group(0)
assert live_section == body
inline = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(inline) == 1


def execute(source, budget):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, page["snapshot"] + ":inline", "exec"), {})
    rows = [ast.literal_eval(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 4
    assert {(r["特徵寬度"], r["不同記錄"]) for r in rows} == {(8, 16), (8, 64), (16, 16), (16, 64)}
    assert all(r["有效目標預算"] == budget and r["品質"] == "尚待訓練與量測" for r in rows)
    return {"code_sha256": sha(source.encode()), "stdout": output.getvalue(), "rows": rows}


inline_runs = {
    "unchanged_frozen_code": execute(inline[0], 3200),
    "specified_exercise_4_4_200": execute(inline[0].replace("4, 8, 100", "4, 4, 200"), 3200),
    "counterexample_4_4_100": execute(inline[0].replace("4, 8, 100", "4, 4, 100"), 1600),
}
parameters = {}
for width, expected in ((8, 6104), (16, 13744), (32, 33632)):
    model = TinyLM(ModelConfig(width=width))
    observed = model.description()["parameters"]
    assert observed == expected == 12 * width**2 + 667 * width
    parameters[str(width)] = {"observed": observed, "formula": "12*w*w + 667*w", "config": model.description()["config"]}

result_path = Path("docs/course-experiments/results/text_foundation.json")
result = json.loads(result_path.read_bytes())
pool = [
    {"text": f"object={i};color={'red' if i % 2 else 'blue'};shape={'circle' if i % 3 else 'square'}.", "family": str(i)}
    for i in range(80)
]
groups = {row["family"]: row for row in pool}
keys = sorted(groups)
random.Random(42).shuffle(keys)
parts = {name: [groups[k] for k in selected] for name, selected in (("train", keys[:64]), ("validation", keys[64:72]), ("test", keys[72:]))}
splits = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = sha(raw)
    assert digest == result["results"]["scaling_data"][name]["sha256"]
    splits[name] = {"records": len(rows), "effective_targets_one_pass": sum(len(r["text"].encode()) + 1 for r in rows), "jsonl_sha256": digest, "families": [r["family"] for r in rows]}
assert set(splits["train"]["families"]).isdisjoint(splits["validation"]["families"])
assert set(splits["train"]["families"]).isdisjoint(splits["test"]["families"])
assert set(splits["validation"]["families"]).isdisjoint(splits["test"]["families"])

sample_budgets = {}
for count, expected in ((16, 20765), (64, 20623)):
    rows = parts["train"][:count]
    sampler = random.Random(42)
    sampled = [r for _ in range(150) for r in sampler.choices(rows, k=4)]
    exposure = sum(len(r["text"].encode()) + 1 for r in sampled)
    assert exposure == expected
    sample_budgets[str(count)] = {"draws": len(sampled), "effective_target_exposures": exposure, "one_pass_training_targets": sum(len(r["text"].encode()) + 1 for r in rows), "records_sha256": sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()), "unique_sampled_families": len({r["family"] for r in sampled})}

table = []
for match in re.finditer(r"^\| (16|32) \| (16|64)篇 \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", body, re.M):
    w, n, tokens, training, validation, test = match.groups()
    record = result["results"]["scaling"][f"scaling-w{w}-n{n}"]
    assert record["training"]["steps"] == 150
    assert record["parameters"] == parameters[w]["observed"]
    assert record["unique_training_records"] == int(n)
    assert record["training"]["records_sha256"] == sample_budgets[n]["records_sha256"]
    assert int(tokens.replace(",", "")) == record["training"]["effective_tokens"] == sample_budgets[n]["effective_target_exposures"]
    assert f'{record["training"]["final_loss"]:.5f}' == training
    entry = {"run": f"scaling-w{w}-n{n}", "training_final_loss_preserved": record["training"]["final_loss"], "training_display": training, "training_denominator_one_pass": sample_budgets[n]["one_pass_training_targets"], "training_exposure": int(tokens.replace(",", "")), "evaluations": {}}
    for split, display in (("validation", validation), ("test", test)):
        ev = record["evaluation"][split]
        recomputed = ev["nll_sum"] / ev["effective_tokens"]
        assert ev["effective_tokens"] == splits[split]["effective_targets_one_pass"]
        assert ev["records"] == ev["examples"] == 8
        assert abs(recomputed - ev["nll"]) <= 1e-15
        assert f"{recomputed:.5f}" == display
        entry["evaluations"][split] = {"nll_sum_preserved": ev["nll_sum"], "effective_tokens": ev["effective_tokens"], "recomputed_nll": recomputed, "display": display}
    table.append(entry)
assert len(table) == 4
for width in (16, 32):
    small = result["results"]["scaling"][f"scaling-w{width}-n16"]
    large = result["results"]["scaling"][f"scaling-w{width}-n64"]
    assert large["evaluation"]["validation"]["nll"] < small["evaluation"]["validation"]["nll"]
    assert large["training"]["final_loss"] > small["training"]["final_loss"]

historical = {}
for path in ("tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"):
    old = subprocess.check_output(["git", "show", result["revision"] + ":" + path])
    assert sha(old) == result["code_sha256"][path]
    historical[path] = {"recorded_sha256": sha(old), "current_sha256": sha(Path(path).read_bytes()), "same_bytes": old == Path(path).read_bytes()}

print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_5_13",
    "environment": {"python": sys.version, "torch": torch.__version__, "device": "CPU; model construction only", "platform": platform.platform()},
    "source_freeze": {"path": str(freeze_path), "sha256": sha(freeze_path.read_bytes())},
    "section_sha256": sha(snapshot), "inline_runs": inline_runs, "parameters": parameters,
    "flops": {"convention": "two floating-point operations per multiply-accumulate", "8_to_8": 2*8*8, "16_to_16": 2*16*16, "ratio": 4, "exact_dot_product_without_bias": {"8_to_8": 8*(2*8-1), "16_to_16": 16*(2*16-1)}},
    "original_result": {"path": str(result_path), "sha256": sha(result_path.read_bytes()), "revision": result["revision"], "seed": result["seed"], "status": result["status"], "run_id": result["modal"]["run_id"], "gpu": result["gpu"], "python": result["python_version"], "torch": result["torch_version"]},
    "splits": splits, "sample_budgets": sample_budgets, "table": table, "historical_code_sha_checks": historical,
    "limitations": ["No neural forward, training, gradient, optimizer update, checkpoint download/loading, or service dispatch.", "Training final_loss is checked as a preserved scalar and rounded transcription; original training logits/loss sums are not re-evaluated.", "Validation/test NLLs are recomputed from preserved nll_sum/count, without regenerating the original logits.", "Sampling and split hashes are independently reconstructed from the inspected original deterministic recipe.", "FLOPs are an arithmetic convention for a dense Linear position, not a measured whole-model runtime."]
}, ensure_ascii=False, indent=2))
