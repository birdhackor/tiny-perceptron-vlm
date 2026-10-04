"""Fresh CPU audit of 7.14: no checkpoint load, model training, or GPU use."""

import copy
import hashlib
import json
import math
import platform
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import masked_loss


ROOT = Path(__file__).resolve().parents[3]
raw = json.loads((ROOT / "docs/course-experiments/results/sft_ablation.json").read_text())
report = raw["results"]
tok = ByteTokenizer()
torch.set_num_threads(1)


def independently_correct(question):
    pieces = question.split(";")
    values = dict(piece.split("=", 1) for piece in pieces[:-1])
    return {"describe": values["shape"], "shape?": values["shape"], "color?": values["color"], "pitch?": values["pitch"], "joint?": values["shape"] + "," + values["pitch"]}[pieces[-1]]


def digest(value):
    return hashlib.sha256(value).hexdigest()


result = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "Recompute objective, gradient, masks, data and RNG denominators, and saved raw-result metrics; no training or checkpoint inference rerun.",
}

# Closed-form objective and gradient, independently checked against float64 autograd.
toy = []
for last in (1.0, 5.0, 8.0):
    z = torch.tensor([[0.0, 0.0, 3.0, last]], dtype=torch.float64, requires_grad=True)
    denominator = 2 + math.exp(3) + math.exp(last)
    probabilities = [1 / denominator, 1 / denominator, math.exp(3) / denominator, math.exp(last) / denominator]
    row = {"logits": z.detach().tolist()[0], "probabilities": probabilities, "losses": {}, "gradients": {}}
    for label in (2, 3):
        loss = F.cross_entropy(z, torch.tensor([label]))
        manual = math.log(denominator) - float(z[0, label].detach())
        gradient = torch.autograd.grad(loss, z, retain_graph=True)[0][0]
        expected_gradient = torch.tensor(probabilities, dtype=torch.float64)
        expected_gradient[label] -= 1
        assert abs(loss.item() - manual) < 1e-12
        assert torch.allclose(gradient, expected_gradient, atol=1e-12, rtol=0)
        row["losses"][str(label)] = loss.item()
        row["gradients"][str(label)] = gradient.tolist()
    assert row["gradients"]["3"][3] < 0
    toy.append(row)
assert toy[2]["losses"]["3"] < toy[1]["losses"]["3"] < toy[0]["losses"]["3"]
assert toy[2]["losses"]["2"] > toy[1]["losses"]["2"] > toy[0]["losses"]["2"]
result["toy_objective_and_exercise"] = toy

# Recreate exactly the original family split and corruption rule, without updates.
parts = split_records(generate_records("attributes-sft"), seed=42)
split_stats = {}
families = {name: {row["family"] for row in rows} for name, rows in parts.items()}
for name, rows in parts.items():
    assert all(row["messages"][-1]["content"] == independently_correct(row["messages"][0]["content"]) for row in rows)
    data_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    encoded = text_examples(rows, mode="sft")
    effective = sum(int((y != IGNORE).sum()) for x, y in encoded)
    assert digest(data_bytes) == report["data"]["attributes"][name]["sha256"]
    assert len(rows) == report["data"]["attributes"][name]["records"]
    split_stats[name] = {"records": len(rows), "families": sorted(families[name]), "jsonl_sha256": digest(data_bytes), "effective_tokens": effective, "max_input_positions": max(len(x) for x, y in encoded)}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["test"] & families["validation"])
clean = parts["train"]
noisy = copy.deepcopy(clean)
corruptions = []
for index, row in enumerate(noisy):
    answer = row["messages"][-1]["content"]
    if answer in ("circle", "square") and len(corruptions) < max(1, len(noisy) // 10):
        row["messages"][-1]["content"] = "square" if answer == "circle" else "circle"
        corruptions.append({"row": index, "family": row["family"], "correct": answer, "wrong": row["messages"][-1]["content"]})
assert corruptions == report["corruptions"]
assert len(corruptions) == 4 and len({row["family"] for row in corruptions}) == 2
assert all(a["messages"][:-1] == b["messages"][:-1] for a, b in zip(clean, noisy, strict=True))
result["split_and_corruptions"] = {"splits": split_stats, "corruptions": corruptions, "changed_record_fraction": len(corruptions) / len(clean), "randomly_distributed": False}
upstream = json.loads((ROOT / "docs/course-experiments/results/sft.json").read_text())
assert upstream["results"]["after"] == report["before"]["A_attributes"]
assert upstream["results"]["checkpoint"] == "model.pt"
assert upstream["results"]["training"]["steps"] == 900
result["direct_sft_base_report_check"] = {"upstream_report_sha256": digest((ROOT / "docs/course-experiments/results/sft.json").read_bytes()), "checkpoint": "model.pt", "upstream_updates": 900, "before_ablation_equals_full_upstream_after_metrics_and_samples": True}

mask_stats = {}
for name, rows in (("clean", clean), ("noisy", noisy)):
    examples = text_examples(rows, mode="sft")
    assert records_sha256(rows) == report["runs"][name]["training"]["records_sha256"]
    for row, (x, y) in zip(rows, examples, strict=True):
        expected_targets = tok.encode(row["messages"][-1]["content"]) + [tok.eos_id]
        assert y[y != IGNORE].tolist() == expected_targets
        assert x.dtype == y.dtype == torch.int64
        assert len(x) <= 128
    x, y, valid = pad_batch(examples)
    logits = torch.zeros((*y.shape, tok.vocab_size), dtype=torch.float64, requires_grad=True)
    loss = masked_loss(logits, y)
    loss.backward()
    assert abs(loss.item() - math.log(tok.vocab_size)) < 1e-12
    assert torch.count_nonzero(logits.grad[y == IGNORE]).item() == 0
    assert torch.count_nonzero(logits.grad[y != IGNORE]).item() > 0
    assert not (y[~valid] != IGNORE).any()
    assert all(int((a[1] != IGNORE).sum()) == int((b[1] != IGNORE).sum()) for a, b in zip(text_examples(clean, "sft"), text_examples(noisy, "sft"), strict=True))
    rng = random.Random(42)
    chosen = [rng.choices(range(len(examples)), k=16) for _ in range(300)]
    counts = Counter(index for batch in chosen for index in batch)
    effective = sum(int((examples[index][1] != IGNORE).sum()) for batch in chosen for index in batch)
    changed_indices = {row["row"] for row in corruptions}
    changed_exposures = sum(counts[index] for index in changed_indices)
    changed_answer_tokens = sum(counts[index] * int((examples[index][1] != IGNORE).sum()) for index in changed_indices)
    assert effective == report["runs"][name]["training"]["effective_tokens"] == 33733
    assert report["runs"][name]["training"]["steps"] == 300
    mask_stats[name] = {"dtype": str(x.dtype), "batch_shape": list(x.shape), "valid_input_positions": int(valid.sum()), "supervised_positions_one_dataset_pass": int((y != IGNORE).sum()), "ignored_positions_including_padding": int((y == IGNORE).sum()), "uniform_logit_loss": loss.item(), "ignored_logit_gradient_nonzero": 0, "sampled_examples": 4800, "updates": 300, "batch_size": 16, "sampler_seed": 42, "sampled_effective_tokens": effective, "draw_index_sha256": digest(json.dumps(chosen).encode()), "selected_corrupted_record_exposures": changed_exposures, "selected_corrupted_record_answer_tokens_including_eos": changed_answer_tokens, "records_sha256": records_sha256(rows)}
assert mask_stats["clean"]["draw_index_sha256"] == mask_stats["noisy"]["draw_index_sha256"]
result["masking_and_training_denominator"] = mask_stats

# Recompute saved complete-generation metrics from raw token IDs, not decoded strings.
metrics = {}
for name in ("clean", "noisy"):
    metrics[name] = {}
    for split in ("validation", "test"):
        saved = report["runs"][name]["attributes"][split]
        rows = parts[split]
        assert len(saved["samples"]) == len(rows) == saved["records"] == saved["examples"]
        checked = []
        for sample, row in zip(saved["samples"], rows, strict=True):
            expected = independently_correct(row["messages"][0]["content"])
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == expected
            generated = sample["generated_ids"]
            ended = tok.eos_id in generated
            content = generated[: generated.index(tok.eos_id)] if ended else generated
            exact = content == tok.encode(expected)
            assert ended == sample["eos"]
            assert exact == sample["exact"]
            assert tok.decode(content) == sample["generated"]
            checked.append({"question": row["messages"][0]["content"], "expected": expected, "generated": sample["generated"], "generated_ids": generated, "exact_recomputed": exact, "eos_recomputed": ended})
        matches = sum(row["exact_recomputed"] for row in checked)
        eos_count = sum(row["eos_recomputed"] for row in checked)
        effective = split_stats[split]["effective_tokens"]
        assert matches == saved["matches"]
        assert matches / len(rows) == saved["exact_match"]
        assert eos_count / len(rows) == saved["eos_rate"] == 1
        assert effective == saved["effective_tokens"]
        assert abs(saved["nll_sum"] / effective - saved["nll"]) < 1e-15
        metrics[name][split] = {"matches": matches, "records": len(rows), "eos_count": eos_count, "effective_tokens": effective, "nll_sum": saved["nll_sum"], "nll_recomputed": saved["nll_sum"] / effective, "nll_display_5dp": f'{saved["nll"]:.5f}', "samples": checked}
assert [metrics[name][split]["matches"] for name in ("clean", "noisy") for split in ("validation", "test")] == [2, 7, 1, 6]
assert metrics["noisy"]["test"]["nll_recomputed"] < metrics["clean"]["test"]["nll_recomputed"]
result["raw_generation_and_metric_recount"] = metrics

relevant_paths = ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]
result["recorded_experiment_code_comparison"] = {path: {"recorded_sha256": raw["code_sha256"][path], "current_sha256": digest((ROOT / path).read_bytes()), "identical": raw["code_sha256"][path] == digest((ROOT / path).read_bytes())} for path in relevant_paths}
plan = json.loads((ROOT / "docs/course-experiments/plan.json").read_text())
experiment = next(row for row in plan["sequence"] if row["id"] == "sft_ablation")
assert experiment["module"] == "text" and experiment["function"] == "run_sft_ablation"
assert experiment["dependencies"] == ["sft"] and "7.14" in experiment["lessons"]
tool_choice = next(row for row in plan["supporting_experiments"] if row["id"] == "tool_choice")
assert "7.14" not in tool_choice["lessons"] and tool_choice["module"] == "tool_choice"
result["current_plan_dependency_inspection"] = {"plan_sha256": digest((ROOT / "docs/course-experiments/plan.json").read_bytes()), "sft_ablation_entry": experiment, "new_tool_choice_entry": tool_choice, "finding": "The new supporting tool_choice entry does not change the existing sft_ablation route, dependencies or 7.14 evidence."}
result["assertions_passed"] = True
print(json.dumps(result, ensure_ascii=False, indent=2))
