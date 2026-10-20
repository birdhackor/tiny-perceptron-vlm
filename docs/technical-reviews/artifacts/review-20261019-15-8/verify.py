"""核對路由算例、保存的實測紀錄與有效位置統計；不進行訓練。"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _routing
from tiny_perceptron.data import pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def entropy(fractions):
    return -sum(f * math.log(f) for f in fractions if f > 0)


def main():
    torch.set_num_threads(1)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-8/verify.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "platform": platform.platform(),
        },
        "inputs": {
            path: digest(path)
            for path in (
                "docs/course-review-20261019/freeze-01/sources/15.8.md",
                "docs/course-experiments/results/moe.json",
                "scripts/course_experiments/architecture.py",
                "tiny_perceptron/modern.py",
                "tiny_perceptron/model.py",
                "tiny_perceptron/data.py",
            )
        },
    }
    original, first_exercise, split_exercise = [], [], []
    for advantage in (8.0, 0.1):
        scores = torch.tensor([[advantage, 0.0, 0.0]]).repeat(12, 1)
        probability = scores.softmax(-1)
        chosen = probability.topk(1, dim=-1).indices
        counts = torch.bincount(chosen.flatten(), minlength=3)
        exact = [math.exp(advantage) / (math.exp(advantage) + 2)] + [1 / (math.exp(advantage) + 2)] * 2
        assert scores.shape == (12, 3) and chosen.shape == (12, 1)
        assert chosen.flatten().shape == (12,)
        assert counts.tolist() == [12, 0, 0]
        assert torch.allclose(probability[0], torch.tensor(exact), atol=1e-7, rtol=0)
        original.append({"advantage": advantage, "softmax_float64_formula": exact, "softmax_float32": probability.mean(0).tolist(), "counts": counts.tolist(), "load_fraction": (counts / counts.sum()).tolist()})
        alternative = torch.tensor([[advantage, 0.2, 0.0]]).repeat(12, 1)
        alt_count = torch.bincount(alternative.softmax(-1).topk(1, dim=-1).indices.flatten(), minlength=3)
        expected = [12, 0, 0] if advantage == 8 else [0, 12, 0]
        assert alt_count.tolist() == expected
        first_exercise.append({"advantage": advantage, "counts": alt_count.tolist()})
        split = torch.tensor([[0.1, 0.2, 0.0]] * 6 + [[0.2, 0.1, 0.0]] * 6)
        split_count = torch.bincount(split.softmax(-1).topk(1, dim=-1).indices.flatten(), minlength=3)
        assert split.shape == (12, 3) and split_count.tolist() == [6, 6, 0]
        split_exercise.append({"advantage": advantage, "counts": split_count.tolist()})
    result["examples"] = {"original": original, "first_exercise": first_exercise, "split_exercise": split_exercise}

    record = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_text())
    variants = record["results"]["variants"]
    variant = variants["top1_aux0"]
    routing = variant["validation_routing"]
    layer = routing["layers"][1]
    counts = layer["dispatch_counts"]
    total = sum(counts)
    fractions = [c / total for c in counts]
    percentages = [round(100 * f, 2) for f in fractions]
    normalized = entropy(fractions) / math.log(len(counts))
    assert counts == [8536, 25630, 4875, 215] and total == 39256
    assert percentages == [21.74, 65.29, 12.42, 0.55]
    assert routing["padding_excluded"] is True
    assert total == layer["dispatch_denominator"] == routing["effective_input_tokens"]
    assert variant["model"]["config"]["top_k"] == 1
    assert variant["training"]["auxiliary_weight"] == 0
    assert abs(normalized - layer["normalized_load_entropy"]) < 1e-12
    availability = []
    for name, value in variants.items():
        current = value["validation_routing"]
        availability.append({"variant": name, "experts": value["model"]["config"]["experts"], "validation_routing_is_null": current is None, "layers_available": isinstance(current, dict) and "layers" in current})
    result["saved_experiment"] = {
        "record_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
        "validation_split": record["results"]["dataset"]["validation"],
        "validation_chunks": variant["heldout"]["validation"]["examples"],
        "config": variant["model"]["config"],
        "training": {k: variant["training"][k] for k in ("requested_steps", "steps", "optimizer_updates", "effective_tokens", "batch_size", "auxiliary_weight")},
        "routing": routing,
        "percentages": percentages,
        "normalized_entropy_recomputed": normalized,
        "variants_field_availability": availability,
        "scope": "本次讀取並重算已保存實測紀錄；未重跑原始 CUDA 訓練。",
    }
    result["entropy_checks"] = {
        "e": math.e,
        "log_quarter": math.log(0.25),
        "quarter_term": -0.25 * math.log(0.25),
        "uniform_entropy": entropy([0.25] * 4),
        "uniform_normalized": entropy([0.25] * 4) / math.log(4),
        "concentrated_normalized": entropy([1, 0, 0, 0]) / math.log(4),
        "actual_counts_normalized_rounded": round(normalized, 5),
        "zero_limit_samples": [{"fraction": f, "term": -f * math.log(f)} for f in (1e-3, 1e-6, 1e-12)],
        "pure_batch_entropies": [entropy([1, 0, 0, 0]), entropy([0, 1, 0, 0]), entropy([0, 0, 1, 0]), entropy([0, 0, 0, 1])],
        "equal_pure_batches_aggregate_normalized": entropy([0.25] * 4) / math.log(4),
    }

    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8, layers=2, heads=2, max_length=8, experts=4, top_k=1))
    examples = []
    for index in range(17):
        length = index % 7 + 1
        x = (torch.arange(length) + index + 1) % 100
        examples.append((x, x + 1))
    actual = _routing(model, examples, SimpleNamespace(device="cpu"))
    manual = [{"counts": [0] * 4, "probability_sum": torch.zeros(4, dtype=torch.float64), "batches": []} for _ in model.blocks]
    current_valid = None

    def hook_for(index):
        def capture(module, args, output):
            probability = module.router(args[0][current_valid]).float().softmax(-1)
            selected = output[2][current_valid.reshape(-1)].flatten().tolist()
            count = [selected.count(i) for i in range(4)]
            entry = manual[index]
            entry["counts"] = [a + b for a, b in zip(entry["counts"], count)]
            entry["probability_sum"] += probability.double().sum(0)
            load = torch.tensor(count, dtype=torch.float32) / len(selected)
            aux = float(4 * (load * probability.mean(0)).sum())
            entry["batches"].append({"valid_positions": len(probability), "counts": count, "auxiliary": aux})
        return capture

    handles = [block.ffn.register_forward_hook(hook_for(i)) for i, block in enumerate(model.blocks)]
    padded_positions = valid_positions = 0
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(examples), 16):
                x, _, current_valid = pad_batch(examples[start:start + 16])
                padded_positions += x.numel()
                valid_positions += int(current_valid.sum())
                model(x, valid=current_valid)
    finally:
        for handle in handles:
            handle.remove()
    checked_layers = []
    for expected, observed in zip(manual, actual["layers"]):
        weighted = sum(b["auxiliary"] * b["valid_positions"] for b in expected["batches"]) / valid_positions
        probability = (expected["probability_sum"] / valid_positions).tolist()
        assert expected["counts"] == observed["dispatch_counts"]
        assert sum(expected["counts"]) == valid_positions == observed["dispatch_denominator"]
        assert abs(weighted - observed["mean_batch_auxiliary"]) < 1e-12
        assert all(abs(a - b) < 1e-12 for a, b in zip(probability, observed["mean_router_probability"]))
        checked_layers.append({"batches": expected["batches"], "manual_weighted_auxiliary": weighted, "observed": observed})
    result["routing_function_check"] = {"examples": len(examples), "valid_positions": valid_positions, "positions_including_padding": padded_positions, "layers": checked_layers}
    result["result"] = "所有數字與路由統計斷言通過；Dense 對照組的 validation_routing 為 null。"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
