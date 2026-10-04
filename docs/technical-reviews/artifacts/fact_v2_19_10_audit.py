"""Independent CPU review of lesson 19.10; no training or GPU timing claims."""

import hashlib
import json
import platform
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from scripts.course_experiments.capstone import _balanced_sample
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.capstone import (
    TOK,
    CapstoneModel,
    build_dataset,
    calculator_runtime,
    default_config,
    evaluate_rows,
    expected_final,
    load_capstone,
    parse_action,
    prepare_batch,
    prompt_ids,
)
from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantizable_weights
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss
from tiny_perceptron.quantization import QuantizedLinear, quantize_symmetric, unpack_int4
from tiny_perceptron.training import seed_everything

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_10_"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def write_json(name, data):
    (OUT / f"{PREFIX}{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{prefix}.{key}".strip("."))


def inventory(value):
    return {
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": t.numel(),
            "bytes": t.numel() * t.element_size(),
            "sha256": sha(t.detach().cpu().contiguous().numpy().tobytes()),
        }
        for name, t in tensors(value)
    }


def verify_records(path, rows):
    data = read_json(path)
    assert len(data["records"]) == len(rows) == data["count"]
    counts = {}
    for record, row in zip(data["records"], rows, strict=True):
        assert (record["id"], record["family"], record["task"]) == (row["id"], row["family"], row["task"])
        assert record["expected_action"] == row["answer"]
        assert record["expected_final"] == expected_final(row)
        assert record["action_trace"]["prompt_ids"] == prompt_ids(row)
        for trace in (record["action_trace"], record["final_trace"]):
            if trace is not None:
                assert trace["raw"] == TOK.decode(trace["generated_ids"])
                assert trace["eos"] == (trace["generated_ids"][-1] == TOK.eos_id)
                assert trace["stop_reason"] == ("eos" if trace["eos"] else trace["stop_reason"])
        parsed = parse_action(record["action_trace"])
        assert parsed == record["parsed_action"]
        answer = parsed.get("content") if parsed["status"] in ("direct", "ask") else None
        if parsed["status"] == "tool":
            runtime = calculator_runtime(parsed, row["available"])
            assert runtime == record["runtime"]
            if runtime["status"] == "ok":
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{runtime['result']}。請回答。"
                assert record["final_trace"]["prompt_ids"] == prompt_ids(followup)
                final = parse_action(record["final_trace"])
                answer = final.get("content") if final["status"] == "direct" else None
        assert record["answer"] == answer
        action_correct = record["action_trace"]["eos"] and record["action_trace"]["raw"] == row["answer"]
        end_correct = action_correct and answer == expected_final(row)
        assert (record["action_correct"], record["end_to_end_correct"]) == (action_correct, end_correct)
        task = counts.setdefault(row["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        task["count"] += 1
        task["action_correct"] += int(action_correct)
        task["end_to_end_correct"] += int(end_correct)
    assert counts == data["by_task"]
    assert sum(x["action_correct"] for x in counts.values()) == data["action_correct"]
    assert sum(x["end_to_end_correct"] for x in counts.values()) == data["end_to_end_correct"]
    return data


def ids(record):
    return [record[k]["generated_ids"] if record[k] else None for k in ("action_trace", "final_trace")]


def main():
    torch.set_num_threads(2)
    splits, manifest = build_dataset(42)
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}}
    dep = read_json("docs/course-experiments/results/capstone_deployment.json")
    student = read_json("docs/course-experiments/results/capstone_student.json")
    assert dep["results"]["data_manifest"] == student["results"]["data_manifest"] == manifest
    result["split_counts"] = {k: len(v) for k, v in splits.items()}
    result["data_manifest_sha256"] = sha(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode())
    for bits in (4, 8):
        torch.manual_seed(42)
        layer = nn.Linear(8, 4)
        compressed = QuantizedLinear(layer, bits=bits)
        x = torch.ones(1, 8)
        with torch.no_grad():
            original, approximate = layer(x), compressed(x)
        result[f"example{bits}"] = {
            "input": x.tolist(),
            "original": original.tolist(),
            "approximate": approximate.tolist(),
            "max_abs_error": float((original - approximate).abs().max()),
            "output_shape": list(original.shape),
            "original_bytes": sum(p.numel() * p.element_size() for p in layer.parameters()),
            "compressed_bytes": compressed.storage_bytes(),
            "buffers": inventory(dict(compressed.named_buffers())),
            "requires_grad_inside_no_grad": original.requires_grad,
        }
    seed_everything(42)
    initial = CapstoneModel(replace(default_config(dense=True), width=48))
    initial_sha = sha(b"".join(t.detach().cpu().numpy().tobytes() for t in initial.state_dict().values()))
    assert initial_sha == student["results"]["initial_student_state_sha256"]
    result["initial_student"] = {"state_sha256": initial_sha, "description": initial.description()}
    sampler = random.Random(42 + 5000)
    batches = []
    for step in range(350):
        sampled = _balanced_sample(splits["train"], sampler, 24)
        _, labels = prepare_batch(sampled)
        batches.append(
            {"step": step + 1, "row_ids": [r["id"] for r in sampled], "effective_tokens": int((labels != IGNORE).sum())}
        )
    tokens = sum(b["effective_tokens"] for b in batches)
    assert tokens == 145163
    write_json(
        "cpu_reconstructed_sampling",
        {"scope": "CPU deterministic recipe replay, not saved GPU batch logs", "batches": batches},
    )
    result["sampling"] = {
        "seed": 5042,
        "steps": 350,
        "batch_size": 24,
        "effective_tokens": tokens,
        "scope": "recipe replay only",
    }
    logits = torch.tensor([[[0.3, -0.7, 1.2], [0.0, 0.2, -0.1], [0.5, -0.2, 0.3]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0, 0.2], [0.3, -0.1, 0.2], [-0.4, 0.5, 0.1]]], requires_grad=True)
    labels = torch.tensor([[-100, 1, 2]])
    p, logq = (teacher.detach() / 2).softmax(-1), (logits / 2).log_softmax(-1)
    direct_kl = (p * (p.log() - logq)).sum(-1)[labels != -100].mean() * 4
    actual_kl = distillation_kl(logits, teacher, labels, temperature=2)
    loss = distillation_loss(logits, teacher, labels, alpha=0.5, temperature=2)
    expected_loss = 0.5 * masked_loss(logits, labels) + 0.5 * direct_kl
    assert torch.allclose(actual_kl, direct_kl) and torch.allclose(loss, expected_loss)
    loss.backward()
    assert teacher.grad is None and torch.equal(logits.grad[0, 0], torch.zeros(3))
    result["kd_formula"] = {
        "ce": float(masked_loss(logits, labels).detach()),
        "direct_scaled_kl": float(direct_kl.detach()),
        "helper_scaled_kl": float(actual_kl.detach()),
        "mixed_loss": float(loss.detach()),
        "teacher_grad": None,
        "student_grad": logits.grad.tolist(),
        "effective_positions": 2,
        "T": 2,
        "alpha": 0.5,
    }
    filenames = {
        "joint": ("deployment", "test-joint"),
        "joint4": ("deployment", "test-joint-ptq4"),
        "joint8": ("deployment", "test-joint-ptq8"),
        "dpo": ("deployment", "test-dpo"),
        "dpo4": ("deployment", "test-ptq4"),
        "dpo8": ("deployment", "test-ptq8"),
        "ce": ("student", "test-ce"),
        "kd": ("student", "test-kd"),
        "kd4": ("student", "test-kd-ptq4"),
    }
    checked = {
        name: verify_records(f"docs/course-experiments/capstone-evidence/{folder}/{file}.json", splits["test"])
        for name, (folder, file) in filenames.items()
    }
    for name in ("ce", "kd"):
        validation = verify_records(
            f"docs/course-experiments/capstone-evidence/student/validation-{name}.json", splits["validation"]
        )
        result[f"validation_{name}"] = {k: v for k, v in validation.items() if k != "records"}
    result["full_record_recalculation"] = {
        name: {k: v for k, v in data.items() if k != "records"} for name, data in checked.items()
    }
    result["differences"] = {}
    for left, right in [
        ("joint", "joint4"),
        ("joint", "joint8"),
        ("dpo", "dpo4"),
        ("dpo", "dpo8"),
        ("kd", "kd4"),
        ("ce", "kd"),
    ]:
        changed = []
        for a, b, row in zip(checked[left]["records"], checked[right]["records"], splits["test"], strict=True):
            if ids(a) != ids(b):
                changed.append({"row": row, "left": a, "right": b})
        result["differences"][f"{left}_vs_{right}"] = {"count": len(changed), "rows": changed}
    dep_dir = next((ROOT / "outputs/integration-runs/v2-deployment/capstone-review").glob("*/*"))
    stu_dir = next((ROOT / "outputs/integration-runs/v2-student/capstone-review").glob("*/*"))
    weight_paths = {
        "joint": dep_dir / "joint.pt",
        "joint4": dep_dir / "joint-int4.pt",
        "joint8": dep_dir / "joint-int8.pt",
        "dpo": dep_dir / "dpo.pt",
        "dpo4": dep_dir / "dpo-int4.pt",
        "dpo8": dep_dir / "dpo-int8.pt",
        "ce": stu_dir / "ce/model.pt",
        "kd": stu_dir / "kd/model.pt",
        "kd4": stu_dir / "model-int4.pt",
    }
    all_weights = {}
    result["cpu_generations"] = {}
    for name, path in weight_paths.items():
        packed = name in ("joint4", "joint8", "dpo4", "dpo8", "kd4")
        model, payload = load_quantized_capstone(path) if packed else load_capstone(path)
        tables = inventory({"model": payload["model"], "quantized": payload.get("quantized", {})})
        restored = inventory(model.state_dict())
        record = {
            "local_origin": str(path.relative_to(ROOT)),
            "file_sha256": sha(path.read_bytes()),
            "file_bytes": path.stat().st_size,
            "format_version": payload["format_version"],
            "config": payload["config"],
            "metadata": payload["metadata"],
            "tensor_bytes": sum(t["bytes"] for t in tables.values()),
            "stored_tensors": tables,
            "restored_tensors": restored,
            "restored_tensor_bytes": sum(t["bytes"] for t in restored.values()),
            "all_restored_float32": all(t["dtype"] == "torch.float32" for t in restored.values()),
        }
        if packed:
            source_name = {"joint4": "joint", "joint8": "joint", "dpo4": "dpo", "dpo8": "dpo", "kd4": "kd"}[name]
            _, source = load_capstone(weight_paths[source_name])
            assert set(payload["quantized"]) == set(quantizable_weights(model))
            errors = {}
            for key, original in source["model"].items():
                if key in payload["model"]:
                    assert torch.equal(original, payload["model"][key])
                else:
                    q, scale = quantize_symmetric(original, payload["quantization"]["bits"], per_channel=True)
                    stored = payload["quantized"][key]
                    unpacked = (
                        unpack_int4(stored["values"], stored["shape"])
                        if payload["quantization"]["bits"] == 4
                        else stored["values"]
                    )
                    assert torch.equal(q, unpacked) and torch.equal(scale, stored["scale"])
                    errors[key] = float((original - model.state_dict()[key]).abs().max())
            record["exact_requantization_matches"] = True
            record["quantization"] = payload["quantization"]
            record["max_errors"] = errors
        all_weights[name] = record
        evaluation = evaluate_rows(model, splits["test"])
        actual = evaluation["records"]
        expected = checked[name]["records"]
        matches = [ids(a) == ids(b) for a, b in zip(actual, expected, strict=True)]
        result["cpu_generations"][name] = {
            "count": len(actual),
            "action_correct": evaluation["action_correct"],
            "end_to_end_correct": evaluation["end_to_end_correct"],
            "all_generation_ids_match_saved_gpu": all(matches),
            "ids_equal_by_row": matches,
            "changed_rows": [
                {"saved": b, "cpu": a} for a, b, match in zip(actual, expected, matches, strict=True) if not match
            ],
        }
        print(
            name,
            record["file_bytes"],
            record["tensor_bytes"],
            evaluation["end_to_end_correct"],
            all(matches),
            flush=True,
        )
    write_json("weight_inventory", all_weights)
    write_json("audit_result", result)
    compact = [
        {
            "id": row["id"],
            "task": row["task"],
            "user": row["user"],
            "variants": {
                name: {
                    "action": data["records"][i]["action_trace"]["raw"],
                    "final": data["records"][i]["answer"],
                    "correct": data["records"][i]["end_to_end_correct"],
                    "generation_ids_sha256": sha(json.dumps(ids(data["records"][i])).encode()),
                }
                for name, data in checked.items()
            },
        }
        for i, row in enumerate(splits["test"])
    ]
    write_json("all_rows_inspection", compact)
    print("CPU audit completed; no GPU timing reproduced.", flush=True)


if __name__ == "__main__":
    main()
