"""5.9 的有界 CPU 診斷：執行凍結正文，不訓練、不使用 GPU。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

import tiny_perceptron.attention as attention
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, pad_batch, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
REVISION = "b52935d99f58b694cd932ac158c10c4d8d92d4c2"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    result = {
        "started_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_5_9",
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_5_9-v1/verify_lesson.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_git_version": torch.version.git_version,
            "device": "cpu",
            "platform": platform.platform(),
            "threads": str(torch.get_num_threads()),
            "interop_threads": str(torch.get_num_interop_threads()),
        },
        "training_performed": False,
        "gpu_used": False,
    }
    freeze = json.loads(FREEZE.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "5.9")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    body = raw.decode()
    blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
    assert len(blocks) == 1
    code = blocks[0]
    assert page["figures_sha256"] == {}
    assert not re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
    result["source"] = {"snapshot": page["snapshot"], "sha256": digest(raw), "code_sha256": digest(code.encode()), "figures": {}}
    stdout, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, page["snapshot"] + ":python-block-1", "exec"), namespace)
    model = namespace["model"]
    result["embedded_code"] = {
        "stdout": stdout.getvalue(),
        "parameters": namespace["parameters"],
        "fp32_parameter_bytes": namespace["parameters"] * 4,
        "average_seconds": namespace["average_seconds"],
        "input_shape": list(namespace["x"].shape),
        "input_dtype": str(namespace["x"].dtype),
        "parameter_dtype": sorted({str(p.dtype) for p in model.parameters()}),
        "parameter_device": sorted({str(p.device) for p in model.parameters()}),
        "timed_iterations": 10,
        "warmup_iterations": 1,
        "training_flag_after_eval": model.training,
    }
    assert namespace["parameters"] == 6104
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    result["parameter_components"] = {
        "embedding": model.embedding.weight.numel(),
        "position": model.position.weight.numel(),
        "final_norm": sum(p.numel() for p in model.final_norm.parameters()),
        "output": model.output.weight.numel(),
        "block": sum(p.numel() for p in model.blocks[0].parameters()),
        "actual_storage_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "float32_bits": torch.finfo(torch.float32).bits,
        "float32_element_bytes": torch.zeros((), dtype=torch.float32).element_size(),
        "vocabulary": ByteTokenizer.vocab_size,
        "special_count": len(SPECIALS),
    }
    assert list(result["parameter_components"][k] for k in ["embedding", "position", "final_norm", "output", "block"]) == [2112, 1024, 16, 2112, 840]
    assert result["parameter_components"]["actual_storage_bytes"] == 24416
    other_stdout, exercise = io.StringIO(), {}
    assert code.count("[[1, 2, 3]]") == 1
    with contextlib.redirect_stdout(other_stdout):
        exec(compile(code.replace("[[1, 2, 3]]", "[[1,2,3,1,2,3]]"), "5.9-six-position-exercise", "exec"), exercise)
    assert exercise["parameters"] == 6104
    result["six_position_exercise"] = {
        "stdout": other_stdout.getvalue(),
        "parameters": exercise["parameters"],
        "fp32_parameter_bytes": exercise["parameters"] * 4,
        "average_seconds": exercise["average_seconds"],
        "input_shape": list(exercise["x"].shape),
        "timed_iterations": 10,
        "warmup_iterations": 1,
        "timing_scope": "Fresh CPU model, eval, no_grad, same ten-forward mean; no cross-length speed conclusion.",
    }
    normal = model(namespace["x"])["logits"]
    with torch.no_grad():
        untracked = model(namespace["x"])["logits"]
    result["grad_mode"] = {"eval_output_requires_grad": normal.requires_grad, "no_grad_output_requires_grad": untracked.requires_grad, "parameter_grads_all_none": all(p.grad is None for p in model.parameters())}
    assert normal.requires_grad and not untracked.requires_grad
    assert all(p.grad is None for p in model.parameters())
    del normal, untracked
    shapes = []
    original = attention.manual_attention

    def record_shapes(q, k, v, allowed):
        shapes.append({"q": list(q.shape), "k": list(k.shape), "scores": list((q @ k.transpose(-2, -1)).shape), "allowed": list(allowed.shape)})
        return original(q, k, v, allowed)

    attention.manual_attention = record_shapes
    try:
        with torch.no_grad():
            model(torch.tensor([[1, 2, 3]]))
            model(torch.tensor([[1, 2, 3, 1, 2, 3]]))
    finally:
        attention.manual_attention = original
    assert [s["scores"] for s in shapes] == [[1, 1, 3, 3], [1, 1, 6, 6]]
    result["attention_shape_diagnostic_outside_timing"] = shapes
    large = TinyLM(ModelConfig(width=64, layers=2))
    result["historical_recipe_structure"] = {"parameters": large.description()["parameters"], "actual_fp32_parameter_bytes": sum(p.numel() * p.element_size() for p in large.parameters())}
    assert result["historical_recipe_structure"] == {"parameters": 141568, "actual_fp32_parameter_bytes": 566272}
    del large
    historical = json.loads((ROOT / "docs/course-experiments/results/text_foundation.json").read_bytes())
    historical_names = ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py", "scripts/prepare_data.py"]
    old = {name: subprocess.check_output(["git", "show", REVISION + ":" + name], cwd=ROOT) for name in historical_names}
    historical_hashes = {name: digest(raw) for name, raw in old.items()}
    for name in historical_names[:-1]:
        assert historical_hashes[name] == historical["code_sha256"][name]
    result["historical_code"] = {"revision": REVISION, "full_file_sha256": historical_hashes}
    pure_namespace = {"random": random, "json": json, "hashlib": hashlib, "torch": torch, "ByteTokenizer": ByteTokenizer, "shifted": shifted, "render_chat": render_chat}
    targets = {"scripts/course_experiments/common.py": {"split_records", "text_examples"}, "scripts/prepare_data.py": {"generate_records"}}
    for name, functions in targets.items():
        tree = ast.parse(old[name].decode())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        assert {n.name for n in nodes} == functions
        exec(compile(ast.Module(body=nodes, type_ignores=[]), REVISION + ":" + name, "exec"), pure_namespace)
    records = pure_namespace["generate_records"]("toy-text")
    parts = pure_namespace["split_records"](records, seed=42)
    examples = pure_namespace["text_examples"](parts["train"], mode="text", max_length=128)
    train_hash = digest(json.dumps(parts["train"], sort_keys=True, ensure_ascii=False).encode())
    assert train_hash == historical["results"]["training"]["records_sha256"]
    split_checks = {}
    for split, rows in parts.items():
        raw_split = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        split_checks[split] = {"records": len(rows), "sha256": digest(raw_split)}
        assert split_checks[split]["sha256"] == historical["results"]["data"][split]["sha256"]
    sampler, count = random.Random(42), 0
    for _ in range(600):
        batch = sampler.choices(examples, k=16)
        x, y, valid = pad_batch(batch)
        count += int((y != IGNORE).sum())
    assert count == 342462 == historical["results"]["training"]["effective_tokens"]
    result["sampling_only_audit_no_training"] = {"seed": 42, "steps": 600, "batch_size": 16, "draws_with_replacement": 9600, "unique_training_records": len(parts["train"]), "examples": len(examples), "example_valid_target_counts": [int((y != IGNORE).sum()) for x, y in examples], "effective_target_exposures": count, "records_sha256": train_hash, "split_checks": split_checks, "label_scope": "UTF-8 byte targets plus EOS; padding IGNORE=-100 excluded; repeated exposures, not distinct text or articles."}
    result["preserved_measurement_audit_not_reproduction"] = {k: historical[k] for k in ["revision", "gpu", "device", "seed", "torch_version", "python_version", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status"]}
    result["preserved_measurement_audit_not_reproduction"]["training"] = {k: historical["results"]["training"][k] for k in ["steps", "seconds", "effective_tokens", "parameters", "records"]}
    result["preserved_measurement_audit_not_reproduction"]["scaling_names"] = list(historical["results"]["scaling"])
    assert round(historical["results"]["training"]["seconds"], 4) == 6.1885
    assert round(historical["elapsed_seconds"], 4) == 16.0354
    result["completed_at"] = datetime.now(UTC).isoformat()
    result["result"] = "All assertions passed; exact embedded code, six-position edit, structure, grad modes, attention shapes, historical data fingerprints and 342462 target exposures checked on CPU. No training or GPU measurement."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
