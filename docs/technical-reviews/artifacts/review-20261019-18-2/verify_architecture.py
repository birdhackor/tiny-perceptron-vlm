import copy
import ast
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron import attention
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().contiguous().numpy().tobytes())
    return value.hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
rows = []
for width, layers, heads in ((16, 2, 1), (8, 1, 1), (16, 1, 1), (64, 2, 1), (16, 1, 2), (32, 1, 2)):
    config = ModelConfig(width=width, layers=layers, heads=heads)
    model = TinyLM(config)
    count = sum(p.numel() for p in model.parameters())
    formula = (2 * config.vocab_size + config.max_length + 2) * width + layers * (12 * width**2 + 9 * width)
    assert count == formula
    assert all(p.requires_grad for p in model.parameters())
    rows.append({
        "config": asdict(config),
        "parameters": count,
        "fp32_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "formula": formula,
        "dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "parameters_by_name": {n: {"shape": list(p.shape), "numel": p.numel()} for n, p in model.named_parameters()},
    })

shapes = []
original_attention = attention.manual_attention
for width, layers in ((16, 2), (8, 1)):
    model = TinyLM(ModelConfig(width=width, layers=layers))
    record = {"width": width, "layers": layers, "attention": []}

    def capture(q, k, v, allowed):
        out, weights = original_attention(q, k, v, allowed)
        record["attention"].append({
            "q_shape": list(q.shape),
            "attention_score_shape": list((q @ k.transpose(-2, -1)).shape),
            "weights_shape": list(weights.shape),
            "weights_numel": weights.numel(),
        })
        return out, weights

    attention.manual_attention = capture
    try:
        with torch.no_grad():
            ids = torch.tensor([[1, 8, 9, 10, 11, 2]], dtype=torch.long)
            record["hidden_shape"] = list(model.embedding(ids).shape)
            record["logits_shape"] = list(model(ids)["logits"].shape)
    finally:
        attention.manual_attention = original_attention
    shapes.append(record)
assert shapes[0]["attention"][0]["weights_shape"] == shapes[1]["attention"][0]["weights_shape"]
assert shapes[0]["logits_shape"] == shapes[1]["logits_shape"]

record_path = ROOT / "docs/course-experiments/results/distillation.json"
stored = json.loads(record_path.read_bytes())
case = stored["results"]["tasks"]["attributes"]
stored_runs = {}
copies = {}
for width in (16, 32):
    runs = {method: case["runs"][f"w{width}_{method}"] for method in ("ce", "teacher_hard", "ce_kl")}
    assert len({r["training"]["initialization_sha256"] for r in runs.values()}) == 1
    assert len({r["storage"]["parameter_count"] for r in runs.values()}) == 1
    assert all(r["storage"]["retained_float_modules"].count("output") == 1 for r in runs.values())
    stored_runs[str(width)] = {
        method: {"storage": r["storage"], "training": {k: r["training"][k] for k in (
            "initialization_sha256", "batch_plan_sha256", "effective_supervised_tokens", "steps", "batch_size", "objective"
        )}}
        for method, r in runs.items()
    }
    initial = TinyLM(ModelConfig(width=width, layers=1, heads=2))
    students = {method: copy.deepcopy(initial) for method in runs}
    before = {method: fingerprint(model) for method, model in students.items()}
    assert len(set(before.values())) == 1
    original_shapes = {n: list(p.shape) for n, p in initial.named_parameters()}
    ids = torch.tensor([[1, 8, 9, 10]], dtype=torch.long)
    labels = torch.tensor([[-100, 9, 10, 2]], dtype=torch.long)
    for method, model in students.items():
        logits = model(ids)["logits"]
        ce = masked_loss(logits, labels)
        loss = ce
        if method == "ce_kl":
            teacher_logits = torch.randn_like(logits)
            loss = 0.5 * ce + 0.5 * distillation_kl(logits, teacher_logits, labels, temperature=2.0)
        loss.backward()
        assert {n: list(p.shape) for n, p in model.named_parameters()} == original_shapes
        assert len([m for n, m in model.named_modules() if n == "output"]) == 1
    copies[str(width)] = {"initialization_fingerprints": before, "output_shape": list(initial.output.weight.shape), "schema_unchanged_after_backward": True}

inputs = (
    "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
    "tiny_perceptron/data.py", "tiny_perceptron/alignment.py", "scripts/course_experiments/compression.py",
    "docs/course-experiments/results/distillation.json", "docs/course-review-20261019/freeze-01/sources/18.2.md",
)
code_hashes = {p: sha(ROOT / p) for p in inputs}
matching_recorded_code = {p: code_hashes[p] == stored["code_sha256"][p] for p in inputs if p in stored["code_sha256"]}
assert all(value for path, value in matching_recorded_code.items() if path.startswith("tiny_perceptron/"))
recorded_script = subprocess.run(
    ["git", "show", "dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17:scripts/course_experiments/compression.py"],
    cwd=ROOT, check=True, capture_output=True,
).stdout
assert hashlib.sha256(recorded_script).hexdigest() == stored["code_sha256"]["scripts/course_experiments/compression.py"]
historical_text = recorded_script.decode()
historical_functions = {node.name: ast.get_source_segment(historical_text, node) for node in ast.parse(historical_text).body if isinstance(node, ast.FunctionDef) and node.name in ("_distill_case", "_fit_text")}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-2/verify_architecture.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "byte_tokenizer": {"vocab_size": ByteTokenizer.vocab_size, "specials": list(SPECIALS), "byte_id_count": len(set(ByteTokenizer().encode(bytes(range(128)).decode("ascii"))))},
    "parameter_formula": "N = (2*V + P + 2)*w + L*(12*w*w + 9*w), V=264, P=128, untied LayerNorm GELU MHA",
    "models": rows,
    "shape_probe": {"batch_size": 1, "sequence_length": 6, "heads": 1, "dtype": "float32", "models": shapes},
    "stored_experiment": {"path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path), "environment": {k: stored[k] for k in ("python_version", "torch_version", "device", "gpu", "seed", "revision")}, "teacher_config": case["teacher_provenance"]["config"], "teacher_storage": case["teacher_storage"], "runs": stored_runs},
    "copy_and_loss_probe": {"seed": 42, "updates": 0, "input_shape": [1, 4], "valid_target_tokens": 3, "temperature": 2.0, "alpha": 0.5, "models": copies},
    "input_sha256": code_hashes,
    "recorded_code_matches_current": matching_recorded_code,
    "recorded_script": {"command": "git show dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17:scripts/course_experiments/compression.py", "sha256": hashlib.sha256(recorded_script).hexdigest().lower(), "initial_copy_code_present": "copy.deepcopy(initial)" in historical_text, "functions": historical_functions},
    "api": {"parameters_source_file": inspect.getsourcefile(torch.nn.Module.parameters), "float32_element_size": torch.empty(1, dtype=torch.float32).element_size(), "numel_example": torch.zeros(2, 3).numel()},
    "result": "所有參數數字與實驗紀錄一致；特徵寬度縮小時，每層 attention score 與輸出 logits 的形狀未縮小。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
