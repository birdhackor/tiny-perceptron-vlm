"""核對凍結算例、練習與既有量化實驗的實作範圍。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.compression import _QATLinear  # noqa: E402
from tiny_perceptron.quantization import QuantizedLinear  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in b] for row in a]


def error(a, b):
    return sum(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)) / 4


def decimal(a):
    return [[float(x) for x in row] for row in a]


def tensor_matches(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(decimal(expected)), atol=1e-6, rtol=0)


frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/17.11.md"
raw = frozen.read_bytes()
code = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)[0]
namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(code, str(frozen), "exec"), namespace)

x = [[F(7, 10), F(2, 10)], [F(-7, 10), F(12, 10)]]
q = [[F(round(value / F(1, 2)), 2) for value in row] for row in x]
reference, weight_only, both = product(x, x), product(x, q), product(q, q)
assert reference == [[F(53, 100), F(-1, 4)], [F(-1, 4), F(193, 100)]]
assert weight_only == [[F(35, 100), F(-15, 100)], [F(-35, 100), F(155, 100)]]
assert both == [[F(1, 4), F(-1, 4)], [F(-1, 4), F(5, 4)]]
assert error(reference, weight_only) == F(19, 100)
assert error(reference, both) == F(24, 100)
for name, expected in (("reference", reference), ("weight_only", weight_only), ("both", both), ("qx", q), ("qw", q)):
    tensor_matches(namespace[name], expected)
    assert namespace[name].dtype == torch.float32

exercise = namespace["x"].clone()
exercise[0] = torch.tensor([1.0, 0.0])
exercise_quantized = namespace["simulate_quantization"](exercise)
exercise_weight_only = exercise @ namespace["qw"].T
exercise_both = exercise_quantized @ namespace["qw"].T
assert torch.equal(exercise[0], exercise_quantized[0])
assert torch.equal(exercise_weight_only[0], torch.tensor([0.5, -0.5]))
assert torch.equal(exercise_weight_only[0], exercise_both[0])

dx = namespace["qx"] - namespace["x"]
dw = namespace["qw"] - namespace["w"]
weight_error = namespace["x"] @ dw.T
input_extra = dx @ namespace["qw"].T
torch.testing.assert_close(namespace["both"] - namespace["weight_only"], input_extra, atol=1e-6, rtol=0)
torch.testing.assert_close(namespace["both"] - namespace["reference"], weight_error + input_extra, atol=1e-6, rtol=0)

implementation_paths = ["tiny_perceptron/quantization.py", "scripts/course_experiments/compression.py"]
records = {}
historical_sources = {}
for name in ("quantization", "qat"):
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    record = json.loads(path.read_text())
    results = record["results"]
    matches = {p: record["code_sha256"][p] == sha(ROOT / p) for p in implementation_paths}
    for relative, matches_current in matches.items():
        if matches_current:
            continue
        url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + record["revision"] + "/" + relative
        if url not in historical_sources:
            command = ["curl", "-fsSL", "--max-time", "30", url]
            original = subprocess.run(command, check=True, capture_output=True).stdout
            assert hashlib.sha256(original).hexdigest() == record["code_sha256"][relative]
            definitions = ("_packed", "_QATLinear", "_qat_layers", "run_quantization", "run_qat")
            def nodes(raw):
                return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            current_nodes, original_nodes = nodes((ROOT / relative).read_bytes()), nodes(original)
            same = {key: current_nodes[key] == original_nodes[key] for key in definitions}
            assert all(same.values())
            historical_sources[url] = {"command_argv": command, "sha256": hashlib.sha256(original).hexdigest(), "matches_recorded_sha256": True, "relevant_definitions_ast_equal": same, "inspection_note": "量化相關定義與目前檔案相同；全文不落地保存。"}
    records[name] = {
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "revision": record["revision"], "status": record["status"],
        "original_environment": {k: str(record[k]) for k in ("device", "torch_version", "python_version", "seed")},
        "limitations": results["limitations"],
        "activation_quantization": results.get("activation_quantization"),
        "fake_quantization": results.get("fake_quantization"),
        "implementation_hashes_match_record": matches,
    }
assert records["quantization"]["limitations"][0] == "weight-only per-channel Linear quantization; no activation calibration"
assert records["qat"]["activation_quantization"] is False

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(namespace["w"])
forward_checks = {}
for name, candidate in (("packed", QuantizedLinear(layer, bits=4)), ("qat", _QATLinear(layer, bits=4))):
    with patch.object(functional, "linear", wraps=functional.linear) as call:
        output = candidate(namespace["x"])
    passed_input, passed_weight = call.call_args.args[:2]
    assert passed_input is namespace["x"]
    assert passed_input.dtype == torch.float32
    assert passed_weight.dtype == torch.float32
    forward_checks[name] = {
        "input_object_unchanged": passed_input is namespace["x"],
        "input_values": passed_input.tolist(), "input_dtype": str(passed_input.dtype),
        "weight_dtype": str(passed_weight.dtype), "output_dtype": str(output.dtype),
        "forward_formula": "F.linear(x, reconstructed_weight, bias)",
    }

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-11/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "program_sha256": sha(Path(__file__)), "source_sha256": hashlib.sha256(raw).hexdigest(),
    "exact_rational": {"reference": [[str(v) for v in row] for row in reference], "weight_only": [[str(v) for v in row] for row in weight_only], "both": [[str(v) for v in row] for row in both], "mae_weight_only": str(error(reference, weight_only)), "mae_both": str(error(reference, both))},
    "quantized_x_and_w": decimal(q), "frozen_program_stdout": printed.getvalue(),
    "exercise": {"first_row_weight_only": exercise_weight_only[0].tolist(), "first_row_both": exercise_both[0].tolist(), "input_first_row_unchanged": True},
    "error_decomposition": {"weight_error": weight_error.tolist(), "extra_input_term": input_extra.tolist(), "identity": "Qx @ Qw.T - x @ w.T = x @ (Qw-w).T + (Qx-x) @ Qw.T", "max_abs_residual": float(((namespace["both"] - namespace["reference"]) - weight_error - input_extra).abs().max())},
    "implementation_sha256": {p: sha(ROOT / p) for p in implementation_paths},
    "experiment_records": records, "historical_source_verification": historical_sources, "weight_only_forward_checks": forward_checks,
    "result": "所有精確算例、浮點容忍差、練習及權重量化範圍核對通過；沒有重跑訓練或 GPU 分數。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
