from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import platform
import runpy
import sys
import unicodedata

import torch
from tiny_perceptron.natural_concepts import text_error_report

a = Path(__file__).resolve().parent
predictions = json.loads((a / "predictions-before-execution.json").read_text())
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    runpy.run_path(str(a / "original-example.py"))
cases = {}
for name in ("original_example", "exercise", "equal_string_check"):
    case = predictions[name]
    actual = text_error_report(case["reference"], case["prediction"])
    expected = case["expected"]
    selected = {key: actual[key] for key in expected}
    assert selected == expected, (name, selected, expected)
    cases[name] = {"reference": case["reference"], "prediction": case["prediction"], "actual": selected, "matches_saved_prediction": True}
normalization = {"fullwidth_A": unicodedata.normalize("NFKC", "Ａ"), "tai_character": unicodedata.normalize("NFKC", "臺")}
assert normalization == predictions["normalization_example"]["expected"]
result = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_threads": torch.get_num_threads(),
    "original_example_stdout": stdout.getvalue(),
    "cases": cases,
    "normalization_example": {"actual": normalization, "matches_saved_prediction": True},
    "scope": "Executed one small source example and bounded string checks offline; no model, dataset, training or benchmark replication."
}
(a / "cpu-execution-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
