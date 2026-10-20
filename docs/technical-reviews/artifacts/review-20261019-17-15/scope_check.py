"""核對實驗題組的任務範圍，以及權重量化保留的浮點模組。"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.prepare_data import generate_records
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers

command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-15/scope_check.py"
tok_pairs = {(r["messages"][-2]["content"], r["messages"][-1]["content"]) for r in generate_records("attributes-sft")}
receipts = {}
for name in ("quantization", "qat"):
    path = ROOT / f"docs/course-experiments/results/{name}.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    variants = {}
    for variant, run in report["results"]["runs"].items():
        variants[variant] = {}
        for split in ("validation", "test"):
            samples = run[split]["generated_samples"]
            assert all((s["question"], s["expected"]) in tok_pairs for s in samples)
            variants[variant][split] = {
                "examples": len(samples),
                "all_pairs_in_attribute_generator": True,
                "questions": [s["question"] for s in samples],
            }
    receipts[name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "data": report["results"]["data"], "variants": variants}
assert receipts["quantization"]["data"]["sha256"] == receipts["qat"]["data"]["sha256"]

torch.manual_seed(42)
original = nn.Sequential(nn.Embedding(8, 4), nn.LayerNorm(4), nn.Linear(4, 3)).eval()
converted = replace_linear_layers(copy.deepcopy(original), bits=4)
assert isinstance(converted[2], QuantizedLinear)
assert isinstance(converted[0], nn.Embedding) and isinstance(converted[1], nn.LayerNorm)
assert torch.equal(original[0].weight, converted[0].weight)
assert torch.equal(original[1].weight, converted[1].weight)
assert torch.equal(original[2].bias, converted[2].bias)
assert converted[2].values.dtype == torch.uint8
assert converted[2].scale.dtype == torch.float32
with torch.no_grad():
    output = converted(torch.tensor([[1, 2]]))
assert output.dtype == torch.float32 and output.shape == (1, 2, 3)
result = {
    "command": command,
    "result": "兩份紀錄的全部 validation/test 問答均符合合成屬性題組；小型模組轉換只替換 Linear，embedding、LayerNorm 與 bias 保留浮點數值。",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "experiment_scope": receipts,
    "weight_only_probe": {"input_ids": [[1, 2]], "output_shape": list(output.shape), "output_dtype": str(output.dtype), "packed_weight_dtype": str(converted[2].values.dtype), "scale_dtype": str(converted[2].scale.dtype), "retained_float_modules": [type(converted[0]).__name__, type(converted[1]).__name__], "bias_unchanged": True},
    "limitations": "只核對紀錄的涵蓋範圍與小型轉換的行為，不重新測得真模型品質、GPU 記憶體或速度。",
}
target = Path(__file__).with_name("scope-check.json")
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"result": result["result"], "path": str(target.relative_to(ROOT))}, ensure_ascii=False))
