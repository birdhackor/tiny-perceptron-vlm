"""接續核對保存的版本、名詞索引與實驗來源；既有證據保持原樣。"""

import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import MoEFFN

BASE = Path(__file__).parent
result = {"command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-glossary/verify_supplement-01.py", "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"}}
old = json.loads((BASE / "verification-output.json").read_text())
result["source_hash_comparison"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in old["repository_files"].items()}
assert all(result["source_hash_comparison"].values())
body = Path("docs/course-review-20261019/freeze-01/sources/glossary.md").read_text()
index = body.split("### 英文與縮寫索引", 1)[1]
tokens = []
for line in index.splitlines():
    if not line.startswith("**"):
        continue
    for item in line.split("　", 1)[1].split(" · "):
        token = "decode" if item.startswith("decode") else re.match(r"\[([^]]+)\]", item).group(1)
        tokens.append(token)
key = lambda word: re.sub(r"[^a-z0-9]", "", word.lower())
result["index_order"] = {"count": len(tokens), "sorted": tokens == sorted(tokens, key=key), "unique": len(tokens) == len(set(tokens))}
assert result["index_order"]["sorted"] and result["index_order"]["unique"]
tree = ast.parse(Path("scripts/course_experiments/architecture.py").read_text())
node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_parameter_budget")
namespace = {"MoEFFN": MoEFFN}
exec(compile(ast.Module(body=[node], type_ignores=[]), "architecture.py", "exec"), namespace)
model = TinyLM(ModelConfig(width=4, layers=2, experts=4, top_k=2, max_length=8))
budget = namespace["_parameter_budget"](model)
non_expert = budget["total_parameters"]-budget["expert_parameters"]
expected = non_expert + budget["expert_parameters"]//2
assert budget["active_parameter_proxy_per_token"] == expected
result["moe_budget"] = {**budget, "expected": expected, "non_expert_parameters": non_expert, "one_layer_active_ffn": sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())*2 + sum(p.numel() for p in model.blocks[0].ffn.router.parameters())}
result["teacher_provenance"] = {}
for filename in ["distillation.json", "multimodal_distillation.json"]:
    path = Path("docs/course-experiments/results") / filename
    if path.exists():
        data = json.loads(path.read_text())
        matches = []
        def visit(value, location=""):
            if isinstance(value, dict):
                for field, item in value.items():
                    if field == "teacher_provenance":
                        matches.append({"location": location+".teacher_provenance", "value": item})
                    else:
                        visit(item, location+"."+field)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    visit(item, location+f"[{i}]")
        visit(data)
        result["teacher_provenance"][str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": matches}
result["product_training"] = {}
for name in ["initial-b-execution.json", "initial-c-execution.json"]:
    path = Path("docs/lesson-products/results")/name
    data = json.loads(path.read_text())
    result["product_training"][str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "job": data["job"], "top_level_keys": list(data)}
result["selftrained_initialization"] = json.loads(Path("docs/selftrained/v2-manifest.json").read_text())["initialization"]
result["result"] = "所有補充斷言通過；既有程式證據的來源 hash 均維持一致。"
with (BASE/"supplement-output-01.json").open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
