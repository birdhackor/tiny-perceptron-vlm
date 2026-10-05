"""Inspect only named raw provenance/configuration/step-count JSON pointers."""
import ast
import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1]
raw = (A / "inputs/moe-original.json").read_bytes()
j = json.loads(raw)
paths = ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "scripts/course_experiments/architecture.py"]
result = {"raw_sha256": hashlib.sha256(raw).hexdigest(), "pointer_read_log": ["/revision", "/device", "/seed", "/torch_version", "/python_version"] + ["/code_sha256/" + p.replace("/", "~1") for p in paths], "provenance": {k: j[k] for k in ["revision", "device", "seed", "torch_version", "python_version"]}, "code_versions": {}, "variants": {}}
for p in paths:
    saved = A / "code" / ("experiment-original-" + Path(p).name)
    sha = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert sha == j["code_sha256"][p]
    result["code_versions"][p] = {"recorded": j["code_sha256"][p], "frozen_original": sha, "current_snapshot": hashlib.sha256((A / "code" / ("current-" + Path(p).name)).read_bytes()).hexdigest()}
expected = {"dense_active_top1", "dense_active_top2", "dense_total", "top1_aux0", "top1_aux0.01", "top2_aux0", "top2_aux0.01"}
assert set(j["results"]["variants"]) == expected
result["variant_count"] = len(expected)
for name, value in j["results"]["variants"].items():
    config = value["model"]["config"]
    result["pointer_read_log"].append("/results/variants/" + name + "/model/config")
    assert not set(config) & {"capacity", "capacity_factor", "drop_tokens", "fallback", "reroute"}
    measurements = {k: value["training"][k] for k in ["requested_steps", "steps", "optimizer_updates", "skipped_updates", "effective_tokens"]}
    result["pointer_read_log"] += ["/results/variants/" + name + "/training/" + k for k in measurements]
    assert measurements["steps"] == measurements["requested_steps"] == measurements["optimizer_updates"] == 180
    assert measurements["skipped_updates"] == 0
    result["variants"][name] = {"config": config, "raw_step_measurements": measurements}
tree = ast.parse((A / "code/experiment-original-modern.py").read_bytes())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoEFFN")
forward = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
source = ast.get_source_segment((A / "code/experiment-original-modern.py").read_text(), forward)
assert "torch.where(chosen == i)" in source and "flat[token_rows]" in source and "output.index_add_" in source
assert not any(isinstance(n, ast.Name) and n.id in {"capacity", "capacity_factor", "fallback", "reroute"} for n in ast.walk(forward))
result["code_scope"] = "Original MoEFFN.forward lines 67-84 processes every selected row without a capacity bound; ModelConfig/Block and run_moe configurations inspected. Seven raw configurations and step measurements are archival, not rerun training. Heldout scores, author result notes, and old reports were not read."
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
