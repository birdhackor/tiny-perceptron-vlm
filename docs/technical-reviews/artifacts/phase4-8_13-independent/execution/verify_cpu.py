"""Independent bounded checks: algebra, tutorial variants, loader contract, raw run metadata.

No optimization, training data downloads, historical model inference, or retained weights.
"""
import ast
import copy
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.model import ModelConfig, TinyLM

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
output = {"environment": {"python": sys.version, "torch": str(torch.__version__),
          "torch_git": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda),
          "cwd": str(Path.cwd()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "scope": "Fixed matrix algebra and temporary synthetic loader inputs only; no optimizer or LM forward calls."}

scalar = []
for alpha in (2, 4, 6):
    delta = Decimal(alpha) / Decimal(2) * Decimal("0.3")
    scalar.append({"rank": 2, "alpha": alpha, "delta": str(delta), "effective_weight": str(Decimal("0.7") + delta)})
assert [x["effective_weight"] for x in scalar] == ["1.0", "1.3", "1.6"]
output["scalar_exact_decimal"] = scalar

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
with torch.no_grad():
    layer.b.fill_(1.0)
    a, b, w = layer.a.clone(), layer.b.clone(), layer.base.weight.clone()
    d1 = layer.merged_weight() - layer.base.weight
    variants = []
    for alpha, factor in ((4, 2), (6, 3)):
        layer.alpha = alpha
        delta = layer.merged_weight() - layer.base.weight
        error = float((delta - factor * d1).abs().max())
        close = torch.allclose(delta, factor * d1, atol=1e-6, rtol=0)
        assert close and torch.equal(a, layer.a) and torch.equal(b, layer.b) and torch.equal(w, layer.base.weight)
        variants.append({"alpha": alpha, "factor": factor, "max_abs_error": error, "allclose": close,
                         "delta": delta.tolist(), "direct_BA_scale_error": float((delta - (b @ a) * alpha / 2).abs().max())})
    output["fixed_rank_variants"] = {"A_shape": list(a.shape), "B_shape": list(b.shape), "W_shape": list(w.shape),
           "A": a.tolist(), "B": b.tolist(), "W": w.tolist(), "delta1": d1.tolist(),
           "A_matrix_rank": int(torch.linalg.matrix_rank(a)), "B_matrix_rank": int(torch.linalg.matrix_rank(b)),
           "BA_matrix_rank": int(torch.linalg.matrix_rank(b @ a)), "rank_setting": layer.rank, "variants": variants,
           "parameter_update_count": 0}
    # Allclose with rtol=0 ignores operand magnitude: a single excessive error fails.
    zero = torch.zeros(2, dtype=torch.float64)
    below = torch.tensor([0.5e-6, 0.0], dtype=torch.float64)
    above = torch.tensor([1.5e-6, 0.0], dtype=torch.float64)
    output["tolerance_probe"] = {"atol": 1e-6, "rtol": 0,
       "below": torch.allclose(below, zero, atol=1e-6, rtol=0),
       "above": torch.allclose(above, zero, atol=1e-6, rtol=0),
       "large_operand_difference": torch.allclose(torch.tensor([1e6 + .001], dtype=torch.float64),
                                                 torch.tensor([1e6], dtype=torch.float64), atol=1e-6, rtol=0)}
    assert output["tolerance_probe"] == {"atol": 1e-6, "rtol": 0, "below": True, "above": False,
                                          "large_operand_difference": False}
    # A rank change also changes factors. This explicit counterexample is algebra, not trained performance.
    a2 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    b2 = torch.ones(3, 2)
    a4 = torch.cat((a2, torch.tensor([[2., 2., 0., 0.], [2., 2., 0., 0.]])))
    b4 = torch.ones(3, 4)
    d2, d4 = b2 @ a2, (b4 @ a4) * 0.5
    assert not torch.equal(d4, d2 * 0.5)
    output["rank_counterexample"] = {"alpha": 2, "rank2_shapes": [list(a2.shape), list(b2.shape)],
       "rank4_shapes": [list(a4.shape), list(b4.shape)], "scales": [1.0, 0.5],
       "rank2_delta": d2.tolist(), "rank4_delta": d4.tolist(), "is_half": torch.equal(d4, d2 * .5),
       "scope": "Changing factors invalidates a scaling-only inference; no trained rank selection was attempted."}

# The hash is independently re-derived on explicit synthetic FP32 names/shapes/bytes.
sample = {"z": torch.tensor([1., 2.], dtype=torch.float32), "a": torch.tensor([[3.]], dtype=torch.float32)}
manual = hashlib.sha256(b"a" + b"(1, 1)" + struct.pack("=f", 3.) + b"z" + b"(2,)" + struct.pack("=ff", 1., 2.)).hexdigest()
assert base_state_sha256(sample) == manual == base_state_sha256(dict(reversed(list(sample.items()))))
changed = {k: v.clone() for k, v in sample.items()}
changed["z"][0] += 1
assert base_state_sha256(changed) != manual
output["state_fingerprint"] = {"manual_sha256": manual, "helper_sha256": base_state_sha256(sample),
       "dict_order_invariant": True, "one_value_change_detected": True,
       "encoding": "sorted names UTF-8 + str(tuple(shape)) UTF-8 + contiguous original FP32 native bytes"}

config = ModelConfig(vocab_size=12, width=4, layers=1, heads=1, max_length=4)
torch.manual_seed(13)
model = TinyLM(config)
state = {k: v.detach().clone() for k, v in model.state_dict().items()}
payload = {"format_version": "lora-v1", "config": asdict(config), "scaling": "alpha/rank",
  "base_sha256": base_state_sha256(state), "rank": 2, "alpha": 2,
  "adapter": {"output": {"a": torch.full((2, 4), .1), "b": torch.full((12, 2), .2), "rank": 2, "alpha": 2}}}
cases = []
with tempfile.TemporaryDirectory(prefix="8_13-contract-", dir=HERE) as temporary:
    path = Path(temporary) / "synthetic-only.pt"
    torch.save(payload, path)
    successful = copy.deepcopy(model)
    info = load_lora_adapter(successful, path)
    assert isinstance(successful.output, LoRALinear) and info["scaling"] == "alpha/rank"
    assert info["modules"][0]["rank"] == 2 and info["modules"][0]["alpha"] == 2
    cases.append({"case": "valid", "result": "accepted", "metadata": info})
    for name in ("scaling", "config", "base_sha256", "module_path", "rank_shape", "alpha_setting", "actual_base"):
        bad = copy.deepcopy(payload)
        fresh = copy.deepcopy(model)
        if name == "scaling": bad["scaling"] = "alpha/sqrt(rank)"
        elif name == "config": bad["config"]["max_length"] = 5
        elif name == "base_sha256": bad["base_sha256"] = "0" * 64
        elif name == "module_path": bad["adapter"]["missing.output"] = bad["adapter"].pop("output")
        elif name == "rank_shape": bad["adapter"]["output"]["a"] = torch.zeros(3, 4)
        elif name == "alpha_setting": bad["alpha"] = 4
        elif name == "actual_base":
            with torch.no_grad(): fresh.output.weight[0, 0] += 1
        torch.save(bad, path)
        try:
            load_lora_adapter(fresh, path, base_state=state)
        except ValueError as error:
            assert not any(isinstance(m, LoRALinear) for m in fresh.modules())
            cases.append({"case": name, "result": "rejected_before_insertion", "message": str(error)})
        else:
            raise AssertionError(f"Expected rejection: {name}")
assert not list(HERE.glob("8_13-contract-*"))
output["loader_contract"] = {"cases": cases, "temporary_weights_deleted": True,
      "scope": "Synthetic native TinyLM and adapter only; no historic artifact load, model forward, or generation."}

run_path = ARTIFACT / "inputs/docs/course-experiments/results/lora.json"
run = json.loads(run_path.read_bytes())
results = run["results"]
historical_path = ARTIFACT / "sources/behavior-a7cdffec.py"
historic = historical_path.read_bytes()
assert hashlib.sha256(historic).hexdigest() == run["code_sha256"]["scripts/course_experiments/behavior.py"]
tree = ast.parse(historic)
adder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_add_lora")
assert [ast.literal_eval(x) for x in adder.args.defaults] == [4, 4]
recipe = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_lora")
calls = [n for n in ast.walk(recipe) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_add_lora"]
assert len(calls) == 1 and len(calls[0].args) == 1 and not calls[0].keywords
assert results["rank"] == results["alpha"] == 4 and results["scaling"] == "alpha/rank"
assert set(results["runs"]) == {"concise", "vivid"}
summaries = {name: {"steps": data["training"]["steps"], "effective_tokens": data["training"]["effective_tokens"],
                   "layer_count": len(data["layers"])} for name, data in results["runs"].items()}
assert all(x["steps"] == 450 and x["layer_count"] == 11 for x in summaries.values())
code_match = {}
for name in ("tiny_perceptron/alignment.py", "tiny_perceptron/adapters.py", "scripts/course_experiments/behavior.py"):
    current_hash = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    code_match[name] = {"recorded_sha256": run["code_sha256"][name], "current_sha256": current_hash,
                        "matches_recorded": current_hash == run["code_sha256"][name]}
assert code_match["tiny_perceptron/alignment.py"]["matches_recorded"]
assert code_match["tiny_perceptron/adapters.py"]["matches_recorded"]
output["existing_run_metadata"] = {"json_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
    "run_revision": run["revision"], "run_id": run["modal"]["run_id"], "original_device": run["device"],
    "original_torch": run["torch_version"], "base_sha256": results["base_sha256"],
    "rank": results["rank"], "alpha": results["alpha"], "scaling": results["scaling"],
    "computed_scale": results["alpha"] / results["rank"], "adapter_runs": summaries,
    "denominators": {"adapter_styles": 2, "rank_settings": 1, "scaling_formulas": 1, "steps_per_adapter": 450},
    "code_versions": code_match, "historical_recipe_verified": True,
    "scope": "Recomputed JSON configuration and checked versioned writer source; no historical checkpoint download or inference."}
(HERE / "cpu-results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
