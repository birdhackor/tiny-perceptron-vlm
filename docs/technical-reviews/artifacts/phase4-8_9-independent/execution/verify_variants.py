"""Independent bounded CPU verification: hand-set matrices and payload contracts only.

No training, trained-model loading, TinyLM forward, generation or downloads.
Temporary .pt files are removed by TemporaryDirectory; no neural weights retained.
"""
import contextlib
import copy
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import ByteTokenizer, SPECIALS
from tiny_perceptron.model import ModelConfig, TinyLM
from scripts.course_experiments.behavior import _adapter_state, _merge_lora, _restore_adapter

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute_fence(raw):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(raw, "current-8.9-fence", "exec"), namespace)
    return namespace, output.getvalue()


original = (HERE / "fence-1.py").read_bytes()
namespace, stdout = execute_fence(original)
assert stdout == "切回A的B平均 0.1\nA兩個矩陣都恢復 True\n"
layer = namespace["layer"]
snapshot_a = namespace["adapter_a"]
assert layer.a.data_ptr() != snapshot_a["a"].data_ptr()
assert layer.b.data_ptr() != snapshot_a["b"].data_ptr()
assert not snapshot_a["a"].requires_grad and not snapshot_a["b"].requires_grad
assert not layer.base.weight.requires_grad and not layer.base.bias.requires_grad
assert layer.a.requires_grad and layer.b.requires_grad
variant = original.replace(b'    layer.a.copy_(adapter_a["a"])\n', b"")
assert variant != original
(HERE / "variant-omit-a.py").write_bytes(variant)
bad_namespace, bad_stdout = execute_fence(variant)
assert bad_stdout == "切回A的B平均 0.1\nA兩個矩陣都恢復 False\n"
assert not torch.equal(bad_namespace["layer"].a, bad_namespace["adapter_a"]["a"])
assert torch.equal(bad_namespace["layer"].b, bad_namespace["adapter_a"]["b"])

# Alpha scales the update, not the base or its bias.
with torch.no_grad():
    delta_two = (layer.b @ layer.a) * (layer.alpha / layer.rank)
    layer.alpha = 4
    delta_four = (layer.b @ layer.a) * (layer.alpha / layer.rank)
    assert torch.equal(delta_four, delta_two * 2)
    layer.alpha = 2
    state_keys = list(layer.state_dict())
    assert set(state_keys) == {"a", "b", "base.weight", "base.bias"}
    assert "alpha" not in state_keys and "rank" not in state_keys
    same_mean = layer.b.clone()
    same_mean[0, 0] += 0.01
    same_mean[0, 1] -= 0.01
    assert round(float(same_mean.mean()), 2) == 0.1
    assert not torch.equal(same_mean, layer.b)

# Current real helper on a container of one hand-set linear layer. No language
# model forward is called. Same-rank snapshots are the helper's intended scope.
container = nn.Sequential(copy.deepcopy(layer))
base_before = {k: v.clone() for k, v in container[0].base.state_dict().items()}
adapter_a = _adapter_state(container)
with torch.no_grad():
    container[0].a.fill_(0.2)
    container[0].b.fill_(-0.1)
    container[0].alpha = 4
adapter_b = _adapter_state(container)
_restore_adapter(container, adapter_a)
assert container[0].alpha == 2 and container[0].rank == 2
assert torch.equal(container[0].a, adapter_a["0"]["a"])
assert torch.equal(container[0].b, adapter_a["0"]["b"])
assert all(torch.equal(v, base_before[k]) for k, v in container[0].base.state_dict().items())
merged = _merge_lora(container)
assert isinstance(merged[0], nn.Linear)
assert not any(isinstance(m, LoRALinear) for m in merged.modules())
assert torch.equal(merged[0].bias, base_before["bias"])
assert torch.equal(merged[0].weight, container[0].merged_weight())
delta = container[0].b @ container[0].a
double_applied = merged[0].weight + delta
expected_double = base_before["weight"] + delta * 2
torch.testing.assert_close(double_applied, expected_double, rtol=0, atol=1e-7)
assert not torch.equal(double_applied, merged[0].weight)
# Compare plain matrix expressions on deterministic input, with no model call.
x = torch.tensor([[1.0, -2.0, 3.0, 0.5], [0.0, 1.0, 2.0, -1.0]])
unmerged_expression = x @ base_before["weight"].T + base_before["bias"] + x @ container[0].a.T @ container[0].b.T
merged_expression = x @ merged[0].weight.T + merged[0].bias
expression_difference = float((unmerged_expression - merged_expression).abs().max())
assert expression_difference < 1e-6

# Formal lora-v1 save/load: initialize an untrained tiny model only to inspect
# the loader's metadata/state behavior; never call its forward or generation.
torch.manual_seed(71)
native = TinyLM(ModelConfig(width=4, layers=1, heads=1, max_length=16))
base_state = {k: v.detach().clone() for k, v in native.state_dict().items()}
payload = {
    "format_version": "lora-v1",
    "adapter": {"output": {"a": torch.full((2, 4), 0.2), "b": torch.full((264, 2), 0.1), "rank": 2, "alpha": 2}},
    "config": asdict(native.config),
    "base_sha256": base_state_sha256(base_state),
    "scaling": "alpha/rank",
    "optimizer": {"state": {}, "param_groups": []},
    "step": 0,
    "torch_rng": torch.get_rng_state(),
    "cuda_rng": [],
    "sampler_rng": random.Random(71).getstate(),
}
rejections = {}
with tempfile.TemporaryDirectory(prefix="factual-8_9-") as temporary:
    path = Path(temporary) / "adapter.pt"
    torch.save(payload, path)
    target = copy.deepcopy(native)
    rng_before = torch.get_rng_state().clone()
    info = load_lora_adapter(target, path)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(target.output.a, payload["adapter"]["output"]["a"])
    assert torch.equal(target.output.b, payload["adapter"]["output"]["b"])
    assert target.output.alpha == 2 and target.output.rank == 2
    assert torch.equal(target.output.base.weight, base_state["output.weight"])
    assert not target.output.base.weight.requires_grad
    for label in ("wrong-base", "wrong-a-shape", "wrong-scaling", "already-adapted"):
        damaged = copy.deepcopy(payload)
        candidate = copy.deepcopy(native)
        if label == "wrong-base":
            with torch.no_grad():
                candidate.output.weight[0, 0] += 0.01
        elif label == "wrong-a-shape":
            damaged["adapter"]["output"]["a"] = torch.full((1, 4), 0.2)
        elif label == "wrong-scaling":
            damaged["scaling"] = "alpha/sqrt(rank)"
        else:
            candidate = target
        torch.save(damaged, path)
        try:
            load_lora_adapter(candidate, path)
        except ValueError as error:
            rejections[label] = str(error)
        else:
            raise AssertionError(label + " was accepted")
    temporary_exists_after_context = str(Path(temporary))
assert not Path(temporary_exists_after_context).exists()

# Re-read the immutable recorded experiment JSON, and match its original git
# blobs. This audits existing evidence; none of its training/inference is rerun.
result_path = ROOT / "docs/course-experiments/results/lora.json"
raw_result = result_path.read_bytes()
recorded = json.loads(raw_result)
measurements = recorded["results"]
original_hash_receipts = {}
for name in ["scripts/course_experiments/behavior.py", "tiny_perceptron/alignment.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/training.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]:
    raw = subprocess.check_output(["git", "show", recorded["revision"] + ":" + name], cwd=ROOT)
    assert sha(raw) == recorded["code_sha256"][name]
    original_hash_receipts[name] = {"recorded": sha(raw), "current": sha((ROOT / name).read_bytes()), "same": raw == (ROOT / name).read_bytes()}
probe = [1, 3, 57, 51, 57, 69, 71, 2, 4]
tokens = [SPECIALS[v] if v < 8 else chr(v - 8) for v in probe]
assert ByteTokenizer().decode(probe) == "1+1=?"
assert measurements["a_b_a_max_difference"] == 0.0
assert measurements["adapter_a_b_logit_difference"] > 0
assert measurements["merge_max_difference"] == 6.67572021484375e-06
assert measurements["merge_max_difference"] < 0.0001
assert abs(measurements["merge_max_difference"] - 0.00000668) < 0.000000005
assert measurements["rank"] == measurements["alpha"] == 4
assert measurements["scaling"] == "alpha/rank"

output = {
    "scope": "Original fence and bounded hand-set CPU matrix/payload variants. Existing recorded measurements audited without training, trained weights, language-model inference or generation.",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(), "cwd": str(Path.cwd())},
    "original_fence": {"sha256": sha(original), "stdout": stdout, "matrix_shapes": {"a": list(layer.a.shape), "b": list(layer.b.shape), "base_weight": list(layer.base.weight.shape), "base_bias": list(layer.base.bias.shape)}, "mean_b": float(layer.b.mean()), "both_restored": True, "independent_clone_storage": True, "snapshots_require_grad": False},
    "omit_a_variant": {"sha256": sha(variant), "stdout": bad_stdout, "mean_b": float(bad_namespace["layer"].b.mean()), "both_restored": False},
    "alpha_variant": {"rank": 2, "alpha_before": 2, "alpha_after": 4, "delta_is_exactly_doubled": True, "scale_before": 1, "scale_after": 2},
    "state_and_merge": {"state_dict_keys": state_keys, "adapter_entry_keys": list(adapter_a["0"]), "same_rank_alpha_restored": container[0].alpha, "base_unchanged": True, "merged_has_no_lora_branch": True, "bias_unchanged": True, "double_add_formula_tolerance": 1e-7, "plain_matrix_expression_shape": list(merged_expression.shape), "plain_matrix_expression_max_difference": expression_difference, "same_mean_does_not_imply_equal": True, "scope": "_restore_adapter restores a/b/alpha into pre-existing same-rank modules, not rank, optimizer or RNG. This is a hand-set linear container; no TinyLM forward."},
    "payload_load": {"payload_keys": list(payload), "adapter_keys": list(payload["adapter"]["output"]), "loader_info": {k:v for k,v in info.items() if k != "path"}, "matrices_restored": True, "base_unchanged": True, "caller_rng_preserved": True, "rejections": rejections, "temporary_pt_removed": True, "scope": "Loader accepts lora-v1 metadata and state; initialized untrained native TinyLM only, never inference. Extra optimizer/step/RNG fields are not applied by this inference loader."},
    "recorded_measurements": {"input_path": str(result_path.relative_to(ROOT)), "input_sha256": sha(raw_result), "revision": recorded["revision"], "device": recorded["device"], "gpu": recorded["gpu"], "torch": recorded["torch_version"], "python": recorded["python_version"], "seed": recorded["seed"], "step_scale": recorded["step_scale"], "base_sha256": measurements["base_sha256"], "rank": measurements["rank"], "alpha": measurements["alpha"], "scaling": measurements["scaling"], "probe_ids": probe, "probe_tokens": tokens, "decoded_question": ByteTokenizer().decode(probe), "logit_shape_from_code": [1, 9, 264], "compared_logit_entries_per_pair": 2376, "statistic": "max absolute difference, no averaged denominator and no correctness score", "a_b_a_max_difference": measurements["a_b_a_max_difference"], "adapter_a_b_logit_difference": measurements["adapter_a_b_logit_difference"], "merge_max_difference": measurements["merge_max_difference"], "switch_allowed": 1e-6, "merge_allowed": 1e-4, "training_counts": {s:{"steps":measurements["runs"][s]["training"]["steps"], "effective_tokens":measurements["runs"][s]["training"]["effective_tokens"], "train_records":measurements["runs"][s]["data"]["train"]["records"], "layers":len(measurements["runs"][s]["layers"])} for s in ["concise", "vivid"]}, "code_hash_receipts": original_hash_receipts, "rerun": False, "scope": "One fixed dialogue prefix and all 9x264 logits; does not show all-input equivalence, style quality, full resumed training or byte-identical trained checkpoints. Trained .pt hashes are read as recorded metadata only, with no file fetch/recomputation."},
}
(HERE / "variants-result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
