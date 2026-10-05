"""Independent bounded CPU checks for 15.2; no training, evaluation, or weight files."""
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
PROOF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "torch_git_version": torch.version.git_version, "device": "cpu",
                  "cuda_build": str(torch.version.cuda), "threads": torch.get_num_threads()}))

# Execute the exact original fence in a fresh namespace, then check the requested variation.
fence = PROOF / "execution-original/fence-1.py"
namespace = {"__name__": "__main__"}
exec(compile(fence.read_bytes(), str(fence), "exec"), namespace)
experts, x, repeated = namespace["experts"], namespace["x"], namespace["repeated"]
assert len(experts) == 3 and len(list(experts.parameters())) == 3
assert len({id(e.weight) for e in experts}) == 3
assert len({e.weight.data_ptr() for e in experts}) == 3
assert [e(x).tolist() for e in experts] == [[[1.0, 2.0]], [[2.0, 4.0]], [[2.0, 1.0]]]
assert [tuple(e(x).shape) for e in experts] == [(1, 2)] * 3
assert [p.numel() for p in experts.parameters()] == [4, 4, 4]
assert sum(p.numel() for p in repeated.parameters()) == 4
assert len(list(repeated.named_parameters(remove_duplicate=False))) == 3
assert all(e.weight.requires_grad and e.weight.grad is None for e in experts)
identities = [id(e.weight) for e in experts]
with torch.no_grad():
    experts[0].weight.mul_(3)
changed = [e(x).tolist() for e in experts]
assert changed == [[[3.0, 6.0]], [[2.0, 4.0]], [[2.0, 1.0]]]
assert identities == [id(e.weight) for e in experts]
print("Independent modification:", changed)
try:
    experts(x)
except NotImplementedError:
    print("ModuleList alone has no forward: NotImplementedError as expected")
else:
    raise AssertionError("ModuleList unexpectedly performs a forward")

# Same values do not bind storage; asymmetric weights expose Linear's x @ W.T axis.
separate = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    separate.weight.copy_(experts[1].weight)
assert torch.equal(separate.weight, experts[1].weight)
assert separate.weight.data_ptr() != experts[1].weight.data_ptr()
with torch.no_grad():
    separate.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
assert separate(x).tolist() == [[5.0, 11.0]]
print("Axis variation x @ W.T:", separate(x).tolist())

# Check all four recorded structural conditions without loading weights or running a model.
raw = PROOF / "inputs/docs/course-experiments/results/moe.json"
report = json.loads(raw.read_bytes())
for variant in ("top1_aux0", "top1_aux0.01", "top2_aux0", "top2_aux0.01"):
    row = report["results"]["variants"][variant]
    model = TinyLM(ModelConfig(**row["model"]["config"]))
    expert_modules = [e for block in model.blocks for e in block.ffn.experts]
    tensors = [p for expert in expert_modules for p in expert.parameters()]
    expert_sizes = [sum(p.numel() for p in expert.parameters()) for expert in expert_modules]
    expert_parameters = sum(expert_sizes)
    router_parameters = sum(p.numel() for block in model.blocks for p in block.ffn.router.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert len(expert_modules) == 8 and len(set(map(id, expert_modules))) == 8
    assert len(tensors) == 32 and len({p.data_ptr() for p in tensors}) == 32
    assert expert_sizes == [33088] * 8
    assert expert_parameters == row["budget"]["expert_parameters"] == 264704
    assert router_parameters == row["budget"]["router_parameters"] == 512
    assert total == row["model"]["parameters"] == row["budget"]["total_parameters"] == 340608
    assert 8 * (64 * 256 + 256 + 256 * 64 + 64) == expert_parameters
    assert total - expert_parameters - router_parameters == 75392
    print(variant, json.dumps({"layers": len(model.blocks), "experts_per_layer": 4,
          "independent_experts": len(expert_modules), "parameters_per_expert": expert_sizes,
          "expert_parameters": expert_parameters, "router_parameters": router_parameters,
          "shared_nonrouter_parameters": total - expert_parameters - router_parameters,
          "total_parameters": total}))

# Preserve only schema/type inventories for measurements outside the structural claim.
variant = report["results"]["variants"]["top2_aux0.01"]
print("heldout_schema:", {k: type(v).__name__ for k, v in variant["heldout"].items()})
for split in ("validation", "test"):
    print("heldout_" + split + "_schema:", {k: type(v).__name__ for k, v in variant["heldout"][split].items()})
print("validation_routing_schema:", {k: type(v).__name__ for k, v in variant["validation_routing"].items()})
print("routing_layer_schemas:", [{k: type(v).__name__ for k, v in layer.items()}
      for layer in variant["validation_routing"]["layers"]])

for name, selected in (
    ("scripts/course_experiments/architecture.py", ("_text_dataset", "_parameter_budget", "_heldout", "_routing")),
    ("scripts/course_experiments/common.py", ("new_lm", "text_examples", "evaluate_lm")),
):
    current = (ROOT / name).read_text()
    original = (PROOF / "inputs/recorded-revision" / name).read_text()
    def nodes(text):
        return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name in selected}
    a, b = nodes(current), nodes(original)
    for key in selected:
        assert a[key] == b[key], (name, key, "method changed from original")
        print("Original method byte-equivalent:", name, key,
              hashlib.sha256(b[key].encode()).hexdigest())
print("All bounded 15.2 assertions passed; no optimizer, training, generation, or model data download executed.")
