"""Bounded CPU verification; no model training, full evaluation or file checkpoint."""
import ast
import hashlib
import io
import json
import platform
from pathlib import Path
import sys

import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import TinyLM, ModelConfig

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
results = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
    "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)}}

# Original exercise: change the second row's second coordinate, not the first.
embedding = nn.Embedding(3, 2)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
tied = nn.Linear(2, 3, bias=False)
tied.weight = embedding.weight
copied = nn.Linear(2, 3, bias=False)
with torch.no_grad():
    copied.weight.copy_(embedding.weight)
h = torch.tensor([[2., 1.]])
assert torch.equal(tied(h), h @ embedding.weight.T)
with torch.no_grad():
    embedding.weight[1, 1] += 1
assert tied(h).tolist() == [[2., 2., 3.]]
assert copied(h).tolist() == [[2., 1., 3.]]
results["second_row_exercise"] = {"tied": tied(h).tolist(), "copied": copied(h).tolist(),
    "input_shape": list(h.shape), "weight_shape": list(embedding.weight.shape),
    "output_shape": list(tied(h).shape), "logits_are_not_probabilities": tied(h).sum().item() != 1.}

# Both uses accumulate derivatives in one Parameter. No optimizer or training.
combined = nn.ModuleDict({"embedding": embedding, "output": tied})
assert list(combined.parameters()) == [embedding.weight]
loss = embedding(torch.tensor([0])).sum() + tied(h).sum()
loss.backward()
expected_grad = torch.tensor([[3., 2.], [2., 1.], [2., 1.]])
assert torch.equal(embedding.weight.grad, expected_grad)
results["shared_gradient"] = {"distinct_parameters": len(list(combined.parameters())),
    "unique_scalars": sum(p.numel() for p in combined.parameters()),
    "gradient": embedding.weight.grad.tolist(), "optimizer_steps": 0,
    "derivation": "input lookup contributes [1,1] only to row 0; summed logits contributes h=[2,1] to all rows"}

# Saving tensor storage does not select the architecture of the destination modules.
buffer = io.BytesIO()
torch.save(combined.state_dict(), buffer)
buffer.seek(0)
state = torch.load(buffer, weights_only=True)
assert state["embedding.weight"].untyped_storage().data_ptr() == state["output.weight"].untyped_storage().data_ptr()
separate = nn.ModuleDict({"embedding": nn.Embedding(3,2), "output": nn.Linear(2,3,bias=False)})
separate.load_state_dict(state)
assert separate["embedding"].weight is not separate["output"].weight
assert torch.equal(separate["embedding"].weight, separate["output"].weight)
restored = nn.ModuleDict({"embedding": nn.Embedding(3,2), "output": nn.Linear(2,3,bias=False)})
restored["output"].weight = restored["embedding"].weight
restored.load_state_dict(state)
assert restored["embedding"].weight is restored["output"].weight
results["serialization"] = {"serialized_storage_shared": True,
    "untied_destination_equal_values": True, "untied_destination_same_parameter": False,
    "tied_destination_same_parameter": True, "bytesio_size": len(buffer.getvalue()), "saved_weight_files": 0}

# Check exact recorded copy mechanism, loaded from the immutable measured source.
source = OUT / "sources/architecture-measured-revision.py"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3"
node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "_copy_matching")
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
torch.manual_seed(42)
base = TinyLM(ModelConfig(width=64, layers=2, heads=4))
torch.manual_seed(42)
target = TinyLM(ModelConfig(width=64, layers=2, heads=4, tied=True))
copied_names = namespace["_copy_matching"](base, target)
assert "output.weight" not in copied_names
assert target.output.weight is target.embedding.weight
assert torch.equal(target.embedding.weight, base.embedding.weight)
assert not torch.equal(target.output.weight, base.output.weight)
for name in copied_names:
    assert torch.equal(base.state_dict()[name], target.state_dict()[name])
base_count = sum(p.numel() for p in base.parameters())
target_count = sum(p.numel() for p in target.parameters())
assert (base_count, target_count, base_count-target_count) == (141568, 124672, 264*64)
probe_h = torch.ones(1,64)
assert not torch.equal(base.output(probe_h), target.output(probe_h))
results["measured_structure_and_initialization"] = {"baseline_parameters": base_count,
    "tied_parameters": target_count, "saved_parameters": base_count-target_count,
    "vocab_size": 264, "width": 64, "all_copied_initial_tables_equal": True,
    "tied_output_is_embedding": True, "independent_output_table_copied": False,
    "same_hidden_probe_output_distribution_differs": True,
    "executed_node": f"_copy_matching lines {node.lineno}-{node.end_lineno}; no _train or heldout execution"}

# Recalculate only selected original measurements, not model evaluation.
selected = json.loads((OUT / "raw/inspected-pointers.json").read_text())
metrics = {}
for variant in ["baseline", "tied"]:
    p = "/results/variants/" + variant
    for split in ["initial", "validation", "test"]:
        if split == "initial":
            item = selected[p+"/training/initial"]
        else:
            item = {k: selected[p+"/heldout/"+split+"/"+k] for k in ["nll","nll_sum","effective_tokens","examples","records"]}
        recalculated = item["nll_sum"] / item["effective_tokens"]
        assert abs(recalculated-item["nll"]) < 1e-12
        metrics[variant+"/"+split] = {**item,"recalculated":recalculated,"rounded5":round(recalculated,5)}
    assert selected[p+"/training/steps"] == selected[p+"/training/optimizer_updates"] == 240
    assert selected[p+"/training/skipped_updates"] == 0
assert metrics["tied/initial"]["rounded5"] == 43.87647
assert metrics["baseline/initial"]["rounded5"] == 5.76007
assert metrics["tied/validation"]["rounded5"] == 2.36276
assert metrics["tied/test"]["rounded5"] == 2.38422
assert metrics["baseline/validation"]["rounded5"] == 2.26331
assert metrics["baseline/test"]["rounded5"] == 2.29163
results["raw_measurement_recalculation"] = metrics
results["budgets"] = {"train_records":409,"validation_records":51,"test_records":52,
    "initial_effective_tokens":329581,"train_example_chunks":2789,"training_token_presentations":452102,
    "validation_effective_tokens":39256,"test_effective_tokens":41914,
    "validation_example_chunks":334,"test_example_chunks":352,"optimizer_updates":240,
    "batch_size":16,"seed":42,"new_training_or_evaluation":False}
results["status"] = "all bounded checks passed"
print(json.dumps(results, ensure_ascii=False, indent=2))
