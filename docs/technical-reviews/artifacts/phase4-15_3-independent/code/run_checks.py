"""Bounded CPU review: original fence, variants, and existing raw aggregates.

No training, optimizer steps, GPU use, dataset/model downloads, or full model evaluation.
"""
import ast
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_revision": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "platform": platform.platform(),
}
(BASE / "execution/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print("ENVIRONMENT", json.dumps(environment))
namespace = {"__name__": "__main__"}
fence = BASE / "code/fence-1.py"
print("ORIGINAL_FENCE_SHA256", hashlib.sha256(fence.read_bytes()).hexdigest())
exec(compile(fence.read_bytes(), str(fence), "exec"), namespace)
shapes = {n: list(namespace[n].shape) for n in ["x", "router_weight", "scores", "prob", "expert_outputs", "output"]}
assert shapes == {"x": [1, 2], "router_weight": [2, 3], "scores": [1, 3], "prob": [1, 3], "expert_outputs": [1, 3, 2], "output": [1, 2]}
assert namespace["prob"][:, :, None].shape == (1, 3, 1)
assert torch.allclose(namespace["prob"], torch.tensor([[.25, .25, .5]]), atol=1e-7, rtol=0)
assert torch.allclose(namespace["output"], torch.tensor([[2.25, 4.5]]), atol=1e-6, rtol=0)
assert all(not namespace[n].requires_grad for n in shapes)
print("AXES", json.dumps(shapes))
print("MATH", json.dumps({"e": math.e, "log2": math.log(2), "exp0": math.exp(0), "exp_log2": math.exp(math.log(2)), "weighted_multiplier": .25 + .25 * 2 + .5 * 3}))

def mix(x, weight):
    score = x @ weight
    probability = score.softmax(-1)
    experts = torch.stack([x, 2*x, 3*x], dim=1)
    return score, probability, (probability[:, :, None] * experts).sum(1)

_, equal, output = mix(namespace["x"], torch.zeros(2, 3))
assert torch.allclose(equal, torch.full((1, 3), 1/3), atol=1e-7, rtol=0)
assert torch.allclose(output, torch.tensor([[2., 4.]]), atol=1e-6, rtol=0)
print("EXERCISE", json.dumps({"probabilities": equal.tolist(), "output": output.tolist()}))
score, probability, output = mix(torch.tensor([[2., 2.], [1., 9.]]), namespace["router_weight"])
assert torch.allclose(score, torch.tensor([[0., 0., math.log(4)], [0., 0., math.log(2)]]), atol=1e-7, rtol=0)
assert torch.allclose(probability, torch.tensor([[1/6, 1/6, 2/3], [.25, .25, .5]]), atol=1e-7, rtol=0)
assert torch.allclose(output, torch.tensor([[5., 5.], [2.25, 20.25]]), atol=3e-6, rtol=0)
print("INPUT_VARIANT", json.dumps({"scores": score.tolist(), "probabilities": probability.tolist(), "output": output.tolist()}))

# Demonstrate an available learning signal; deliberately perform no update.
weight = namespace["router_weight"].clone().requires_grad_()
_, _, output = mix(namespace["x"], weight)
output.sum().backward()
assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
print("GRADIENT_ONLY_NO_UPDATE", weight.grad.tolist())

# Expert output scale changes the task loss and hence the gate gradient.
# Neither coefficient changes when the correctness target alone is replaced.
scale_gradients = []
for scale in (1., 10.):
    logits = torch.tensor([0., 0., math.log(2)], requires_grad=True)
    p = logits.softmax(-1)
    y = (p * torch.tensor([1., 2., 3. * scale])).sum()
    (y-3).square().backward()
    scale_gradients.append({"scale": scale, "probabilities": p.tolist(), "output": y.item(), "gradient": logits.grad.tolist()})
assert scale_gradients[0]["probabilities"] == scale_gradients[1]["probabilities"]
assert scale_gradients[0]["gradient"] != scale_gradients[1]["gradient"]
print("MIXTURE_COEFFICIENT_SCOPE", json.dumps(scale_gradients))

original = json.loads((BASE / "inputs/moe.json").read_text())
variant = original["results"]["variants"]["top2_aux0.01"]
routing = variant["validation_routing"]
layer = routing["layers"][1]
config = variant["model"]["config"]
auxiliary = variant["training"]["auxiliary_weight"]
assert (config["experts"], config["top_k"], config["layers"], auxiliary) == (4, 2, 2, .01)
tokens = routing["effective_input_tokens"]
denominator = sum(layer["dispatch_counts"])
assert denominator == layer["dispatch_denominator"] == tokens * config["top_k"] == 78512
fractions = [n / denominator for n in layer["dispatch_counts"]]
assert all(abs(a-b) < 1e-15 for a,b in zip(fractions, layer["load_fraction"], strict=True))
assert [round(p, 4) for p in layer["mean_router_probability"]] == [.2875, .1772, .2652, .2701]
assert [round(p, 4) for p in fractions] == [.2065, .2692, .2260, .2983]
assert abs(sum(fractions)-1) < 1e-15
assert abs(sum(layer["mean_router_probability"])-1) < 1e-6
historical = (BASE / "inputs/historical-architecture.py").read_bytes()
assert hashlib.sha256(historical).hexdigest() == original["code_sha256"]["scripts/course_experiments/architecture.py"]
modern = (BASE / "inputs/historical-modern.py").read_bytes()
assert hashlib.sha256(modern).hexdigest() == original["code_sha256"]["tiny_perceptron/modern.py"]
trees = [ast.parse((BASE / "inputs" / name).read_bytes()) for name in ["historical-architecture.py", "architecture.py"]]
functions = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_routing") for t in trees]
assert ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False)
measurements = {
    "pointers": [
        "/revision", "/device", "/seed", "/torch_version", "/python_version", "/step_scale", "/code_sha256",
        "/results/variants/top2_aux0.01/model/config",
        "/results/variants/top2_aux0.01/training/auxiliary_weight",
        "/results/variants/top2_aux0.01/validation_routing/effective_input_tokens",
        "/results/variants/top2_aux0.01/validation_routing/padding_excluded",
        "/results/variants/top2_aux0.01/validation_routing/layers/1",
    ],
    "historical_revision": original["revision"],
    "device_of_original_experiment": original["device"],
    "effective_input_tokens": tokens,
    "padding_excluded": routing["padding_excluded"],
    "experts": config["experts"], "top_k": config["top_k"],
    "layer_index_zero_based": 1, "auxiliary_weight": auxiliary,
    "dispatch_counts": layer["dispatch_counts"], "dispatch_denominator": denominator,
    "recomputed_load_fractions": fractions,
    "stored_mean_router_probabilities": layer["mean_router_probability"],
    "probability_sum": sum(layer["mean_router_probability"]),
    "routing_method_unchanged_ast": True,
    "scope": "Existing aggregate evidence inspected and recalculated; no original per-token probability trace, no retraining or reevaluation.",
}
(BASE / "execution/measurement-check.json").write_text(json.dumps(measurements, indent=2) + "\n")
print("EXISTING_AGGREGATE_CHECK", json.dumps(measurements))
print("ALL_BOUNDED_CHECKS_PASSED")
