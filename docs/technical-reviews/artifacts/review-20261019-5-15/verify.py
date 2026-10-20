"""核對種子、初始化、抽樣與透明數值算例。"""

import hashlib
import inspect
import json
import platform
import sys
from decimal import Decimal
from pathlib import Path

import torch
from torch.utils.data import RandomSampler

from tiny_perceptron.model import ModelConfig, TinyLM, generate


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


torch.set_default_device("cpu")
torch.set_num_threads(1)
a_scores = list(map(Decimal, ["0.70", "0.72", "0.69"]))
b_scores = list(map(Decimal, ["0.71", "0.71", "0.73"]))
result = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
    },
    "arithmetic": {
        "differences": [str(b - a) for a, b in zip(a_scores, b_scores)],
        "maximum_a": str(max(a_scores)),
        "maximum_b": str(max(b_scores)),
        "mean_a": str(sum(a_scores) / 3),
        "mean_b": str(sum(b_scores) / 3),
        "range_a": [str(min(a_scores)), str(max(a_scores))],
        "range_b": [str(min(b_scores)), str(max(b_scores))],
    },
    "initialization": [],
}
for seed in [1, 2, 3]:
    torch.manual_seed(seed)
    before = torch.get_rng_state().clone()
    model = TinyLM(ModelConfig(width=8))
    weight = model.embedding.weight
    explicit_std = ((weight.double() - weight.double().mean()).square().sum() / (weight.numel() - 1)).sqrt()
    ids = torch.tensor([[0, 1, 263]])
    result["initialization"].append(
        {
            "seed": seed,
            "shape": list(weight.shape),
            "numel": weight.numel(),
            "requires_grad": weight.requires_grad,
            "std": weight.std().item(),
            "explicit_sample_std": explicit_std.item(),
            "initialization_advances_rng": not torch.equal(before, torch.get_rng_state()),
            "lookup_matches_rows": torch.equal(model.embedding(ids), weight[ids]),
        }
    )

seed = 1
torch.manual_seed(seed)
a = TinyLM(ModelConfig(width=8))
after_a = torch.get_rng_state().clone()
b_without_reset = TinyLM(ModelConfig(width=8))
torch.manual_seed(seed)
b_with_reset = TinyLM(ModelConfig(width=8))
result["reconstruction"] = {
    "seed_variable": seed,
    "equal_with_reset": torch.equal(a.embedding.weight, b_with_reset.embedding.weight),
    "equal_without_reset": torch.equal(a.embedding.weight, b_without_reset.embedding.weight),
    "reset_and_rebuild_repeats_state": torch.equal(after_a, torch.get_rng_state()),
}

orders = {}
for width in [8, 16]:
    torch.manual_seed(1)
    model = TinyLM(ModelConfig(width=width))
    orders[str(width)] = list(RandomSampler(range(20)))
independent_orders = {}
for width in [8, 16]:
    data_rng = torch.Generator().manual_seed(7)
    torch.manual_seed(1)
    model = TinyLM(ModelConfig(width=width))
    independent_orders[str(width)] = list(RandomSampler(range(20), generator=data_rng))
result["sampling"] = {
    "shared_global_rng_orders": orders,
    "shared_orders_equal": orders["8"] == orders["16"],
    "independent_data_rng_orders": independent_orders,
    "independent_orders_equal": independent_orders["8"] == independent_orders["16"],
    "sample_count": 20,
    "model_seed": 1,
    "data_seed": 7,
}

torch.manual_seed(1)
model = TinyLM(ModelConfig(width=8))
ids = torch.tensor([[1, 3]])
generation = {}
for temperature in [0.0, 1.0]:
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    sequence = generate(model, ids, max_new_tokens=2, temperature=temperature, eos_id=-1)
    generation[str(temperature)] = {
        "sequence": sequence.tolist(),
        "advances_rng": not torch.equal(before, torch.get_rng_state()),
    }
result["generation"] = generation

result["repository_sources"] = {}
for name in ["tiny_perceptron/model.py", "docs/review-tools/section_facts.py"]:
    raw = Path(name).read_bytes()
    result["repository_sources"][name] = {"sha256": digest(raw)}
result["torch_implementation"] = {
    "manual_seed": inspect.getsource(torch.manual_seed),
    "embedding_reset_parameters": inspect.getsource(torch.nn.Embedding.reset_parameters),
    "random_sampler_iter": inspect.getsource(RandomSampler.__iter__),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
