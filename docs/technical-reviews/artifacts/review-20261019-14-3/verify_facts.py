"""核對向量、softmax、練習與本機模型的支援範圍。"""

import ast
import hashlib
import inspect
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import CausalAttention
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def lists(tensor):
    return tensor.detach().tolist()


torch.set_default_device("cpu")
torch.manual_seed(0)
q = torch.tensor([[1.0, 1.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 100.0]])
raw = q @ k.T
qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
unit = qn @ kn.T
raw_probs = raw.softmax(-1)
unit_probs = unit.softmax(-1)
angles = [math.degrees(math.acos(float(v))) for v in (qn.double() @ kn.double().T)[0]]
assert all(abs(v - 45.0) < 2e-6 for v in angles)
assert torch.equal(raw, torch.tensor([[1.0, 100.0]]))
assert torch.allclose(qn, torch.tensor([[1 / math.sqrt(2), 1 / math.sqrt(2)]]), atol=1e-7, rtol=0)
assert torch.equal(kn, torch.eye(2))
assert torch.equal(unit_probs, torch.tensor([[0.5, 0.5]]))
assert torch.equal(raw_probs.round(decimals=4), torch.tensor([[0.0, 1.0]]))
small_probability = math.exp(-99) / (1 + math.exp(-99))
assert abs(raw_probs[0, 0].item() - small_probability) < 1.5e-45
scores = torch.tensor([1.0, 0.0])
plain_probs, scaled_probs = scores.softmax(-1), (scores * 4).softmax(-1)
assert torch.allclose(plain_probs, torch.tensor([0.7311, 0.2689]), atol=5e-5, rtol=0)
assert torch.allclose(scaled_probs, torch.tensor([0.9820, 0.0180]), atol=5e-5, rtol=0)
exercise_k = torch.tensor([[1.0, 0.0], [0.0, 0.1]])
exercise_raw = q @ exercise_k.T
exercise_unit = qn @ F.normalize(exercise_k, dim=-1).T
assert torch.equal(exercise_raw, torch.tensor([[1.0, 0.1]]))
assert torch.equal(exercise_unit.softmax(-1), torch.tensor([[0.5, 0.5]]))
temperatures = {str(t): lists((scores / t).softmax(-1)) for t in (0.25, 1.0, 4.0)}
assert temperatures["0.25"][0] > temperatures["1.0"][0] > temperatures["4.0"][0]
config_fields = [f.name for f in fields(ModelConfig)]
assert not any("qk" in name for name in config_fields)
attention_parameters = str(inspect.signature(CausalAttention))
assert "qk" not in attention_parameters
attention_forward = inspect.getsource(CausalAttention.forward)
assert "normalize" not in attention_forward and "q_norm" not in attention_forward
model_checks = {}
with torch.no_grad():
    for norm in ("layer", "rms"):
        model = TinyLM(ModelConfig(norm=norm))
        result = model(torch.tensor([[1, 2, 3]]))
        assert result["logits"].shape == (1, 3, 264)
        assert torch.isfinite(result["logits"]).all()
        model_checks[norm] = {
            "logits_shape": list(result["logits"].shape),
            "attention_modules": [n for n, _ in model.blocks[0].attention.named_modules()],
        }
record = json.loads((ROOT / "docs/course-experiments/results/modern.json").read_bytes())
variants = list(record["results"]["variants"])
expected_variants = ["baseline", "rope", "rmsnorm", "relu2", "swiglu", "tied"]
assert variants == expected_variants
tree = ast.parse((ROOT / "scripts/course_experiments/architecture.py").read_text())
run_modern = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_modern")
assignment = next(n for n in run_modern.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "variants" for t in n.targets))
assert list(ast.literal_eval(assignment.value)) == expected_variants
paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
    "scripts/course_experiments/architecture.py", "docs/course-experiments/results/modern.json",
]
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-3/verify_facts.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(q.dtype)},
    "result": "全部斷言通過；未訓練模型。",
    "inputs": {"q": lists(q), "k": lists(k)},
    "geometry": {"q_length": math.sqrt(2), "k_lengths": [1, 100], "angles_degrees": angles, "q_normalized": lists(qn), "k_normalized": lists(kn)},
    "raw": {"shape": list(raw.shape), "scores": lists(raw), "probabilities": lists(raw_probs), "rounded": lists(raw_probs.round(decimals=4)), "first_probability_float64_formula": small_probability, "first_probability_float32_absolute_tolerance": 1.5e-45},
    "normalized": {"shape": list(unit.shape), "scores": lists(unit), "probabilities": lists(unit_probs)},
    "scale": {"base_scores": lists(scores), "unscaled": lists(plain_probs), "times_four": lists(scaled_probs), "temperature_probabilities": temperatures},
    "exercise": {"k": lists(exercise_k), "raw_scores": lists(exercise_raw), "raw_probabilities": lists(exercise_raw.softmax(-1)), "normalized_scores": lists(exercise_unit), "normalized_probabilities": lists(exercise_unit.softmax(-1))},
    "axis_check": {"q_shape": list(q.shape), "k_shape": list(k.shape), "k_transposed_shape": list(k.T.shape), "last_feature_norms": lists(kn.norm(dim=-1)), "candidate_axis_normalized_scores": lists(qn @ F.normalize(k, dim=0).T)},
    "support_boundaries": {"model_config_fields": config_fields, "attention_constructor": attention_parameters, "attention_forward": attention_forward, "model_checks": model_checks, "record_variants": variants, "run_modern_variants": list(ast.literal_eval(assignment.value)), "file_sha256": {p: digest(p) for p in paths}},
}, ensure_ascii=False, indent=2))
