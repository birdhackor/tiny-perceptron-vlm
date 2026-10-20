"""獨立核對 LayerNorm 的算例、軸、仿射參數與模型呼叫順序。"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_default_device("cpu")
torch.set_num_threads(1)
x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
layer = nn.LayerNorm(3)


def values(tensor):
    return tensor.detach().tolist()


with torch.no_grad():
    mean = x.mean(-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(-1, keepdim=True)
    manual = (x - mean) / torch.sqrt(variance + layer.eps)
    y = layer(x)
    row_derivations = []
    for row in x[0].tolist():
        m = sum(row) / len(row)
        centered = [v - m for v in row]
        v = sum(d * d for d in centered) / len(row)
        row_derivations.append({
            "input": row,
            "mean": m,
            "centered": centered,
            "variance": v,
            "standard_deviation": math.sqrt(v),
            "normalized_without_epsilon": [d / math.sqrt(v) for d in centered],
            "normalized_with_epsilon": [d / math.sqrt(v + layer.eps) for d in centered],
            "variance_with_epsilon": v / (v + layer.eps),
        })

    translated = x.clone()
    translated[0, 1] += 100
    moved = layer(translated)
    one_changed = x.clone()
    one_changed[0, 1, 0] += 100
    single = layer(one_changed)
    other = torch.tensor([[[4.0, 8.0, 9.0], [3.0, -1.0, 7.0]]])
    batch = torch.cat((x, other), dim=0)
    batch_changed = batch.clone()
    batch_changed[1] = torch.tensor([[900.0, -800.0, 70.0], [-5.0, 20.0, 100.0]])
    constant = torch.full_like(x, 7.0)
    constant_output = layer(constant)
    biased = y.var(-1, unbiased=False)
    correction_zero = y.var(-1, correction=0)

    defaults = {
        "weight": values(layer.weight),
        "bias": values(layer.bias),
        "weight_shape": list(layer.weight.shape),
        "bias_shape": list(layer.bias.shape),
        "weight_requires_grad": layer.weight.requires_grad,
        "bias_requires_grad": layer.bias.requires_grad,
        "epsilon": layer.eps,
    }
    affine = nn.LayerNorm(3)
    affine.weight.fill_(2.0)
    affine.bias.fill_(1.0)
    affine_y = affine(x)
    per_feature = nn.LayerNorm(3)
    per_feature.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    per_feature.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))
    per_feature_y = per_feature(x)

    wrong = nn.LayerNorm((2, 3), elementwise_affine=False)
    wrong_before = wrong(x)
    wrong_after = wrong(translated)
    causal_weights = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    causal_before = causal_weights @ wrong_before
    causal_after = causal_weights @ wrong_after

    torch.manual_seed(0)
    model = TinyLM(ModelConfig(width=3, heads=1, layers=1, max_length=2, vocab_size=4))
    model.eval()
    block = model.blocks[0]
    trace = []
    captures = {}
    hooks = []

    def output_hook(name):
        def capture(module, inputs, output):
            trace.append(name)
            captures[name] = output.detach().clone()
        return capture

    def input_hook(name):
        def capture(module, inputs):
            trace.append(name)
            captures[name] = inputs[0].detach().clone()
        return capture

    for name, module in (("norm1", block.norm1), ("norm2", block.norm2), ("final_norm", model.final_norm)):
        hooks.append(module.register_forward_hook(output_hook(name)))
    for name, module in (("attention", block.attention), ("ffn", block.ffn), ("output", model.output)):
        hooks.append(module.register_forward_pre_hook(input_hook(name)))
    result = model(torch.tensor([[1, 2]]))
    for hook in hooks:
        hook.remove()

    checks = {
        "tenfold_input": torch.equal(x[0, 1], 10 * x[0, 0]),
        "manual_formula": torch.allclose(y, manual, atol=1e-6, rtol=1e-6),
        "mean_assertion": torch.allclose(y.mean(-1), torch.zeros(1, 2), atol=1e-6),
        "population_variance_alias": torch.equal(biased, correction_zero),
        "translated_first_position": torch.allclose(y[0, 0], moved[0, 0]),
        "translated_second_position": torch.allclose(y[0, 1], moved[0, 1]),
        "single_change_first_position": torch.allclose(y[0, 0], single[0, 0]),
        "single_change_second_position_differs": not torch.allclose(y[0, 1], single[0, 1]),
        "batch_independence": torch.equal(layer(batch)[0], layer(batch_changed)[0]),
        "constant_finite_zero": bool(torch.isfinite(constant_output).all() and torch.equal(constant_output, torch.zeros_like(x))),
        "affine_formula": torch.allclose(affine_y, 2 * y + 1, atol=1e-6, rtol=1e-6),
        "affine_mean": torch.allclose(affine_y.mean(-1), torch.ones(1, 2), atol=1e-6),
        "affine_variance": torch.allclose(affine_y.var(-1, unbiased=False), 4 * biased, atol=1e-6, rtol=1e-6),
        "per_feature_shared_affine": torch.allclose(per_feature_y, y * per_feature.weight + per_feature.bias, atol=1e-6, rtol=1e-6),
        "wrong_axes_change_prefix": not torch.allclose(wrong_before[0, 0], wrong_after[0, 0]),
        "causal_attention_retains_leak": not torch.allclose(causal_before[0, 0], causal_after[0, 0]),
        "forward_order": trace == ["norm1", "attention", "norm2", "ffn", "final_norm", "output"],
        "normalized_attention_input": torch.equal(captures["norm1"], captures["attention"]),
        "normalized_ffn_input": torch.equal(captures["norm2"], captures["ffn"]),
        "normalized_output_input": torch.equal(captures["final_norm"], captures["output"]),
        "absolute_tolerance_accepts": torch.allclose(torch.tensor([0.9e-6]), torch.zeros(1), atol=1e-6),
        "absolute_tolerance_rejects": not torch.allclose(torch.tensor([1.1e-6]), torch.zeros(1), atol=1e-6),
        "shape_and_indexing": list(x.shape) == [1, 2, 3] and list(x[0, 1].shape) == [3] and x[0, 1, 0].item() == 10,
    }

    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-3/verify_layernorm.py",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": str(x.dtype),
            "threads": str(torch.get_num_threads()),
        },
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "repository_code_sha256": hashlib.sha256((ROOT / "tiny_perceptron/model.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_derivations": row_derivations,
        "input_shape": list(x.shape),
        "row_selection": values(x[0, 1]),
        "scalar_selection": x[0, 1, 0].item(),
        "defaults": defaults,
        "output": values(y),
        "output_mean": values(y.mean(-1)),
        "output_variance": values(biased),
        "manual_formula_max_error": (y - manual).abs().max().item(),
        "translated_output": values(moved),
        "single_changed_output": values(single),
        "constant_output": values(constant_output),
        "affine_output": values(affine_y),
        "affine_mean": values(affine_y.mean(-1)),
        "affine_variance": values(affine_y.var(-1, unbiased=False)),
        "wrong_axes_prefix_before": values(wrong_before[0, 0]),
        "wrong_axes_prefix_after": values(wrong_after[0, 0]),
        "forward_trace": trace,
        "model_output_shape": list(result["logits"].shape),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "scope": "固定輸入、CPU float32 與目前版本的可重做核對；不是訓練品質或跨硬體效能測試。",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    if not output["all_checks_passed"]:
        raise SystemExit("LayerNorm 核對失敗，請檢查輸出中的布林結果。")
