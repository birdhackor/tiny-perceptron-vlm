"""核對注意力算例、練習、函式庫行為與既有實驗紀錄。"""

import ast
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import CausalAttention, attention_mask, manual_attention
from scripts.course_experiments.architecture import _flash_manual_attention, _flash_sdpa_attention


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-8/verify_local.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "fresh_gpu_execution": False,
    "probes": {},
}

for seed_value in (0, 1):
    torch.manual_seed(seed_value)
    q, k, v = [torch.randn(1, 1, 4, 3) for _ in range(3)]
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    scores = q @ k.transpose(-2, -1) / (3 ** 0.5)
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    manual = weights @ v
    optimized = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    inverted = F.scaled_dot_product_attention(q, k, v, attn_mask=~allowed, dropout_p=0.0)
    double_scaled = F.scaled_dot_product_attention(q / (3 ** 0.5), k, v, attn_mask=allowed, dropout_p=0.0)
    original_assertion_failed = False
    try:
        assert torch.allclose(manual, inverted, atol=1e-6)
    except AssertionError:
        original_assertion_failed = True
    library, library_weights = manual_attention(q, k, v, allowed)
    causal = attention_mask(torch.arange(4), torch.arange(4))
    assert torch.equal(allowed, causal)
    assert torch.equal(library, manual)
    assert torch.allclose(manual, optimized, atol=1e-6)
    assert original_assertion_failed
    assert bool((weights.masked_select(~allowed) == 0).all())
    assert not torch.allclose(manual, double_scaled, atol=1e-6)
    result["probes"][f"seed_{seed_value}"] = {
        "qkv_shape": list(q.shape), "scores_shape": list(scores.shape),
        "weights_shape": list(weights.shape), "output_shape": list(optimized.shape),
        "allowed": allowed[0, 0].tolist(), "row_weight_sums": weights.sum(-1).tolist(),
        "forbidden_weights_zero": bool((weights.masked_select(~allowed) == 0).all()),
        "max_absolute_error": float((manual - optimized).abs().max()),
        "allclose_atol": 1e-6, "allclose_rtol": 1e-5, "allclose": True,
        "inverted_mask_max_error": float((manual - inverted).abs().max()),
        "inverted_mask_assertion_failed": original_assertion_failed,
        "double_scaling_max_error": float((manual - double_scaled).abs().max()),
        "library_matches_example": torch.equal(library, manual),
        "library_mask_matches_example": torch.equal(allowed, causal),
        "key_transpose_shape": list(k.transpose(-2, -1).shape),
    }


class AttentionDropout(nn.Module):
    def forward(self, q, k, v, probability):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=probability)


module = AttentionDropout().eval()
torch.manual_seed(7)
dropout_output = module(q, k, v, 0.5)
without_dropout = module(q, k, v, 0.0)
assert not torch.equal(dropout_output, without_dropout)
result["probes"]["eval_dropout"] = {
    "training": module.training,
    "explicit_dropout_probability": 0.5,
    "max_error_from_zero_dropout": float((dropout_output - without_dropout).abs().max()),
}

torch.manual_seed(17)
manual_layer = CausalAttention(12, heads=2)
sdpa_layer = copy.deepcopy(manual_layer)
sdpa_layer.backend = "sdpa"
x = torch.randn(2, 4, 12)
valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
a = manual_layer(x, valid=valid)[0]
b = sdpa_layer(x, valid=valid)[0]
a.square().sum().backward()
b.square().sum().backward()
gradient_error = max(float((p.grad - r.grad).abs().max()) for p, r in zip(
    manual_layer.parameters(), sdpa_layer.parameters(), strict=True))
assert torch.allclose(a, b, atol=1e-6)
assert gradient_error < 1e-5
result["probes"]["library_layer"] = {
    "input_shape": list(x.shape), "valid": valid.tolist(), "heads": 2,
    "output_max_error": float((a - b).detach().abs().max()), "parameter_gradient_max_error": gradient_error,
    "includes_empty_padding_query": True,
}

result["probes"]["low_precision_reference"] = {}
for dtype in (torch.float16, torch.bfloat16):
    generator = torch.Generator().manual_seed(42)
    inputs = tuple(torch.randn(2, 4, 8, 4, generator=generator).to(dtype).requires_grad_() for _ in range(3))
    upstream = (torch.randn(2, 4, 8, 4, generator=generator) * 0.125).to(dtype)
    low_scores = inputs[0] @ inputs[1].transpose(-2, -1)
    scores = low_scores.float() * (inputs[0].shape[-1] ** -0.5)
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    probability = scores.masked_fill(~causal, float("-inf")).softmax(-1).to(dtype)
    explicit = probability @ inputs[2]
    actual = _flash_manual_attention(*inputs)
    assert torch.equal(actual, explicit)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    scalar_gradients = torch.autograd.grad((_flash_manual_attention(*inputs) * upstream).sum(), inputs)
    assert all(torch.equal(a, b) for a, b in zip(gradients, scalar_gradients, strict=True))
    cpu_sdpa = _flash_sdpa_attention(*inputs)
    result["probes"]["low_precision_reference"][str(dtype)] = {
        "qk_matmul_dtype": str(low_scores.dtype), "scores_dtype": str(scores.dtype),
        "probabilities_before_pv_dtype": str(probability.dtype), "output_dtype": str(actual.dtype),
        "explicit_reference_identical": True, "vector_jacobian_matches_scalar_loss": True,
        "cpu_sdpa_output_shape": list(cpu_sdpa.shape),
        "scope": "CPU 小尺寸只核對對照路徑與上游梯度的算法；沒有使用 CUDA Flash。",
    }

eff_path = ROOT / "docs/course-experiments/results/efficiency.json"
eff = json.loads(eff_path.read_text())
e = eff["results"]
probe = e["manual_vs_sdpa"]
updates = {}
for name in ("ordinary", "sdpa"):
    t = e["update_variants"][name]["training"]
    updates[name] = {key: t[key] for key in (
        "requested_steps", "steps", "optimizer_updates", "skipped_updates", "effective_tokens", "batch_size",
        "learning_rate", "warm_step_median_seconds", "step_scale")}
    assert t["steps"] == t["optimizer_updates"] == 40 and t["effective_tokens"] == 2328
    assert t["skipped_updates"] == 0
    updates[name]["display_ms"] = f'{t["warm_step_median_seconds"] * 1000:.3f}'
for name, expected in (("manual", "2.990"), ("sdpa", "2.346")):
    timing = probe["timings"][name]
    assert statistics.median(timing["samples_seconds"]) == timing["median_seconds"]
    assert f'{1000 * timing["median_seconds"]:.3f}' == expected
    assert timing["warmup_calls"] == 3 and timing["measured_calls"] == 9
assert probe["output_max_error"] < 1e-5
assert probe["gradient_max_error"] < 1e-8
assert f'{probe["output_max_error"]:.5e}' == "3.81470e-06"
assert f'{probe["gradient_max_error"]:.5e}' == "1.60071e-10"
weight_error = e["update_variants"]["sdpa"]["weight_max_error_from_ordinary_after_updates"]
assert f'{weight_error:.5e}' == "3.51369e-05"
assert probe["shape"] == [4, 53] and probe["dtype"] == "torch.float32"
result["recorded_efficiency"] = {
    "path": str(eff_path.relative_to(ROOT)), "sha256": digest(eff_path),
    "device": eff["device"], "gpu": eff["gpu"], "torch": eff["torch_version"], "seed": eff["seed"],
    "model": e["models"]["mha"]["model"], "manual_vs_sdpa": probe,
    "update_variants": updates, "updated_weight_max_error": weight_error,
    "training_latency_denominator": "各 40 次更新，丟棄最初 3 次後取 37 次中位數；batch_size=8",
    "verification_scope": "本回合重算既有紀錄的中位數、換算、誤差與容差；沒有重跑 NVIDIA L4。",
}

code_path = "scripts/course_experiments/architecture.py"
expected_code_hash = eff["code_sha256"][code_path]
history = subprocess.run(["git", "log", "--all", "--format=%H", "--", code_path],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout.splitlines()
for revision in history:
    old = subprocess.run(["git", "show", revision + ":" + code_path], capture_output=True, cwd=ROOT)
    if old.returncode == 0 and hashlib.sha256(old.stdout).hexdigest() == expected_code_hash:
        current_ast = ast.parse((ROOT / code_path).read_text())
        old_ast = ast.parse(old.stdout)
        names = ("_sdpa_probe", "_gradients", "_gradient_error", "_benchmark", "_train", "run_efficiency")
        comparisons = {name: ast.dump(next(node for node in current_ast.body if getattr(node, "name", None) == name))
                       == ast.dump(next(node for node in old_ast.body if getattr(node, "name", None) == name))
                       for name in names}
        assert all(comparisons.values())
        result["recorded_efficiency"]["implementation_provenance"] = {
            "commit_with_exact_recorded_code": revision, "path": code_path, "sha256": expected_code_hash,
            "relevant_functions_AST_identical_to_current": comparisons,
            "method": "git log --all --format=%H -- path；逐一 git show commit:path，核對 SHA-256 與函式 AST。",
        }
        break
else:
    raise ValueError("找不到符合效率紀錄 SHA-256 的實驗程式")

flash_path = ROOT / "docs/course-experiments/results/flash_probe.json"
flash = json.loads(flash_path.read_text())
f = flash["results"]
routes = {}
for dtype, expected_output, expected_gradient in (("fp16", 0.001953125, 0.000244140625),
                                                ("bf16", 0.015625, 0.001953125)):
    route = f["routes"][dtype]
    c = route["correctness"]
    maximum = max(g["max_absolute_error"] for g in c["gradients"].values())
    assert c["output"]["max_absolute_error"] == expected_output
    assert maximum == expected_gradient
    for item in (c["output"], *c["gradients"].values()):
        assert item["finite"] and item["within_declared_tolerance"]
        assert item["max_absolute_error"] < item["atol"]
    for mode in ("forward", "forward_backward"):
        profile = route["profiles"][mode]
        assert "aten::_scaled_dot_product_flash_attention" in profile["operator_names"]
        assert any("flash_fwd" in name for name in profile["cuda_kernel_names"])
        if mode == "forward_backward":
            assert "aten::_scaled_dot_product_flash_attention_backward" in profile["operator_names"]
            assert any("flash_bwd" in name for name in profile["cuda_kernel_names"])
    routes[dtype] = {
        "status": route["status"], "correctness": c, "profiles": route["profiles"],
        "maximum_qkv_gradient_error": maximum,
        "every_maximum_below_absolute_tolerance": True,
    }
result["recorded_flash"] = {
    "path": str(flash_path.relative_to(ROOT)), "sha256": digest(flash_path),
    "gpu": flash["gpu"], "torch": flash["torch_version"], "seed": flash["seed"],
    "configuration": f["configuration"], "manual_reference": f["manual_reference"], "routes": routes,
    "verification_scope": "讀取既有 profiler 運算子與 CUDA kernel 名稱；重算三份梯度的最大值並檢查已宣告容差。未重跑 GPU。",
}

run = ROOT / "outputs/reviewer-tools/runs/20261020T064522515882Z"
section = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/16.md#16.8" --execute',
    "copied_files": {name: {"sha256": digest(run / name), "content": (run / name).read_text()}
                     for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py")},
}
out = Path(__file__).resolve().parent
(out / "section-execution.json").write_text(json.dumps(section, ensure_ascii=False, indent=2) + "\n")
receipt_path = ROOT / "docs/course-review-20261019/figure-renders-04.json"
receipt = json.loads(receipt_path.read_text())
record = next(row for row in receipt["records"] if row["page_id"] == "16.8")
assert digest(ROOT / record["png"]["path"]) == record["png"]["sha256"]
result["figure_receipt"] = {"path": str(receipt_path.relative_to(ROOT)), "sha256": digest(receipt_path), "record": record}
result["source_hashes"] = {name: digest(ROOT / name) for name in (
    "tiny_perceptron/attention.py", "tiny_perceptron/model.py", "scripts/course_experiments/architecture.py",
    "docs/course-review-20261019/freeze-01/sources/16.8.md", "course/figures/rewrite-16-causal-mask.svg",
    "docs/course-review-20261019/freeze-04.json")}
print(json.dumps(result, ensure_ascii=False, indent=2))
