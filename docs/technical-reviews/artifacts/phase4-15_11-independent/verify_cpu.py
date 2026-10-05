"""Bounded CPU inference checks for 15.11; no training or model downloads."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modern import DenseFFN, MoEFFN

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def build(length):
    torch.manual_seed(0)
    x = torch.randn(2, length, 8)
    return x, DenseFFN(8, hidden=16), MoEFFN(8, experts=4, top_k=2, hidden=4)


def state_digest(layer):
    h = hashlib.sha256()
    for parameter in layer.parameters():
        h.update(parameter.detach().numpy().tobytes())
    return h.hexdigest()


results = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
           "torch_git_version": str(torch.version.git_version), "device": "cpu",
           "cuda_build": str(torch.version.cuda), "threads": torch.get_num_threads()},
           "timing_method": "5 warmup calls then mean of 30 forward calls; five repeats per length; no updates",
           "lengths": []}
for length in (8, 64):
    x, dense, moe = build(length)
    x2, dense2, moe2 = build(length)
    assert torch.equal(x, x2)
    assert state_digest(dense) == state_digest(dense2)
    assert state_digest(moe) == state_digest(moe2)
    item = {"shape": list(x.shape), "flattened_tokens": 2 * length, "seed_repeat_exact": True, "layers": {}}
    for name, layer in (("Dense", dense), ("MoE", moe)):
        before = state_digest(layer)
        layer.eval()
        assert not any(module.training for module in layer.modules())
        times = []
        with torch.no_grad():
            for _ in range(5):
                layer(x)
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(30):
                    layer(x)
                times.append((time.perf_counter() - start) * 1000 / 30)
            result = layer(x)
            output = result if name == "Dense" else result[0]
            assert list(output.shape) == list(x.shape)
            assert not output.requires_grad
            assert torch.isfinite(output).all()
            if name == "MoE":
                flat = x.reshape(-1, 8)
                probabilities = layer.router(flat).softmax(-1)
                weights, chosen = probabilities.topk(2, dim=-1)
                weights = weights / weights.sum(-1, keepdim=True)
                expected = torch.stack([
                    sum(layer.experts[int(chosen[row, slot])](flat[row]) * weights[row, slot]
                        for slot in range(2))
                    for row in range(2 * length)
                ])
                error = float((expected - output.reshape(-1, 8)).abs().max())
                assert error < 1e-6
                item["manual_two_expert_merge_max_abs_error"] = error
                assert list(result[2].shape) == [2 * length, 2]
                assert torch.equal(result[2], chosen)
        assert before == state_digest(layer)
        assert all(parameter.grad is None for parameter in layer.parameters())
        counts = {n: {"shape": list(p.shape), "numel": p.numel()} for n, p in layer.named_parameters()}
        params = sum(p.numel() for p in layer.parameters())
        assert params == (280 if name == "Dense" else 336)
        item["layers"][name] = {"parameters": params, "parameter_tables": counts,
            "milliseconds_per_forward": times, "output_shape": list(output.shape),
            "no_reverse_graph": not output.requires_grad, "unchanged_parameters": True,
            "all_parameter_gradients_none": True, "state_sha256": before}
    item["moe_dense_time_ratios"] = [m / d for m, d in zip(
        item["layers"]["MoE"]["milliseconds_per_forward"],
        item["layers"]["Dense"]["milliseconds_per_forward"])]
    results["lengths"].append(item)
results["matrix_weight_derivation"] = {"dense": "8*16+16*8=256", "two_experts": "2*(8*4+4*8)=128",
    "router": "8*4=32", "dense_total": "256+16+8=280", "moe_total": "4*(64+4+8)+32=336",
    "scope": "Counts matrix entries used per token; excludes bias, activation, routing/indexing/combine and auxiliary work; not FLOPs or quality."}
Path(__file__).with_name("cpu-verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"shapes": [x["shape"] for x in results["lengths"]],
    "parameters": {k:v["parameters"] for k,v in results["lengths"][0]["layers"].items()},
    "milliseconds_per_forward": [{k:v["milliseconds_per_forward"] for k,v in x["layers"].items()} for x in results["lengths"]],
    "manual_merge_max_abs_errors": [x["manual_two_expert_merge_max_abs_error"] for x in results["lengths"]],
    "no_parameter_updates": True}, ensure_ascii=False))
