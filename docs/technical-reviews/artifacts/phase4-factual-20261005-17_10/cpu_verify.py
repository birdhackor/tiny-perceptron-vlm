"""Bounded independent CPU checks for current section 17.10; no training or weights retained."""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.quantization import QuantizedLinear

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.manual_seed(0)
print("ENV", json.dumps({"python": sys.version, "torch": torch.__version__,
    "torch_git": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)}, ensure_ascii=False))

layer = nn.Linear(64, 32)
assert tuple(layer.weight.shape) == (32, 64)
assert layer.weight.dtype == torch.float32
assert sum(p.numel() * p.element_size() for p in layer.parameters()) == 8320
original_linear = F.linear
for bits, codes, total in [(4, 1024, 1280), (8, 2048, 2304)]:
    compressed = QuantizedLinear(layer, bits=bits)
    assert compressed.values.numel() * compressed.values.element_size() == codes
    assert tuple(compressed.scale.shape) == (32, 1)
    assert compressed.scale.numel() * compressed.scale.element_size() == 128
    assert compressed.bias.numel() * compressed.bias.element_size() == 128
    assert compressed.storage_bytes() == total
    observed = []
    def inspect_linear(x, weight, bias):
        observed.append({"weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype),
                         "weight_bytes": weight.numel() * weight.element_size()})
        return original_linear(x, weight, bias)
    F.linear = inspect_linear
    try:
        result = compressed(torch.ones(2, 64))
    finally:
        F.linear = original_linear
    assert tuple(result.shape) == (2, 32) and result.dtype == torch.float32
    assert observed == [{"weight_shape": [32, 64], "weight_dtype": "torch.float32", "weight_bytes": 8192}]
    print("LINEAR", json.dumps({"bits": bits, "code_bytes": codes, "scale_bytes": 128,
        "bias_bytes": 128, "total_bytes": total, "result_shape": list(result.shape),
        "result_dtype": str(result.dtype), "forward_argument": observed}, ensure_ascii=False))

variation = QuantizedLinear(layer, bits=4)(torch.ones(2, 64, dtype=torch.float64))
assert variation.dtype == torch.float64
print("DTYPE_VARIATION", str(variation.dtype), "input dtype determines reconstructed weight/output dtype")

# Execute the historical original helper AST, including @torch.no_grad(), with
# a small CPU model and a fixed-length prompt substitute. No model is downloaded.
source = (BASE / "historical-compression.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == "28d8ce258369672d71a24f7efbcbf68f3e5e8114e073620aba91c4d807043a44"
tree = ast.parse(source)
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_sync", "_timing")]
assert len(selected) == 2
namespace = {"torch": torch, "time": time, "_prompt": lambda record, max_length: torch.arange(46)}
exec(compile(ast.Module(body=selected, type_ignores=[]), "historical-compression.py:_sync,_timing", "exec"), namespace)
class ProbeModel(nn.Module):
    def __init__(self, max_length=64):
        super().__init__()
        self.config = SimpleNamespace(max_length=max_length)
        self.lengths = []
        self.grad_flags = []
    def forward(self, ids):
        self.lengths.append(ids.shape[1])
        self.grad_flags.append(torch.is_grad_enabled())
        # Zero logits select token 0 on every step; the helper has no EOS branch.
        return {"logits": torch.zeros(ids.shape[0], ids.shape[1], 3)}
probe = ProbeModel()
timing = namespace["_timing"](probe, {}, "cpu")
expected_lengths = [46] + ([46] + list(range(47, 55))) * 3
assert probe.lengths == expected_lengths
assert not any(probe.grad_flags) and not probe.training
assert timing["repetitions"] == 3 and timing["prompt_tokens"] == 46 and timing["decode_tokens"] == 8
assert abs(timing["decode_seconds"] / 8 - timing["decode_seconds_per_token"]) < 1e-12
assert timing["cuda_allocated_before"] is None and timing["cuda_peak_allocated"] is None
print("HELPER_CPU_CONTRACT", json.dumps({"calls": len(probe.lengths), "lengths": probe.lengths,
    "warmup_calls": 1, "prefill_calls": 3, "decode_calls": 24,
    "no_grad": not any(probe.grad_flags), "eos_early_stop": False,
    "decode_denominator": "3 repetitions x 8 steps = 24", "GPU_probe_executed": False}, ensure_ascii=False))
too_short = ProbeModel(53)
short_result = namespace["_timing"](too_short, {}, "cpu")
assert short_result["measured"] is False and too_short.lengths == []
print("HELPER_LENGTH_GUARD", "46 + 8 > 53: returns unmeasured before forward")

# Inspect only named measurement and provenance pointers of the retained raw JSON.
raw_path = BASE / "original-l4-quantization.json"
raw = raw_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "30e1d7cd617601bd6d99524efd635210337862124e6eb657e9a0fd2c814beb8a"
data = json.loads(raw)
print("RAW_IDENTITY", json.dumps({"sha256": hashlib.sha256(raw).hexdigest(),
    "revision": data["revision"], "device": data["device"], "gpu": data["gpu"],
    "seed": data["seed"], "torch_version": data["torch_version"], "python_version": data["python_version"]}))
fields = ("measured", "repetitions", "prompt_tokens", "decode_tokens", "prefill_seconds",
          "decode_seconds", "decode_seconds_per_token", "cuda_allocated_before",
          "cuda_peak_allocated", "cuda_additional_peak_bytes")
selected_timings = {}
for name, prefill_ms, decode_ms, additional in [
    ("fp32", 2.405, 2.371, 335872),
    ("packed4", 4.958, 4.882, 376320),
    ("packed8", 2.768, 2.814, 359936),
]:
    t = {k: data["results"]["runs"][name]["timing"][k] for k in fields}
    assert t["measured"] is True and t["repetitions"] == 3
    assert t["prompt_tokens"] == 46 and t["decode_tokens"] == 8
    assert abs(t["prefill_seconds"] * 1000 - prefill_ms) <= 0.0005
    assert abs(t["decode_seconds_per_token"] * 1000 - decode_ms) <= 0.0005
    assert abs(t["decode_seconds"] / 8 - t["decode_seconds_per_token"]) < 1e-15
    assert t["cuda_allocated_before"] == 68655616
    assert t["cuda_peak_allocated"] - t["cuda_allocated_before"] == t["cuda_additional_peak_bytes"] == additional
    selected_timings[name] = t
    print("RAW_POINTER", "/results/runs/" + name + "/timing", json.dumps(t))
assert selected_timings["packed4"]["prefill_seconds"] > selected_timings["fp32"]["prefill_seconds"]
assert selected_timings["packed4"]["decode_seconds_per_token"] > selected_timings["fp32"]["decode_seconds_per_token"]
assert data["peak_allocated_bytes"] == selected_timings["packed8"]["cuda_peak_allocated"] == 69015552
assert data["peak_allocated_bytes"] < selected_timings["packed4"]["cuda_peak_allocated"]
print("RAW_OUTER_PEAK", data["peak_allocated_bytes"], "equals final packed8 probe peak, below earlier packed4 peak")
print("CHECKS_PASSED", "all bounded CPU checks; no GPU timing/model-quality reevaluation/training")
