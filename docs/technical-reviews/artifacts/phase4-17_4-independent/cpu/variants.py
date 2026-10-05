"""Bounded independent CPU checks for lesson 17.4; no training or downloads."""

import hashlib
import json
import os
import platform
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.quantization import pack_int4, quantize_affine, unpack_int4

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def original_method(values):
    x = torch.tensor(values)
    low = min(x.min().item(), 0.0)
    high = max(x.max().item(), 0.0)
    scale = (high - low) / 15
    zero = round(-low / scale)
    q = (x / scale + zero).round().clamp(0, 15).to(torch.uint8)
    restored = (q.float() - zero) * scale
    return x, low, high, scale, zero, q, restored


records = []
for name, values, expected_q, expected_zero in [
    ("original", [-1.0, 0.0, 1.0, 2.0], [0, 5, 10, 15], 5),
    ("exercise", [-2.0, 0.0, 1.0], [0, 10, 15], 10),
    ("off_grid", [-1.0, 0.07, 1.01, 2.0], [0, 5, 10, 15], 5),
    ("positive_only", [0.5, 2.0], [4, 15], 0),
    ("negative_only", [-2.0, -0.5], [0, 11], 15),
]:
    x, low, high, scale, zero, q, restored = original_method(values)
    assert q.tolist() == expected_q and zero == expected_zero
    assert low <= 0 <= high
    assert q.dtype == torch.uint8 and q.element_size() == 1
    assert q.numel() * q.element_size() == len(values)
    assert restored.dtype == torch.float32
    assert torch.all((x - restored).abs() <= scale / 2 + 1e-6)
    if name in ("original", "exercise"):
        assert scale == 0.2
        assert torch.equal(restored, x)
    if name == "off_grid":
        assert (x - restored).abs().max().item() > 0
    records.append({
        "name": name, "values": values, "low": low, "high": high,
        "scale": scale, "zero": zero, "q": q.tolist(),
        "restored": restored.tolist(), "abs_error": (x - restored).abs().tolist(),
        "container_bytes": q.numel() * q.element_size(),
    })

# Degenerate input fails in the unmodified educational formula, as disclosed.
try:
    original_method([0.0, 0.0])
except ZeroDivisionError as error:
    zero_case = {"original_exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("The original formula has no zero-width safeguard")
q0, scale0, zero0 = quantize_affine(torch.zeros(2), bits=4)
assert scale0 > 0 and q0.tolist() == [0, 0] and zero0.item() == 0
assert torch.equal((q0.float() - zero0) * scale0, torch.zeros(2))
zero_case.update(repository_scale=scale0.item(), repository_zero=zero0.item(), repository_q=q0.tolist())

# Control saturation and the elementwise rounding API, separately from calibration.
v = torch.tensor([-2.0, -1.0, 0.0, 2.0, 3.0])
clipped = (v / 0.2 + 5).round().clamp(0, 15).to(torch.uint8)
assert clipped.tolist() == [0, 0, 5, 15, 15]
midpoints = torch.tensor([-0.5, 0.5, 1.5, 2.5])
assert midpoints.round().tolist() == [-0.0, 0.0, 2.0, 2.0]

# The later packer maps signed integers to offset nibbles, with its own zero code.
signed = torch.arange(-8, 8, dtype=torch.int8)
packed = pack_int4(signed)
assert packed.tolist() == [16, 50, 84, 118, 152, 186, 220, 254]
assert packed.numel() == 8 and packed.element_size() == 1
assert torch.equal(unpack_int4(packed, signed.shape), signed)
assert pack_int4(torch.tensor([0, 0], dtype=torch.int8)).tolist() == [136]
try:
    pack_int4(torch.tensor([0, 5, 10, 15], dtype=torch.uint8))
except ValueError as error:
    incompatible_affine_codes = str(error)
else:
    raise AssertionError("Unsigned affine codes >= 8 cannot be passed to this signed packer")

# Exact rational calculation distinguishes 16 codes from 15 intervals.
scale_exact = Fraction(2 - (-1), (2**4 - 1) - 0)
assert scale_exact == Fraction(1, 5)
assert [-1 + i * scale_exact for i in (0, 5, 10, 15)] == [-1, 0, 1, 2]
assert (Fraction(-(-2)) / scale_exact) == 10

result = {
    "environment": {
        "python": sys.version, "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version), "platform": platform.platform(),
        "device": "cpu", "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()),
        "offline": str({k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}),
    },
    "cases": records, "all_zero": zero_case,
    "clipped_codes": clipped.tolist(), "round_half_to_even": midpoints.round().tolist(),
    "packer": {"signed_values": signed.tolist(), "packed": packed.tolist(), "zero_pair_byte": 136,
               "incompatible_affine_codes": incompatible_affine_codes},
    "rational_derivation": {"codes": 16, "intervals": 15, "scale": str(scale_exact), "original_zero": 5, "exercise_zero": 10},
    "original_module_sha256": hashlib.sha256((ROOT / "tiny_perceptron/quantization.py").read_bytes()).hexdigest(),
    "result": "All assertions passed; finite CPU demonstration only, no model/training evaluation.",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
