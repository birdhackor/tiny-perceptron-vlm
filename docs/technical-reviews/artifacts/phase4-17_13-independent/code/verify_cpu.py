import hashlib
import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

import torch
from tiny_perceptron.quantization import quantize_symmetric

torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None
root = Path(__file__).resolve().parents[1]
records = []
for extreme in (100, 2):
    rational_x = [F(1, 10), F(1, 5), F(3, 10), F(extreme)]
    x = torch.tensor([float(v) for v in rational_x], dtype=torch.float32)
    for clip in (False, True):
        rational_source = [min(F(1), max(F(-1), v)) for v in rational_x] if clip else rational_x
        exact_scale = max(abs(v) for v in rational_source) / 7
        exact_q = [min(7, max(-7, round(v / exact_scale))) for v in rational_source]
        exact_restored = [q * exact_scale for q in exact_q]
        exact_error = [abs(a - b) for a, b in zip(rational_x, exact_restored)]
        exact_small_mae = sum(exact_error[:3]) / 3
        exact_all_mae = sum(exact_error) / 4
        source = x.clamp(-1, 1) if clip else x
        q, scale = quantize_symmetric(source, bits=4)
        restored = q.float() * scale
        error = (x - restored).abs()
        assert q.dtype == torch.int8 and q.tolist() == exact_q
        assert scale.shape == torch.Size([]) and scale.dtype == torch.float32
        assert torch.allclose(restored.double(), torch.tensor([float(v) for v in exact_restored], dtype=torch.float64), atol=1e-5, rtol=0)
        observed = [error[:3].mean().item(), error[-1].item(), error.mean().item()]
        expected = [float(exact_small_mae), float(exact_error[-1]), float(exact_all_mae)]
        assert all(abs(a - b) < 1e-5 for a, b in zip(observed, expected))
        record = {
            "extreme": extreme, "clip": clip, "x_shape": list(x.shape),
            "source": source.tolist(), "q": q.tolist(), "q_dtype": str(q.dtype),
            "scale": scale.item(), "scale_exact": str(exact_scale), "scale_shape": list(scale.shape),
            "restored": restored.tolist(), "restored_exact": [str(v) for v in exact_restored],
            "absolute_error": error.tolist(), "absolute_error_exact": [str(v) for v in exact_error],
            "small_mae": observed[0], "small_mae_exact": str(exact_small_mae),
            "extreme_difference": observed[1], "all_mae": observed[2], "all_mae_exact": str(exact_all_mae),
            "denominators": {"small_values": 3, "all_values": 4, "extreme_values": 1},
            "error_reference": "original x, not clipped source",
            "q_bytes": q.numel() * q.element_size(), "scale_bytes": scale.numel() * scale.element_size(),
        }
        records.append(record)
        print(json.dumps(record))

# Shrink scale groups without changing the four inputs or their bit-width.
x = torch.tensor([0.1, 0.2, 0.3, 100.0])
group_q = []
group_scale = []
group_restored = []
for segment in (x[:3], x[3:]):
    q, scale = quantize_symmetric(segment, bits=4)
    group_q.append(q)
    group_scale.append(scale)
    group_restored.append(q.float() * scale)
restored = torch.cat(group_restored)
error = (x - restored).abs()
group_record = {
    "groups": [[0, 1, 2], [3]], "q": torch.cat(group_q).tolist(),
    "scales": [s.item() for s in group_scale], "restored": restored.tolist(),
    "small_mae": error[:3].mean().item(), "all_mae": error.mean().item(),
    "q_bytes": sum(q.numel() * q.element_size() for q in group_q),
    "scale_bytes": sum(s.numel() * s.element_size() for s in group_scale),
    "storage_contract": "reference int8 container, not packed 4-bit storage; excludes headers",
}
assert group_record["scale_bytes"] == 8 and group_record["q_bytes"] == 4
assert group_record["small_mae"] < records[0]["small_mae"]
print(json.dumps({"group_variant": group_record}))

# Verify the rounding contract and sign symmetry used by this quantizer.
ties = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]).round()
assert ties.tolist() == [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]
q_zero, scale_zero = quantize_symmetric(torch.zeros(4), bits=4)
assert q_zero.tolist() == [0, 0, 0, 0] and scale_zero.item() > 0
negative = -torch.tensor([0.1, 0.2, 0.3, 100.0])
negative_q, negative_scale = quantize_symmetric(negative, bits=4)
assert negative_q.tolist() == [0, 0, 0, -7]
api_record = {"round_half_even": ties.tolist(), "zero_input_scale": scale_zero.item(),
              "negative_q": negative_q.tolist(), "negative_scale": negative_scale.item()}
print(json.dumps({"api_boundary_variant": api_record}))

environment = {"python": sys.version, "python_executable": sys.executable,
               "torch": torch.__version__, "torch_git_version": torch.version.git_version,
               "device": "cpu", "cuda_available": torch.cuda.is_available(),
               "cuda_build": torch.version.cuda, "platform": platform.platform(),
               "threads": torch.get_num_threads(), "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / "execution" / "cpu-values.json").write_text(json.dumps({"records": records, "group_variant": group_record,
    "api_boundary_variant": api_record}, indent=2, allow_nan=False) + "\n")
(root / "execution" / "cpu-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print("All numeric, denominator, dtype, shape, rounding and scale-storage assertions passed.")
