"""Independent bounded CPU checks for the current 17.2; no training or model load."""
import json
import platform
import sys

import torch


torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()


def quantize(x, scale):
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    restored = q.float() * scale
    return q, restored


x = torch.tensor([-0.7, 0.2, 0.7, 1.2])
q, restored = quantize(x, 0.5)
assert q.tolist() == [-1, 0, 1, 2]
assert restored.tolist() == [-0.5, 0.0, 0.5, 1.0]
assert torch.allclose(x - restored, torch.tensor([-0.2, 0.2, 0.2, 0.2]), atol=1e-7, rtol=0)
assert q.dtype == torch.int8 and q.element_size() == 1
assert q.numel() == 4 and q.untyped_storage().nbytes() == 4
assert restored.dtype == torch.float32 and restored.device.type == "cpu"
assert not x.requires_grad and not q.requires_grad

q_fine, restored_fine = quantize(x, 0.25)
assert q_fine.tolist() == [-3, 1, 3, 5]
assert restored_fine.tolist() == [-0.75, 0.25, 0.75, 1.25]
assert torch.allclose(x - restored_fine, torch.tensor([0.05, -0.05, -0.05, -0.05]), atol=1e-7, rtol=0)

ties = torch.tensor([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25])
q_ties, restored_ties = quantize(ties, 0.5)
assert q_ties.tolist() == [-2, -2, 0, 0, 2, 2]
assert torch.equal((ties - restored_ties).abs(), torch.full_like(ties, 0.25))

outside = torch.tensor([-100.0, -64.0, -63.5, 63.5, 64.0, 100.0])
q_clip, restored_clip = quantize(outside, 0.5)
assert q_clip.tolist() == [-127, -127, -127, 127, 127, 127]
assert restored_clip.tolist() == [-63.5, -63.5, -63.5, 63.5, 63.5, 63.5]
assert (outside - restored_clip).abs().tolist() == [36.5, 0.5, 0.0, 0.0, 0.5, 36.5]

grid_inputs = torch.linspace(-63.5, 63.5, 25401)
q_grid, restored_grid = quantize(grid_inputs, 0.5)
max_grid_error = (grid_inputs - restored_grid).abs().max().item()
assert max_grid_error <= 0.25 + 1e-6

nearby = torch.tensor([0.51, 0.7, 0.74])
q_nearby, restored_nearby = quantize(nearby, 0.5)
assert q_nearby.tolist() == [1, 1, 1]
assert restored_nearby.tolist() == [0.5, 0.5, 0.5]

rounded_errors = (x - restored).round(decimals=4).tolist()
assert rounded_errors[1] != 0.2
assert abs(rounded_errors[1] - 0.2) < 1e-7
limits = torch.iinfo(torch.int8)
assert (limits.min, limits.max) == (-128, 127)
assert 127 * 0.25 == 31.75 and 127 * 0.5 == 63.5

print(json.dumps({
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "device": "cpu",
        "platform": platform.platform(),
        "default_dtype": str(torch.get_default_dtype()),
    },
    "baseline": {"shape": list(x.shape), "q": q.tolist(), "restored": restored.tolist(), "raw_signed_error": (x-restored).tolist(), "rounded_error_list": rounded_errors},
    "scale_025": {"q": q_fine.tolist(), "restored": restored_fine.tolist(), "signed_error": (x-restored_fine).tolist(), "positive_range_endpoint": 31.75},
    "ties_05": {"input": ties.tolist(), "q": q_ties.tolist(), "restored": restored_ties.tolist()},
    "clamp_05": {"input": outside.tolist(), "q": q_clip.tolist(), "restored": restored_clip.tolist(), "absolute_error": (outside-restored_clip).abs().tolist()},
    "in_range_grid": {"count": grid_inputs.numel(), "range": [-63.5, 63.5], "max_absolute_error": max_grid_error, "tolerance": 1e-6},
    "many_to_one": {"input": nearby.tolist(), "q": q_nearby.tolist(), "restored": restored_nearby.tolist()},
    "int8": {"min": limits.min, "max": limits.max, "used_min": -127, "used_max": 127, "used_code_count": 255, "bytes_per_element": q.element_size(), "baseline_elements": q.numel(), "baseline_storage_bytes": q.untyped_storage().nbytes()},
    "result": "All independent assertions passed; demonstration only, no backward or optimizer update."
}, indent=2, ensure_ascii=False))
