"""核對位置向量示範、練習、初始化與切塊對應；不更新模型參數。"""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import VisionEncoder, patchify

torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(0)

source = ROOT / "docs/course-review-20261019/freeze-01/sources/10.4.md"
original = source.read_bytes()
code = re.search(r"```python\n(.*?)```", original.decode(), re.S)[1]
namespace = {}
exec(code, namespace)
patches, position, x = (namespace[key] for key in ("patches", "position", "x"))
assert patches.shape == (1, 4, 3)
assert position.shape == (1, 4, 1)
assert torch.equal(position.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
assert torch.equal(x, torch.tensor([[[1.0] * 3, [2.0] * 3, [3.0] * 3, [4.0] * 3]]))
assert not torch.equal(x[:, 0], x[:, 3])

exercise = {}
exec(code.replace("x = patches + position", "x = patches"), exercise)
assert torch.equal(exercise["x"][:, 0], exercise["x"][:, 3])
assert exercise["x"].shape == x.shape

encoder = VisionEncoder(width=8)
saved_position = encoder.position.detach().clone()
torch.manual_seed(0)
replayed_encoder = VisionEncoder(width=8)
torch.manual_seed(0)
nn.Linear(48, 8)
expected_position = torch.randn(1, 16, 8) * 0.02
assert torch.equal(replayed_encoder.position, expected_position)
assert encoder.position.shape == (1, 16, 8)
assert isinstance(encoder.position, nn.Parameter)
assert encoder.position.requires_grad
assert dict(encoder.named_parameters())["position"] is encoder.position
assert not torch.equal(saved_position[0, 0], saved_position[0, 3])
assert torch.unique(saved_position[0, 0]).numel() == 8

grid = torch.arange(16.0).reshape(1, 1, 4, 4)
four_patches = patchify(grid, 2)
assert four_patches[0, 3].tolist() == [10.0, 11.0, 14.0, 15.0]
large_grid = torch.arange(256.0).reshape(1, 1, 16, 16)
sixteen_patches = patchify(large_grid, 4)
assert sixteen_patches[0, 3, 0].item() == 12.0
assert sixteen_patches[0, 15, 0].item() == 204.0
try:
    torch.zeros(1, 4, 8) + encoder.position
except RuntimeError as error:
    mismatch_error = str(error)
else:
    raise AssertionError("不同位置數的提示不應直接相加成功")

content = torch.tensor([1, 0])
tips = torch.tensor([0, 10])
permutation = torch.tensor([1, 0])
original_sum = content + tips
content_swap = content[permutation] + tips
joint_swap = content[permutation] + tips[permutation]
assert original_sum.tolist() == [1, 10]
assert content_swap.tolist() == [0, 11]
assert joint_swap.tolist() == [10, 1]
assert torch.equal(joint_swap, original_sum[permutation])
associations = sorted(zip(content.tolist(), tips.tolist()))
joint_associations = sorted(zip(content[permutation].tolist(), tips[permutation].tolist()))
content_only_associations = sorted(zip(content[permutation].tolist(), tips.tolist()))
assert associations == joint_associations
assert associations != content_only_associations

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-4/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": str(patches.device),
        "dtype": str(patches.dtype),
        "seed": "0",
        "threads": "1",
    },
    "source_sha256": hashlib.sha256(original).hexdigest(),
    "repository_code_sha256": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest(),
    "original": {"patches": patches.tolist(), "position": position.tolist(), "x": x.tolist(), "shape": list(x.shape), "equal": torch.equal(x[:, 0], x[:, 3])},
    "exercise": {"x": exercise["x"].tolist(), "shape": list(exercise["x"].shape), "equal": torch.equal(exercise["x"][:, 0], exercise["x"][:, 3])},
    "encoder": {"position_shape": list(encoder.position.shape), "trainable_registered_parameter": True, "initialization_exactly_replayed": True, "initialization_formula": "torch.randn(1, 16, 8) * 0.02", "first_position": saved_position[0, 0].tolist()},
    "patch_order": {"four_patch_grid": four_patches.tolist(), "sixteen_patch_index3_first_value": sixteen_patches[0, 3, 0].item(), "sixteen_patch_index15_first_value": sixteen_patches[0, 15, 0].item(), "incompatible_position_count_error": mismatch_error},
    "reordering": {"original": original_sum.tolist(), "content_only_swap": content_swap.tolist(), "joint_swap": joint_swap.tolist(), "original_associations": associations, "joint_associations": joint_associations, "content_only_associations": content_only_associations},
    "result": "所有數值、維度、初始化與配對關係斷言通過。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
