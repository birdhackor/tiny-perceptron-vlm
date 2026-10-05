"""Bounded, synthetic CPU checks of 10.4; no training or model evaluation."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import VisionEncoder, patchify

torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(0)
assert torch.version.cuda is None and not torch.cuda.is_available()

patches = torch.ones(1, 4, 3)
position = torch.arange(4.0)[None, :, None]
added = patches + position
assert position.shape == (1, 4, 1)
assert added.shape == (1, 4, 3)
assert added[0, 0].tolist() == [1.0, 1.0, 1.0]
assert added[0, 3].tolist() == [4.0, 4.0, 4.0]
assert not torch.equal(added[:, 0], added[:, 3])
no_position = patches
assert no_position.shape == added.shape
assert torch.equal(no_position[:, 0], no_position[:, 3])

# Two distinct contents are necessary to distinguish relocation from a reorder;
# swapping the original all-identical patches would be unobservable.
content = torch.tensor([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0],
                         [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]])
permutation = torch.tensor([3, 1, 2, 0])
original = content + position
content_only = content[:, permutation] + position
paired = content[:, permutation] + position[:, permutation]
assert torch.equal(paired, original[:, permutation])
assert not torch.equal(content_only, paired)

# Directly inspect raw slot contents to tie raster indices to spatial cells.
image = torch.arange(4.0).reshape(1, 1, 2, 2).expand(1, 3, 2, 2)
raw_slots = patchify(image, 1)
assert raw_slots.shape == (1, 4, 3)
assert raw_slots[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]

# Inspect an actual constructor's contract; do not run a trained model.
model = VisionEncoder(width=3, image_size=2, patch_size=1)
assert model.position.shape == (1, 4, 3)
assert model.position.requires_grad
assert dict(model.named_parameters())["position"] is model.position
assert model.position.detach()[0, 0].tolist() != model.position.detach()[0, 3].tolist()
failure = None
try:
    torch.ones(1, 16, 3) + model.position
except RuntimeError as error:
    failure = str(error)
assert failure is not None
larger = VisionEncoder(width=3, image_size=4, patch_size=1)
assert larger.position.shape == (1, 16, 3)

result = {
    "original_arithmetic": {"position_shape": list(position.shape), "x_shape": list(added.shape),
                            "position_0": added[0, 0].tolist(), "position_3": added[0, 3].tolist(),
                            "equal": torch.equal(added[:, 0], added[:, 3])},
    "remove_position": {"x_shape": list(no_position.shape), "position_0": no_position[0, 0].tolist(),
                        "position_3": no_position[0, 3].tolist(), "equal": torch.equal(no_position[:, 0], no_position[:, 3])},
    "reorder": {"original": original.tolist(), "permutation": permutation.tolist(),
                "content_only": content_only.tolist(), "content_and_position": paired.tolist(),
                "paired_equals_permuted_original": torch.equal(paired, original[:, permutation]),
                "content_only_equals_paired": torch.equal(content_only, paired)},
    "raster_slots": raw_slots.tolist(),
    "constructor": {"position_shape": list(model.position.shape), "position_requires_grad": model.position.requires_grad,
                    "registered_parameter": True, "larger_position_shape": list(larger.position.shape),
                    "four_to_sixteen_addition_error": failure},
    "environment": {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "cuda_build": str(torch.version.cuda),
                    "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": torch.get_num_threads(),
                    "cwd": str(Path.cwd()), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")},
    "input_hashes": {"tiny_perceptron/multimodal.py": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest(),
                     "verify.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    "scope": "Exact tensor addition/reordering, raw patch ordering, and constructor shape/parameter inspection only. No optimizer, training, trained model inference, empirical scoring, or weights saved."
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
