"""Bounded CPU checks of section 10.2, independent of the inverse alone."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import patchify, scene, unpatchify

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")


def inspect_coordinates(patch_size, batches=2, height=16, width=16):
    # Distinguish batch, RGB channel, image row and image column.
    values = torch.empty(batches, 3, height, width, dtype=torch.int64)
    for b in range(batches):
        for c in range(3):
            for y in range(height):
                for x in range(width):
                    values[b, c, y, x] = 10000 * b + 1000 * c + 100 * y + x
    actual = patchify(values, patch_size)
    rows, columns = height // patch_size, width // patch_size
    expected_shape = (batches, rows * columns, 3 * patch_size**2)
    assert tuple(actual.shape) == expected_shape
    checked = 0
    for b in range(batches):
        for patch_row in range(rows):
            for patch_column in range(columns):
                n = patch_row * columns + patch_column
                for c in range(3):
                    for local_y in range(patch_size):
                        for local_x in range(patch_size):
                            k = c * patch_size**2 + local_y * patch_size + local_x
                            global_y = patch_row * patch_size + local_y
                            global_x = patch_column * patch_size + local_x
                            expected = 10000 * b + 1000 * c + 100 * global_y + global_x
                            assert actual[b, n, k].item() == expected
                            checked += 1
    restored = unpatchify(actual, 3, height, width, patch_size)
    assert torch.equal(values, restored)
    assert checked == values.numel() == actual.numel()
    return {
        "patch_size": patch_size,
        "input_shape": list(values.shape),
        "patch_shape": list(actual.shape),
        "checked_individual_coordinates": checked,
        "roundtrip_equal": True,
        "patch_top_left_rgb_b0": [
            [actual[0, n, c * patch_size**2].item() for c in range(3)]
            for n in range(rows * columns)
        ],
    }


image = scene()
batched = image[None]
assert tuple(image.shape) == (3, 16, 16)
assert tuple(batched.shape) == (1, 3, 16, 16)
assert torch.equal(batched[0], image)
assert batched.data_ptr() == image.data_ptr()
original = patchify(batched, 4)
assert tuple(original.shape) == (1, 16, 48)
assert torch.equal(batched, unpatchify(original, 3, 16, 16, 4))
eight = patchify(batched, 8)
assert tuple(eight.shape) == (1, 4, 192)
assert torch.equal(batched, unpatchify(eight, 3, 16, 16, 8))

# Known RGB and pixel positions in the actual scene, independent of unpatchify.
scene_samples = []
for n, c, local_y, local_x in [(0, 0, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0),
                               (10, 0, 0, 0), (15, 0, 0, 0), (15, 0, 3, 3)]:
    y, x = (n // 4) * 4 + local_y, (n % 4) * 4 + local_x
    k = c * 16 + local_y * 4 + local_x
    observed = original[0, n, k].item()
    # scene() has the red square at inclusive rows/columns 4 through 12.
    expected = float(c == 0 and 4 <= y <= 12 and 4 <= x <= 12)
    assert observed == expected == image[c, y, x].item()
    scene_samples.append({"patch": n, "feature_offset": k, "image_cyx": [c, y, x],
                          "expected": expected, "observed": observed})

# Four distinct cards. A pair using the same column-first permutation still
# roundtrips, while a direct check of position 1 rejects that convention.
cards = torch.tensor([[[[11, 22], [33, 44]]]])
row_first = patchify(cards, 1)
assert row_first.flatten().tolist() == [11, 22, 33, 44]
assert row_first[0, 1, 0].item() == 22  # upper-right b
permutation = [0, 2, 1, 3]
column_first = row_first[:, permutation]
assert column_first.flatten().tolist() == [11, 33, 22, 44]
assert torch.equal(cards, unpatchify(column_first[:, permutation], 1, 2, 2, 1))
assert column_first[0, 1, 0].item() != 22

result = {
    "environment": {"python": sys.version, "python_executable": sys.executable,
                    "torch": str(torch.__version__), "torch_git_version": torch.version.git_version,
                    "device": "cpu", "cuda_build": str(torch.version.cuda),
                    "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "repository_module_sha256": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest(),
    "none_adds_batch_axis_without_changing_values": True,
    "scene_p4_shape": list(original.shape), "scene_p4_roundtrip": True,
    "scene_p8_shape": list(eight.shape), "scene_p8_roundtrip": True,
    "scene_known_rgb_coordinates": scene_samples,
    "coordinates": [inspect_coordinates(4), inspect_coordinates(8),
                    inspect_coordinates(4, height=8, width=12)],
    "cards": {"encoding": {"a": 11, "b": 22, "c": 33, "d": 44},
              "row_first": row_first.flatten().tolist(),
              "column_first": column_first.flatten().tolist(),
              "column_first_pair_roundtrip": True,
              "direct_second_position_rejects_column_first": True},
    "scope": "Only permutation, coordinates, channels, arithmetic and reconstruction; no optimization, training or model inference.",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
