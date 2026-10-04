"""Independent CPU arithmetic checks for section 11.14; no model loaded."""

import hashlib
import inspect
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture


torch.set_num_threads(1)
assert torch.version.cuda is None
first, second = swapped_picture()
changed = first != second
pool4_first = F.adaptive_avg_pool2d(first[None], (4, 4))
pool4_second = F.adaptive_avg_pool2d(second[None], (4, 4))
pool8_first = F.adaptive_avg_pool2d(first[None], (8, 8))
pool8_second = F.adaptive_avg_pool2d(second[None], (8, 8))

# For exact divisibility, explicitly reshape into 4x4 spatial bins of 8x8 pixels.
block_mean = first.reshape(3, 4, 8, 4, 8).mean(dim=(2, 4))[None]
assert torch.equal(block_mean, pool4_first)
assert int(changed.sum()) == 128
assert int(changed.any(dim=0).sum()) == 64
assert torch.equal(pool4_first, pool4_second)
assert not torch.equal(pool8_first, pool8_second)

# Paper exercise: keep blue in the first 8x8 cell and move red to the next cell.
exercise = first.clone()
exercise[0, 8:16, 0:4] = 0
exercise[0, 8:16, 8:12] = 1
exercise_pool = F.adaptive_avg_pool2d(exercise[None], (4, 4))
assert torch.equal(first.sum(dim=(1, 2)), exercise.sum(dim=(1, 2)))
assert not torch.equal(pool4_first, exercise_pool)

out = Path(__file__).resolve().parent
functional_path = Path(inspect.getfile(F))
snapshot = out / "natural-11.14-installed-functional.py"
snapshot.write_bytes(functional_path.read_bytes())
report = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": str(first.device),
        "dtype": str(first.dtype),
        "cuda_build": str(torch.version.cuda),
        "threads": torch.get_num_threads(),
    },
    "module_sha256": hashlib.sha256(Path(inspect.getfile(swapped_picture)).read_bytes()).hexdigest(),
    "installed_functional_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "original_report": picture_order_report(),
    "first_shape": list(first.shape),
    "second_shape": list(second.shape),
    "first_channel_sums_rgb": first.sum(dim=(1, 2)).tolist(),
    "second_channel_sums_rgb": second.sum(dim=(1, 2)).tolist(),
    "changed_rgb_scalar_values": int(changed.sum()),
    "changed_spatial_positions": int(changed.any(dim=0).sum()),
    "changed_by_channel_rgb": changed.sum(dim=(1, 2)).tolist(),
    "changed_row_range_inclusive": [int(changed.any(dim=0).nonzero()[:, 0].min()), int(changed.any(dim=0).nonzero()[:, 0].max())],
    "changed_column_range_inclusive": [int(changed.any(dim=0).nonzero()[:, 1].min()), int(changed.any(dim=0).nonzero()[:, 1].max())],
    "pool4_shape": list(pool4_first.shape),
    "pool4_equal": bool(torch.equal(pool4_first, pool4_second)),
    "pool4_equals_explicit_8x8_block_mean": bool(torch.equal(block_mean, pool4_first)),
    "pool4_nonzero_cell_row1_col0_rgb": pool4_first[0, :, 1, 0].tolist(),
    "pool4_nonzero_values": int(torch.count_nonzero(pool4_first)),
    "pool8_shape": list(pool8_first.shape),
    "pool8_equal": bool(torch.equal(pool8_first, pool8_second)),
    "pool8_changed_scalar_values": int((pool8_first != pool8_second).sum()),
    "pool8_first_row2_cells0_1_rgb": pool8_first[0, :, 2, :2].T.tolist(),
    "pool8_second_row2_cells0_1_rgb": pool8_second[0, :, 2, :2].T.tolist(),
    "flatten_shapes": [list(first.flatten().shape), list(second.flatten().shape)],
    "flatten_equal": bool(torch.equal(first.flatten(), second.flatten())),
    "exercise_move": "Move red rows 8:16 columns 0:4 to rows 8:16 columns 8:12; keep blue rows 8:16 columns 4:8.",
    "exercise_channel_sums_rgb": exercise.sum(dim=(1, 2)).tolist(),
    "exercise_pool4_equal": bool(torch.equal(pool4_first, exercise_pool)),
    "exercise_pool4_first_two_cells_row1_rgb": exercise_pool[0, :, 1, :2].T.tolist(),
    "exercise_pool4_changed_scalar_values": int((pool4_first != exercise_pool).sum()),
    "all_assertions_passed": True,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
