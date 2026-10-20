"""獨立核對像素差異、分格平均與位置交換。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture


def pool(image):
    return F.adaptive_avg_pool2d(image[None], (4, 4))[0]


first, second = swapped_picture()
expected = torch.zeros(3, 32, 32)
expected[0, 8:16, 0:4] = 1
expected[2, 8:16, 4:8] = 1
manual = first.reshape(3, 4, 8, 4, 8).mean((2, 4))
changed = first != second

moved = first.clone()
moved[0, 8:16, 0:4] = 0
moved[0, 8:16, 8:12] = 1

# 兩個不同排列的格子，各自都有相同的紅藍總量。
cross_before = first.clone()
cross_before[:, 8:16, 8:16] = second[:, 8:16, 0:8]
cross_after = cross_before.clone()
cross_after[:, 8:16, 0:8] = cross_before[:, 8:16, 8:16]
cross_after[:, 8:16, 8:16] = cross_before[:, 8:16, 0:8]

facts = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-14/verify_picture_order.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(first.device),
        "dtype": str(first.dtype),
        "platform": platform.platform(),
    },
    "implementation_sha256": hashlib.sha256(
        (ROOT / "tiny_perceptron/natural_concepts.py").read_bytes()
    ).hexdigest(),
    "helper_report": picture_order_report(),
    "construction": {
        "shape": list(first.shape),
        "first_matches_independent_construction": torch.equal(first, expected),
        "second_matches_first_eight_columns_flipped": torch.equal(second[:, :, :8], first[:, :, :8].flip(-1)),
        "remaining_columns_unchanged": torch.equal(second[:, :, 8:], first[:, :, 8:]),
        "changed_pixel_positions": int(changed.any(0).sum()),
        "changed_scalar_values": int(changed.sum()),
        "changed_values_by_rgb_channel": changed.sum((1, 2)).tolist(),
        "nonzero_changed_coordinate_min": changed.nonzero().min(0).values.tolist(),
        "nonzero_changed_coordinate_max": changed.nonzero().max(0).values.tolist(),
    },
    "pooling": {
        "output_shape": list(pool(first).shape),
        "manual_nonoverlapping_eight_by_eight_matches_api": torch.equal(manual, pool(first)),
        "original_and_swapped_summary_equal": torch.equal(pool(first), pool(second)),
        "occupied_cell_rgb_means": pool(first)[:, 1, 0].tolist(),
        "complete_flattened_sequences_equal": torch.equal(first.flatten(), second.flatten()),
        "flatten_then_reshape_restores_image": torch.equal(first.flatten().reshape(first.shape), first),
    },
    "exercise_move_red_to_columns_eight_through_eleven": {
        "summaries_equal": torch.equal(pool(first), pool(moved)),
        "old_cell_red_before_after": [float(pool(first)[0, 1, 0]), float(pool(moved)[0, 1, 0])],
        "new_cell_red_before_after": [float(pool(first)[0, 1, 1]), float(pool(moved)[0, 1, 1])],
        "changed_summary_values": int((pool(first) != pool(moved)).sum()),
    },
    "cross_cell_exchange_counterexample": {
        "operation": "將列 8:16、欄 0:8 與列 8:16、欄 8:16 兩個格子的完整像素互換；前者紅左藍右，後者藍左紅右。",
        "changed_pixel_positions": int((cross_before != cross_after).any(0).sum()),
        "changed_scalar_values": int((cross_before != cross_after).sum()),
        "summaries_equal": torch.equal(pool(cross_before), pool(cross_after)),
        "before_two_cells_rgb_means": pool(cross_before)[:, 1, :2].tolist(),
        "after_two_cells_rgb_means": pool(cross_after)[:, 1, :2].tolist(),
    },
}
assert facts["construction"]["first_matches_independent_construction"]
assert facts["construction"]["changed_values_by_rgb_channel"] == [64, 0, 64]
assert facts["pooling"]["manual_nonoverlapping_eight_by_eight_matches_api"]
assert facts["pooling"]["original_and_swapped_summary_equal"]
assert not facts["exercise_move_red_to_columns_eight_through_eleven"]["summaries_equal"]
assert facts["cross_cell_exchange_counterexample"]["changed_pixel_positions"] == 128
assert facts["cross_cell_exchange_counterexample"]["summaries_equal"]
print(json.dumps(facts, ensure_ascii=False, indent=2))
