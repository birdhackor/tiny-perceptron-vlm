"""Independent bounded CPU check of the original 11.14 calculation and its exercise."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture

torch.set_num_threads(1)
torch.set_default_device('cpu')
a, b = swapped_picture()
assert a.shape == b.shape == (3, 32, 32)
assert a.device.type == b.device.type == 'cpu'
assert not a.requires_grad and not b.requires_grad
pa, pb = (F.adaptive_avg_pool2d(x[None], (4, 4)) for x in (a, b))
assert pa.shape == pb.shape == (1, 3, 4, 4)
# Divide the CHW array into (channel, grid row, within-row, grid col, within-col).
# This independent block reduction does not call adaptive_avg_pool2d.
manual_a = a.reshape(3, 4, 8, 4, 8).mean(dim=(2, 4))[None]
manual_b = b.reshape(3, 4, 8, 4, 8).mean(dim=(2, 4))[None]
assert torch.equal(pa, manual_a) and torch.equal(pb, manual_b)
assert torch.equal(pa, pb)
assert torch.equal(a.sum(dim=(1, 2)), b.sum(dim=(1, 2)))
assert pa[0, :, 1, 0].tolist() == [0.5, 0.0, 0.5]
channel_values_changed = int((a != b).sum())
coordinates_changed = int((a != b).any(dim=0).sum())
assert channel_values_changed == 128
assert coordinates_changed == 64
assert (a != b).sum(dim=(1, 2)).tolist() == [64, 0, 64]
assert not torch.equal(a.flatten(), b.flatten())
assert torch.equal(a.flatten().reshape(3, 32, 32), a)
assert a[0, 8, 2] == 1 and a[2, 8, 6] == 1
assert b[2, 8, 2] == 1 and b[0, 8, 6] == 1
expected_report = {
    'different_pixels': 128,
    'same_4_by_4_summary': True,
    'same_pixel_sequence': False,
}
assert picture_order_report() == expected_report

# Exercise: relocate only the 8x4 red bar from grid col 0 to grid col 1.
moved = a.clone()
moved[0, 8:16, 0:4] = 0
moved[0, 8:16, 8:12] = 1
pm = F.adaptive_avg_pool2d(moved[None], (4, 4))
assert torch.equal(moved.sum(dim=(1, 2)), a.sum(dim=(1, 2)))
assert not torch.equal(pm, pa)
assert pm[0, :, 1, 0].tolist() == [0.0, 0.0, 0.5]
assert pm[0, :, 1, 1].tolist() == [0.5, 0.0, 0.0]
assert int((pm != pa).sum()) == 2

result = {
    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                    'torch_git': torch.version.git_version, 'device': 'cpu',
                    'cuda_build': str(torch.version.cuda), 'threads': torch.get_num_threads()},
    'source_code_sha256': hashlib.sha256((ROOT/'tiny_perceptron/natural_concepts.py').read_bytes()).hexdigest(),
    'axes': {'input': 'C,H,W = 3,32,32', 'pooled': 'N,C,Hout,Wout = 1,3,4,4',
             'flatten': 'channel-major, then row, then column; 3072 scalar values'},
    'units': {'changed_rgb_scalar_values': channel_values_changed,
              'changed_color_coordinates': coordinates_changed,
              'changed_values_by_channel_RGB': (a != b).sum(dim=(1, 2)).tolist(),
              'pool_denominator_per_channel': 64, 'red_pixels': 32, 'blue_pixels': 32},
    'original_report': expected_report,
    'original_cell_RGB_mean': pa[0, :, 1, 0].tolist(),
    'global_RGB_sum': a.sum(dim=(1, 2)).tolist(),
    'within_cell_swap_max_absolute_pool_difference': float((pa-pb).abs().max()),
    'independent_pool_equals_pytorch_exactly': True,
    'cross_cell_move': {'original_cell_RGB_mean': pm[0, :, 1, 0].tolist(),
                        'new_cell_RGB_mean': pm[0, :, 1, 1].tolist(),
                        'changed_summary_values': int((pm != pa).sum()),
                        'summary_equal': bool(torch.equal(pm, pa)),
                        'global_RGB_sum_unchanged': True},
    'assertions_executed': 21,
    'tolerance': 'Exact equality: inputs 0/1, means 0/.5 have exact binary representations.',
    'scope': 'Synthetic deterministic tensors only; no model loading, backward, parameter update, training or natural-image evaluation.',
}
print(json.dumps(result, ensure_ascii=False, indent=2))
