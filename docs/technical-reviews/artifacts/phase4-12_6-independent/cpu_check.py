"""Bounded independent checks for 12.6; no dataset, training or weights."""
import ast
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Optional
import torch
from tiny_perceptron.multimodal import mel_filter_bank, AudioEncoder

root = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
official = root / 'official/torchaudio-functional-v2.5.1.py'
tree = ast.parse(official.read_bytes())
names = {'_hz_to_mel', '_mel_to_hz', '_create_triangular_filterbank', 'melscale_fbanks'}
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
namespace = {'torch': torch, 'Tensor': torch.Tensor, 'Optional': Optional, 'math': math, 'warnings': warnings}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(official), 'exec'), namespace)
records = {}
for bands in (16, 32):
    bank = mel_filter_bank(bands=bands)
    power = torch.ones(201, 3)
    out = bank @ power
    official_bank = namespace['melscale_fbanks'](201, 0., 8000., bands, 16000)
    error = (bank - official_bank.T).abs().max().item()
    assert tuple(bank.shape) == (bands, 201)
    assert tuple(out.shape) == (bands, 3)
    assert (bank >= 0).all() and torch.isfinite(bank).all()
    assert error <= 1e-6
    independent_columns = bank @ torch.stack((torch.ones(201), torch.full((201,), 2.), torch.full((201,), 3.)), dim=1)
    assert torch.allclose(independent_columns[:, 1], 2 * independent_columns[:, 0])
    assert torch.allclose(independent_columns[:, 2], 3 * independent_columns[:, 0])
    overlap = int(((bank > 0).sum(dim=0) == 2).sum())
    assert overlap > 0
    records[str(bands)] = {'bank_shape': list(bank.shape), 'output_shape': list(out.shape), 'nonnegative': bool((bank >= 0).all()), 'requires_grad': bank.requires_grad, 'official_transposed_max_abs_error': error, 'overlapping_frequency_bins': overlap, 'independent_time_columns': True, 'row_sum_first_last': [out[0, 0].item(), out[-1, 0].item()]}
mel_values = [2595 * math.log10(1 + f / 700) for f in (0, 700, 2100)]
assert [round(x) for x in mel_values] == [0, 781, 1562]
max_mel = 2595 * math.log10(1 + 8000 / 700)
mels = torch.linspace(0, max_mel, 18, dtype=torch.float64)
hz = 700 * (10 ** (mels / 2595) - 1)
spacing = hz.diff()
assert (spacing.diff() > 0).all()
example = torch.tensor([[1., .5, 0.], [0., .5, 1.]]) @ torch.tensor([4., 2., 1.])
assert torch.equal(example, torch.tensor([5., 2.]))
bank = mel_filter_bank()
p1 = torch.ones(201)
p2 = p1.clone(); p2[0] += 4
assert not torch.equal(p1, p2) and torch.equal(bank @ p1, bank @ p2)
rank = int(torch.linalg.matrix_rank(bank))
assert rank <= 16
encoder = AudioEncoder()
trainable = {name: list(parameter.shape) for name, parameter in encoder.named_parameters() if parameter.requires_grad}
assert 'projection.weight' in trainable and 'feature.1.weight' in trainable
records.update({'mel_formula': {'frequencies_hz': [0,700,2100], 'mel_values': mel_values, 'rounded': [round(x) for x in mel_values], 'equal_mel_step': max_mel/17, 'first_last_hz_spacing': [spacing[0].item(),spacing[-1].item()], 'hz_spacings_strictly_increasing': True}, 'hand_example': example.tolist(), 'information_loss': {'matrix_rank': rank, 'nullity_lower_bound': 201-16, 'distinct_nonnegative_power_inputs_same_output': True, 'changed_bin': 0, 'powers_at_changed_bin': [1,5]}, 'downstream_encoder': {'trainable_parameter_shapes': trainable, 'constructed_only': True, 'forward_or_update_executed': False}, 'environment': {'python_executable': __import__('sys').executable, 'python': __import__('sys').version, 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'cuda_build': str(torch.version.cuda), 'threads': str(torch.get_num_threads())}, 'source_sha256': hashlib.sha256((root/'extraction/section.md').read_bytes()).hexdigest()})
print(json.dumps(records, ensure_ascii=False, indent=2))
