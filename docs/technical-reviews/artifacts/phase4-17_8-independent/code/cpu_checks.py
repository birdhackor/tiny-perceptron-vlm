"""Independent bounded CPU checks for section 17.8; no training or weights saved."""
import ast
import hashlib
import itertools
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.quantization import (
    QuantizedLinear, pack_int4, unpack_int4, quantize_symmetric,
    replace_linear_layers,
)
from tiny_perceptron.model import TinyLM, ModelConfig

torch.set_num_threads(1)
torch.manual_seed(1708)
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'inputs/docs/course-experiments/results/quantization.json'
raw = json.loads(RAW.read_bytes())
print('ENV', json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
      'torch_git': torch.version.git_version, 'cuda_build': torch.version.cuda,
      'device': 'cpu', 'num_threads': torch.get_num_threads()}, sort_keys=True))
assert torch.version.cuda is None

q = torch.tensor([-8, -1, 0, 1, 7, 2, 3], dtype=torch.int8)
p = pack_int4(q)
assert p.tolist() == [112, 152, 175, 139]
assert p.dtype == torch.uint8 and p.element_size() == 1
assert q.element_size() == 1 and q.numel() * q.element_size() == 7
assert unpack_int4(p, q.shape).dtype == torch.int8
assert torch.equal(unpack_int4(p, q.shape), q)
q8 = torch.cat((q, torch.tensor([4], dtype=torch.int8)))
p8 = pack_int4(q8)
assert p8.tolist() == [112, 152, 175, 203]
assert p8.numel() * p8.element_size() == 4
assert torch.equal(unpack_int4(p8, q8.shape), q8)
print('EXAMPLE', json.dumps({'q': q.tolist(), 'q_bytes': 7, 'packed': p.tolist(),
      'packed_bytes': 4, 'packed_dtype': str(p.dtype), 'restored_dtype': 'torch.int8',
      'padding_code': int(p[-1] >> 4), 'eight_values': q8.tolist(), 'packed8': p8.tolist()}))

pairs = torch.tensor(list(itertools.product(range(-8, 8), repeat=2)), dtype=torch.int8)
packed_pairs = pack_int4(pairs)
expected = [(lo + 8) + 16 * (hi + 8) for lo, hi in pairs.tolist()]
assert packed_pairs.tolist() == expected
assert len(set(expected)) == 256 and min(expected) == 0 and max(expected) == 255
assert torch.equal(unpack_int4(packed_pairs, pairs.shape), pairs)
for byte, pair in zip(expected, pairs.tolist()):
    assert (byte & 15) - 8 == pair[0] and (byte >> 4) - 8 == pair[1]
print('EXHAUSTIVE', json.dumps({'ordered_pairs': 256, 'distinct_bytes': 256,
      'byte_min': 0, 'byte_max': 255, 'formula': '(lo+8)+16*(hi+8)', 'exact_roundtrip': True}))

cases = [torch.tensor([], dtype=torch.int8), torch.tensor(-8, dtype=torch.int8),
         torch.tensor([7], dtype=torch.int8), pairs[:7],
         torch.tensor([[-8, -1, 0], [1, 7, 2], [3, 4, 5]], dtype=torch.int8),
         torch.tensor([[-8, -1, 0], [1, 7, 2]], dtype=torch.int8).T]
for case in cases:
    packed = pack_int4(case)
    restored = unpack_int4(packed, case.shape)
    assert restored.shape == case.shape and restored.dtype == case.dtype
    assert torch.equal(case, restored)
    assert packed.numel() == (case.numel() + 1) // 2
    print('SHAPE', json.dumps({'shape': list(case.shape), 'numel': case.numel(),
          'contiguous': case.is_contiguous(), 'bytes': packed.numel(), 'exact_roundtrip': True}))
for invalid in [-9, 8]:
    try:
        pack_int4(torch.tensor([invalid], dtype=torch.int8))
    except ValueError:
        print('REJECT', invalid)
    else:
        raise AssertionError('out of range accepted')

w = torch.tensor([[-1.0, 0.22, 1.0]], dtype=torch.float32)
qi, s = quantize_symmetric(w, bits=4, per_channel=True)
recovered = unpack_int4(pack_int4(qi), qi.shape)
assert qi.min().item() == -7 and qi.max().item() == 7 and not (qi == -8).any()
assert torch.equal(qi, recovered)
assert not torch.equal(w, recovered * s)
print('LOSS', json.dumps({'float_input': w.tolist(), 'integers': qi.tolist(),
      'scale': s.tolist(), 'dequantized': (recovered * s).tolist(),
      'max_float_error': (w - recovered * s).abs().max().item(), 'packing_integer_loss': 0}))

layer = nn.Linear(3, 3, bias=True)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[-1., .22, 1.], [0., .5, 1.], [-.1, -.2, .7]]))
    layer.bias.copy_(torch.tensor([.1, .2, .3]))
compressed = QuantizedLinear(layer, bits=4)
x = torch.tensor([[1., 2., 3.]])
decoded = unpack_int4(compressed.values, compressed.shape)
reference = nn.functional.linear(x, decoded * compressed.scale, compressed.bias)
assert torch.equal(compressed(x), reference)
assert compressed.storage_bytes() == 5 + 12 + 12
assert list(compressed.shape) == [3, 3] and compressed.bits == 4
assert compressed.scale.shape == (3, 1)
print('LAYER', json.dumps({'shape': compressed.shape, 'bits': compressed.bits,
      'value_shape': list(compressed.values.shape), 'scale_shape': list(compressed.scale.shape),
      'bias_shape': list(compressed.bias.shape), 'storage_bytes': compressed.storage_bytes(),
      'buffer_names': list(dict(compressed.named_buffers())), 'output': compressed(x).tolist(),
      'ordinary_dequantized_linear_equal': True}))

storage = raw['results']['runs']['packed4']['storage']
groups = {suffix: sum(v for name, v in storage['buffers'].items() if name.endswith('.'+suffix))
          for suffix in ['values', 'scale', 'bias']}
assert groups == {'values': 57600, 'scale': 5664, 'bias': 2560}
assert len(storage['packed_linear_modules']) == 13
assert len({name.rsplit('.', 1)[0] for name in storage['buffers']}) == 13
assert sum(groups.values()) == sum(storage['buffers'].values()) == storage['buffer_tensor_bytes'] == 65824
assert storage['float_parameter_count'] * 4 == storage['parameter_tensor_bytes'] == 102912
assert storage['tensor_bytes'] == 102912 + 65824 == 168736
assert groups['values'] * 2 + groups['bias'] // 4 + storage['float_parameter_count'] == storage['parameter_count'] == 141568
print('RAW_STORAGE', json.dumps({'raw_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
      'modules': 13, 'buffer_bytes': groups, 'buffer_total': 65824, 'float_parameter_bytes': 102912,
      'tensor_total': 168736, 'scale_elements': 1416, 'bias_elements': 640,
      'weight_elements': 115200, 'original_revision': raw['revision']}, sort_keys=True))

# Reconstruct shapes from the recorded original architecture only. Random fresh
# weights do not reproduce scores and are not saved. Original counting function
# is selected by AST instead of importing the training entry point.
config = raw['results']['teacher_provenance']['config']
model = TinyLM(ModelConfig(**config))
replace_linear_layers(model, bits=4)
tree = ast.parse((BASE/'code/compression-original.py').read_bytes())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_storage')
ns = {'math': math, 'Path': Path, 'QuantizedLinear': QuantizedLinear}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'compression-original.py:_storage', 'exec'), ns)
with tempfile.TemporaryDirectory(prefix='factual-17_8-count-') as temporary:
    placeholder = Path(temporary) / 'size-placeholder.bin'
    placeholder.write_bytes(b'')
    counted = ns['_storage'](model, placeholder)
for field in ['buffers', 'packed_linear_modules', 'retained_float_modules',
              'parameter_count', 'float_parameter_count', 'parameter_tensor_bytes',
              'buffer_tensor_bytes', 'tensor_bytes']:
    assert counted[field] == storage[field], field
print('ARCHITECTURE_COUNT', json.dumps({'original_config': config,
      'all_buffer_entries_equal_raw': True, 'all_storage_tensor_counts_equal_raw': True,
      'model_forward_executed': False, 'saved_neural_weights': False,
      'shapes': {name: {'weight': module.shape, 'values': list(module.values.shape),
                       'scale': list(module.scale.shape), 'bias': None if module.bias is None else list(module.bias.shape)}
                 for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}}, sort_keys=True))
print('ALL_CHECKS_PASSED')
