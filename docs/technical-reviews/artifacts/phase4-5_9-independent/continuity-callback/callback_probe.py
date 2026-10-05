"""One frozen original fence and dtype/unit check; no backward/training."""
import hashlib
import json
import sys
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = A.parents[4]
sys.path.insert(0, str(ROOT))
import torch

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.manual_seed(42)
code = (A / 'fence-1.py').read_bytes()
namespace = {'__name__': '__main__'}
exec(compile(code, str(A / 'fence-1.py'), 'exec'), namespace)
model = namespace['model']
storage = sum(p.numel() * p.element_size() for p in model.parameters())
count = sum(p.numel() for p in model.parameters())
dtype_names = sorted({str(p.dtype) for p in model.parameters()})
assert count == 6104 and storage == 24416 and dtype_names == ['torch.float32']
assert isinstance(torch.float32, torch.dtype)
assert torch.finfo(torch.float32).bits == 32
assert torch.tensor([1.0], dtype=torch.float32).element_size() == 4
assert (256 - 1).bit_length() == 8 and len(bytes(range(256))) == 2**8
assert 32 // 8 == 4
with torch.no_grad():
    value = model(torch.tensor([[1, 2, 3]]))['logits']
assert not value.requires_grad
assert all(p.grad is None for p in model.parameters())
result = {'environment': {'python': sys.version, 'torch': torch.__version__,
    'torch_git_version': torch.version.git_version, 'device': 'cpu', 'threads': 1,
    'CUDA_build': str(torch.version.cuda)}, 'source_sha256': hashlib.sha256((A / 'section-current.md').read_bytes()).hexdigest(),
    'fence_sha256': hashlib.sha256(code).hexdigest(), 'fence_executed': True,
    'parameters': count, 'raw_FP32_parameter_bytes': storage, 'parameter_dtype_names': dtype_names,
    'float32_is_torch_dtype': True, 'float32_bits': torch.finfo(torch.float32).bits,
    'float32_element_size_bytes': 4, 'one_byte_possible_values': len(bytes(range(256))),
    'bits_per_byte': (256 - 1).bit_length(), 'bits_to_bytes_arithmetic': '32/8=4',
    'warmup_forwards': 1, 'timed_forwards': 10,
    'seconds_per_forward': namespace['average_seconds'], 'no_grad_logits_require_grad': False,
    'backward_calls': 0, 'optimizer_updates': 0,
    'new_claim_scope': 'byte/bit/FP32/dtype terminology only; gradient norm/clipping are not introduced or executed by 5.9.'}
(A / 'callback-probe-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
