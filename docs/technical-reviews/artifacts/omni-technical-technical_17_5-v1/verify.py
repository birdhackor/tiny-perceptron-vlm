"""17.5 凍結教材的有限 CPU 核對；無訓練，不宣稱品質或速度。"""
import ast
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import pathlib
import platform
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.quantization import quantize_symmetric

SNAPSHOT = ROOT / 'docs/course-review-20261019/freeze-01/sources/17.5.md'
text = SNAPSHOT.read_text()
blocks = re.findall(r'```python\n(.*?)```', text, re.S)
assert len(blocks) == 2
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(blocks[0], str(SNAPSHOT) + ':block1', 'exec'), namespace)
inline_namespace = {'torch': torch}
exec(compile(blocks[1], str(SNAPSHOT) + ':block2', 'exec'), inline_namespace)
inline_quantize = inline_namespace['quantize_symmetric']
inline_tree = ast.dump(ast.parse(blocks[1]), include_attributes=False)
repo_tree = ast.dump(ast.parse(inspect.getsource(quantize_symmetric)), include_attributes=False)
assert inline_tree == repo_tree
w = namespace['w']
observations = []
for per_channel in (False, True):
    q, scale = quantize_symmetric(w, bits=4, per_channel=per_channel)
    q_inline, scale_inline = inline_quantize(w, bits=4, per_channel=per_channel)
    assert torch.equal(q, q_inline) and torch.equal(scale, scale_inline)
    restored = q.float() * scale
    mae = (w - restored).abs().mean(-1)
    expected_q = [[0, 0, 0, 0], [1, 3, 4, 7]] if not per_channel else [[1, 3, 4, 7], [1, 3, 4, 7]]
    expected_mae = torch.tensor([0.0425, 2.5]) if not per_channel else torch.tensor([0.0025, 2.5])
    assert q.tolist() == expected_q
    assert torch.allclose(mae, expected_mae, atol=1e-6, rtol=0)
    observations.append({'per_channel': per_channel, 'q': q.tolist(), 'q_dtype': str(q.dtype), 'scale': scale.tolist(), 'scale_shape': list(scale.shape), 'scale_dtype': str(scale.dtype), 'scale_numel': scale.numel(), 'scale_bytes': scale.numel()*scale.element_size(), 'restored': restored.tolist(), 'row_mae': mae.tolist()})
exercise = torch.tensor([[10.0, 30.0, 50.0, 80.0], [10.0, 30.0, 50.0, 80.0]])
exercise_results = []
for pc in (False, True):
    q, scale = inline_quantize(exercise, bits=4, per_channel=pc)
    mae = (exercise-q.float()*scale).abs().mean(-1)
    exercise_results.append({'per_channel': pc, 'q': q.tolist(), 'row_mae': mae.tolist(), 'scale_numel': scale.numel()})
assert exercise_results[0]['q'] == exercise_results[1]['q']
assert exercise_results[0]['row_mae'] == exercise_results[1]['row_mae']
assert [r['scale_numel'] for r in exercise_results] == [1, 2]
edge_results = []
for bits in (4, 8):
    for pc in (False, True):
        values = torch.tensor([[-80., -30., 0., 80.], [0., 0., 0., 0.]], dtype=torch.float32)
        q, scale = inline_quantize(values, bits=bits, per_channel=pc)
        limit = 2**(bits-1)-1
        assert q.min() >= -limit and q.max() <= limit
        assert torch.isfinite(scale).all() and (scale > 0).all()
        assert torch.isfinite(q.float()*scale).all()
        edge_results.append({'bits': bits, 'per_channel': pc, 'q': q.tolist(), 'scale': scale.tolist(), 'scale_shape': list(scale.shape), 'finite': True})
try:
    inline_quantize(w, bits=3)
except ValueError as error:
    invalid_bits = {'type': type(error).__name__, 'message': str(error)}
else:
    raise AssertionError('bits=3 did not raise')
result = {
    'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_17_5-v1/verify.py',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'device': 'cpu', 'default_dtype': str(torch.get_default_dtype()), 'platform': platform.platform()},
    'source_snapshot': {'path': str(SNAPSHOT.relative_to(ROOT)), 'sha256': hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()},
    'implementation': {'path': 'tiny_perceptron/quantization.py', 'sha256': hashlib.sha256((ROOT/'tiny_perceptron/quantization.py').read_bytes()).hexdigest(), 'inline_function_ast_matches_repository': True},
    'embedded_block1_actual_stdout': stdout.getvalue(),
    'original_example': observations,
    'exercise_identical_rows': exercise_results,
    'edge_cases_fp32': edge_results,
    'invalid_bits': invalid_bits,
    'storage_derivation': {'float32_scale_bytes': torch.tensor(0., dtype=torch.float32).element_size(), 'two_row_scale_bytes': 2*4, 'one_scale_per_weight_packed4_bytes': 4+4/8, 'fp32_original_bytes_per_weight': 4, 'one_scale_per_weight_int8_container_bytes': 4+1},
    'limits': '只核本節 FP32、有限張量、CPU、4/8-bit 示意。分組誤差不是每個元素或每組大小都單調改善；未量模型品質、GPU速度、正式壓縮率或新訓練。',
    'result': 'all assertions passed'
}
output = pathlib.Path(__file__).with_name('execution.json')
with output.open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
