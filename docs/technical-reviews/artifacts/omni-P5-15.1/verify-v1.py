"""15.1 review: only artificial inline CPU arithmetic and retained-result recomputation.
No pretrained/final model, checkpoint, training, CUDA, Modal, workflow or publishing.
"""
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json'
PACKET = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v6.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

freeze = json.loads(FREEZE.read_text())
page = next(x for x in freeze['pages'] if x['page_id'] == '15.1')
source = ROOT / page['snapshot']
body = source.read_text()
assert sha(FREEZE) == 'ab1bd5df0dc294cfa5948a22a601d35fbc6389ae48ef6e141bc1df31515dfc35'
assert sha(PACKET) == '4f6af6d8b67565c5c994606c14764662f43ca64c40c4b35f913e8d068551e4fc'
assert sha(source) == page['source_sha256']
raw = (ROOT / page['source']).read_text()
current = re.search(r'^## 15\.1 .*?(?=^## |\Z)', raw, re.M | re.S).group(0)
assert current == body
assert page['figures_sha256'] == {}
assert not re.findall(r'!\[[^\]]*\]\([^)]*\)', body)
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 1
inline = blocks[0]
notebook = ROOT / page['notebook']
assert sha(notebook) == page['notebook_sha256']
notebook_cells = [''.join(c['source']) for c in json.loads(notebook.read_text())['cells'] if c['cell_type'] == 'code']
assert any(c.rstrip() == inline.rstrip() for c in notebook_cells)

torch.set_num_threads(1)
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(inline, str(source) + ':inline-python', 'exec'), namespace)
x, up, down, ffn = [namespace[k] for k in ('x', 'up', 'down', 'ffn')]
assert up.dtype == torch.float32 and down.dtype == torch.float32
assert not up.requires_grad and not down.requires_grad
expected = [[4.0, 5.0], [5.0, 4.0]]
assert ffn(x).tolist() == expected
assert ffn(torch.cat([x, x], dim=0)).tolist() == expected * 2
assert ffn(torch.cat([x, x, x], dim=0)).tolist() == expected * 3
negative = x.clone()
negative[0] = torch.tensor([1.0, -2.0])
assert (negative @ up)[0].tolist() == [1.0, -2.0, -1.0]
assert torch.relu(negative @ up)[0].tolist() == [1.0, 0.0, 0.0]
assert ffn(negative)[0].tolist() == [1.0, 0.0]
assert up.numel() + down.numel() == 12
linear_up = torch.nn.Linear(2, 3)
linear_down = torch.nn.Linear(3, 2)
with torch.no_grad():
    linear_up.weight.copy_(up.T)
    linear_down.weight.copy_(down.T)
    linear_up.bias.zero_()
    linear_down.bias.zero_()
assert torch.equal(linear_down(torch.relu(linear_up(x))), ffn(x))
assert sum(p.numel() for layer in (linear_up, linear_down) for p in layer.parameters()) == 17

result_path = ROOT / 'docs/course-experiments/results/moe.json'
record = json.loads(result_path.read_text())
v = record['results']['variants']['dense_active_top1']
c = v['model']['config']
assert (c['width'], c['layers'], c['vocab_size'], c['max_length'], c['activation']) == (64, 2, 264, 128, 'gelu')
ffn_parameters = 64 * 256 + 256 + 256 * 64 + 64
parts = {'token_embedding': 264 * 64, 'position_embedding': 128 * 64,
         'two_blocks_norms': 2 * 2 * 2 * 64, 'two_blocks_attention': 2 * 4 * 64 * 64,
         'two_blocks_ffn': 2 * ffn_parameters, 'final_norm': 2 * 64, 'output': 264 * 64}
assert ffn_parameters == 33088
assert sum(parts.values()) == v['model']['parameters'] == 141568
tr = v['training']
assert (tr['requested_steps'], tr['steps'], tr['optimizer_updates'], tr['skipped_updates']) == (180, 180, 180, 0)
recomputed = {}
for split, expected_rounded in [('validation', '2.29999'), ('test', '2.32093')]:
    retained = v['heldout'][split]
    ratio = retained['nll_sum'] / retained['effective_tokens']
    assert ratio == retained['nll']
    assert f'{ratio:.5f}' == expected_rounded
    assert retained['records'] == record['results']['dataset'][split]['records']
    recomputed[split] = {k: retained[k] for k in ('records', 'examples', 'effective_tokens', 'nll_sum', 'nll')}
    recomputed[split]['recomputed_nll'] = ratio
    recomputed[split]['rounded_5dp'] = f'{ratio:.5f}'

# Read recorded experiment code from its real Git object, without running it.
revision = record['revision']
recorded_arch = subprocess.check_output(['git', 'show', f'{revision}:scripts/course_experiments/architecture.py'], cwd=ROOT).decode()
assert hashlib.sha256(recorded_arch.encode()).hexdigest() == record['code_sha256']['scripts/course_experiments/architecture.py']
current_arch = (ROOT / 'scripts/course_experiments/architecture.py').read_text()
function_equality = {}
for name in ('_text_dataset', '_train', '_nll', '_steps', '_heldout', 'run_moe'):
    def definition(src):
        return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    function_equality[name] = ast.dump(definition(recorded_arch)) == ast.dump(definition(current_arch))
assert all(function_equality.values())
for name in ('tiny_perceptron/model.py', 'tiny_perceptron/modern.py', 'tiny_perceptron/attention.py', 'scripts/course_experiments/common.py'):
    assert sha(ROOT / name) == record['code_sha256'][name]
plan = json.loads((ROOT / 'docs/course-experiments/plan.json').read_text())
recipe = next(p for p in plan['sequence'] + plan.get('supporting_experiments', []) if p['id'] == 'moe')
assert recipe['module'] == 'architecture' and recipe['function'] == 'run_moe'
output = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': '/root/omni_p5_technical_15_1',
    'scope': 'Exact frozen inline CPU demo and exercises; pure retained JSON arithmetic; static recipe/code identity. No formal model inference or training.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'platform': platform.platform()},
    'source_sha256': sha(source), 'notebook_sha256': sha(notebook), 'inline_sha256': hashlib.sha256(inline.encode()).hexdigest(),
    'inline_stdout': stdout.getvalue(),
    'exercise_six_rows': ffn(torch.cat([x, x, x], dim=0)).tolist(),
    'negative_exercise': {'up': (negative @ up)[0].tolist(), 'after_relu': torch.relu(negative @ up)[0].tolist(), 'output': ffn(negative)[0].tolist()},
    'toy_weight_count': 12, 'nn_linear_toy_count_with_bias': 17,
    'ffn_parameters': ffn_parameters, 'model_parameter_parts': parts, 'model_parameters': sum(parts.values()),
    'result_source': {'path': str(result_path.relative_to(ROOT)), 'sha256': sha(result_path), 'revision': revision, 'seed': record['seed'], 'recorded_device': record['device'], 'recorded_gpu': record['gpu']},
    'training_record': {k: tr[k] for k in ('requested_steps', 'steps', 'optimizer_updates', 'skipped_updates', 'batch_size', 'effective_tokens')},
    'heldout_recomputation': recomputed, 'recorded_vs_current_function_ast_equal': function_equality,
    'recipe_route': {k:recipe[k] for k in ('id', 'module', 'function')},
    'all_assertions_passed': True,
    'limitations': 'Stored sums and update logs were inspected, not reproduced with weights; no GPU, MPS, Colab or live-site rendering check.'
}
print(json.dumps(output, ensure_ascii=False, indent=2))
