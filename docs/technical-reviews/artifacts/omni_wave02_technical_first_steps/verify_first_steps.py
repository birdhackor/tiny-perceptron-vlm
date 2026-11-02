"""Fresh, offline CPU checks of the frozen canonical first-steps page.

This review harness does not change the source, register a persistent kernel,
download training data, or perform any optimizer update.
"""

import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
EXTERNAL = Path('/workspace/.review-artifacts/omni_wave02_technical_first_steps')
SOURCE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07/sources/first-steps.md'
torch.set_num_threads(1)
raw = SOURCE.read_text()
sections = {
    m[1]: m[0]
    for m in re.finditer(r'^## (W\.\d+) .*?(?=^## |\Z)', raw, re.M | re.S)
}
blocks = {k: re.findall(r'```python\n(.*?)```', v, re.S) for k, v in sections.items()}
records = []


def emit(check, **details):
    item = {'check': check, **details}
    records.append(item)
    print(json.dumps(item, ensure_ascii=False))


def execute(code, scope=None):
    scope = {} if scope is None else scope
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, str(SOURCE), 'exec'), scope)
    return scope, out.getvalue()


def run(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=35)
    if result.returncode:
        raise AssertionError((args, result.returncode, result.stdout, result.stderr))
    return result.stdout, result.stderr


emit('environment', python=platform.python_version(), torch=torch.__version__,
     platform=platform.platform(), device='cpu', executable=sys.executable,
     source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
     started_at=datetime.now(timezone.utc).isoformat())

# Run every Python fence exactly, preserving only each section's stated state.
for section in ('W.2', 'W.3', 'W.4', 'W.5', 'W.6', 'W.7'):
    state = {}
    for index, code in enumerate(blocks[section], 1):
        state, stdout = execute(code, state)
        emit('exact_fence', section=section, block=index, stdout=stdout)

state, out = execute(blocks['W.2'][0])
assert out == '貓 2\n貓在睡覺\n看到 貓\n看到 狗\n'
state, out = execute(blocks['W.2'][0].replace('["貓", "狗"]', '["鳥", "貓", "狗"]').replace('== 2', '== 3'))
assert out == '鳥 3\n貓在睡覺\n看到 鳥\n看到 貓\n看到 狗\n'
emit('W2_list_exercise', stdout=out)
_, caption_out = execute(blocks['W.2'][0].replace('"caption": "貓在睡覺"', '"caption": "貓在走路"'))
assert caption_out == '貓 2\n貓在走路\n看到 貓\n看到 狗\n'
emit('W2_caption_independence', stdout=caption_out)

table = [
    ('[5, 2, 7][:-1]', [5, 2]), ('[5, 2, 7][1:]', [2, 7]),
    ('list(range(3))', [0, 1, 2]), ('len(set("貓狗貓"))', 2),
    ('sorted([3, 1, 2])', [1, 2, 3]), ('sorted([3, 1, 2], reverse=True)', [3, 2, 1]),
    ('sorted(set("貓狗貓"))', ['狗', '貓']), ('ord("狗")', 29399), ('ord("貓")', 35987),
    ('max(["貓", "小花貓"], key=len)', '小花貓'),
    ('max(range(3), key=lambda i: scores[i])', 1),
    ('list(zip(["貓", "狗"], [3, 1]))', [('貓', 3), ('狗', 1)]),
    ('dict(zip(["貓", "狗"], [0, 1]))', {'貓': 0, '狗': 1}),
    ('list(enumerate(["貓", "狗"]))', [(0, '貓'), (1, '狗')]),
    ('list(reversed(["貓", "狗", "鳥"]))', ['鳥', '狗', '貓']),
    ('"".join(["貓", "看", "狗"])', '貓看狗'), ('"、".join(["貓", "狗"])', '貓、狗'),
    ('[x * 2 for x in [1, 2, 3]]', [2, 4, 6]),
    ('sum(x * 2 for x in [1, 2, 3])', 12),
    ('next(x for x in [3, 8, 12] if x > 5)', 8),
    ('next((x for x in [3] if x > 5), None)', None),
    ('f"看到{animal}"', '看到貓'), ('7 // 2', 3), ('-7 // 2', -4),
]
table_scope = {'scores': [5, 9, 7], 'animal': '貓'}
for expression, expected in table:
    observed = eval(expression, table_scope)
    assert observed == expected, (expression, observed, expected)
    emit('W2_table', expression=expression, observed=observed)
_, function_out = execute('def first(items):\n    return items[0]\nprint(first(["貓", "狗"]))\n')
assert function_out == '貓\n'
_, auxiliary = execute(blocks['W.2'][1])
assert auxiliary == "16\n找不到 '狗'\n-1\n53 3 5\n"
emit('W2_controlflow_and_bitwise', stdout=auxiliary, first_function=function_out)

state, _ = execute(blocks['W.3'][0])
scores = state['scores']
assert tuple(scores.shape) == (2, 3)
assert scores[0].tolist() == [60, 70, 80]
assert scores[1, 2].item() == 100
assert scores.mean(dim=1).tolist() == [70, 90]
assert scores.mean(dim=0).tolist() == [70, 80, 90]
assert torch.equal(scores.mean(dim=-1), scores.mean(dim=1))
assert scores.reshape(6).tolist() == [60, 70, 80, 80, 90, 100]
assert torch.equal(scores.reshape(6).reshape(2, 3), scores)
try:
    scores.reshape(8)
except RuntimeError as error:
    emit('W3_wrong_reshape', exception=type(error).__name__, message=str(error))
else:
    raise AssertionError('reshape(8) should fail')
variant, out = execute(blocks['W.3'][0].replace('[60.0, 70.0, 80.0]', '[60.0, 70.0, 110.0]'))
assert variant['scores'].mean(dim=1).tolist() == [80, 90]
assert variant['scores'].mean(dim=0).tolist() == [70, 80, 105]
emit('W3_exercise', stdout=out, shape_tuple=list(tuple(scores.shape)),
     nested_list=scores.tolist(), reshaped=scores.reshape(6).tolist())

_, out = execute(blocks['W.4'][0])
assert out == '0.9 0.105\n0.5 0.693\n0.1 2.303\n'
_, variant_out = execute(blocks['W.4'][0].replace('[0.9, 0.5, 0.1]', '[0.9, 0.5, 0.1, 0.25]'))
assert variant_out.endswith('0.25 1.386\n')
try:
    math.log(0)
except ValueError as error:
    log_zero = type(error).__name__ + ': ' + str(error)
else:
    raise AssertionError('log(0) should fail')
ce = torch.nn.functional.cross_entropy(torch.tensor([[math.log(.9), math.log(.1)]]), torch.tensor([0]))
assert abs(ce.item() - (-math.log(.9))) < 1e-6
emit('W4_costs', stdout=variant_out, independent_joint_cost=-math.log(.5 * .5),
     sum_costs=-math.log(.5)-math.log(.5), log_zero=log_zero,
     cross_entropy_raw_scores=float(ce))

state, out1 = execute(blocks['W.5'][0])
assert state['result'].tolist() == [[40, 1], [80, 6]]
state, out2 = execute(blocks['W.5'][1], state)
assert state['layer'](state['recipes']).tolist() == [[45, 1], [85, 6]]
state, out3 = execute(blocks['W.5'][1].replace('[5.0, 0.0]', '[8.0, 0.0]'), state)
assert state['layer'](state['recipes']).tolist() == [[48, 1], [88, 6]]
assert tuple(state['layer'].weight.shape) == (2, 3)
emit('W5_linear_and_exercise', multiplication_stdout=out1, bias5_stdout=out2,
     bias8_stdout=out3, weight_shape=list(state['layer'].weight.shape))

for rate, expected_w, expected_loss in ((.1, 1.4, 2.56), (.6, 3.4, .16), (1.1, 5.4, 5.76)):
    state, out = execute(blocks['W.6'][0].replace('w -= 0.1 * w.grad', f'w -= {rate} * w.grad'))
    w = state['w']
    assert abs(w.item() - expected_w) < 1e-6
    assert abs((w-3).square().item() - expected_loss) < 2e-6
    emit('W6_learning_rates', learning_rate=rate, stdout=out,
         expected_w=expected_w, expected_loss=expected_loss, tolerance='2e-6')
w = torch.tensor(1., requires_grad=True)
(w-3).square().backward()
assert w.item() == 1 and w.grad.item() == -4
first_grad = w.grad.item()
(w-3).square().backward()
assert w.grad.item() == -8
emit('W6_backward_is_not_update', w_before_and_after_backward=w.item(),
     first_grad=first_grad, accumulated_grad=w.grad.item(),
     finite_difference_at_1=((1.000001-3)**2-(1-3)**2)/.000001)

root_out = execute(blocks['W.7'][0])[1]
previous = Path.cwd()
try:
    os.chdir(ROOT / 'notebooks')
    notebooks_out = execute(blocks['W.7'][0])[1]
finally:
    os.chdir(previous)
assert root_out.splitlines()[1:] == ['True', 'True']
assert notebooks_out.splitlines()[1:] == ['False', 'False']
emit('W7_cwd', root_stdout=root_out, notebooks_stdout=notebooks_out)
for label, operation, expected_exception in (
    ('missing_key', lambda: {'貓': 0}['狗'], KeyError),
    ('out_of_range', lambda: ['貓', '狗'][2], IndexError),
    ('missing_file', lambda: (EXTERNAL/'deliberately-absent').read_bytes(), FileNotFoundError),
    ('syntax', lambda: compile('for x in [1]\n    print(x)', '<example>', 'exec'), SyntaxError),
):
    try:
        operation()
    except expected_exception as error:
        emit('W7_exception', example=label, exception=type(error).__name__)
    else:
        raise AssertionError(label)

notebook = json.loads((ROOT / 'notebooks/01/1.1.ipynb').read_text())
setup = next(''.join(c['source']) for c in notebook['cells'] if c.get('metadata', {}).get('course_setup'))
try:
    os.chdir(ROOT / 'notebooks/01')
    state, setup_out = execute(setup)
finally:
    os.chdir(previous)
assert state['root'] == ROOT and state['in_colab'] is False
first_code = next(''.join(c['source']) for c in notebook['cells'] if 'text = "貓看狗，狗看貓。"' in ''.join(c['source']))
state, changed_out = execute(first_code.replace('貓看狗，狗看貓。', '鳥看狗，狗看鳥。'))
assert state['restored'] == '鳥看狗，狗看鳥。'
assert changed_out.splitlines()[-1] == '鳥看狗，狗看鳥。'
emit('W1_notebook_local_bootstrap_and_exercise', root=str(state.get('root', ROOT)),
     bootstrap_stdout=setup_out, changed_text_stdout=changed_out,
     bootstrap_sha256=hashlib.sha256(setup.encode()).hexdigest())

for args in ([sys.executable, 'scripts/check_env.py'], ['git', '--version'],
             ['uv', '--version'], [sys.executable, '-m', 'ipykernel', '--version'],
             [sys.executable, '-m', 'jupyterlab', '--version']):
    stdout, stderr = run(args)
    emit('W1_command', argv=args, stdout=stdout, stderr=stderr)

# Same registration mechanism, isolated prefix to avoid persistent shared state.
prefix = EXTERNAL / 'kernel-prefix'
stdout, stderr = run([sys.executable, '-m', 'ipykernel', 'install', '--prefix', str(prefix),
                      '--name', 'tiny-perceptron', '--display-name', 'Tiny Perceptron'])
spec = json.loads((prefix/'share/jupyter/kernels/tiny-perceptron/kernel.json').read_text())
assert spec['display_name'] == 'Tiny Perceptron' and spec['argv'][0] == sys.executable
emit('W1_kernel_registration', stdout=stdout, stderr=stderr, kernel_spec=spec,
     scope='Isolated --prefix; did not mutate the shared --sys-prefix registry')

# Verify the actual JupyterLab entrypoint and notebooks root, then shut it down.
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
log_path = EXTERNAL / 'jupyterlab-launch.log'
argv = [sys.executable, '-m', 'jupyterlab', 'notebooks', '--no-browser', '--allow-root',
        '--ServerApp.ip=127.0.0.1', f'--ServerApp.port={port}', '--ServerApp.port_retries=0',
        '--IdentityProvider.token=']
with log_path.open('w') as log:
    process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        contents = None
        for _ in range(60):
            if process.poll() is not None:
                raise AssertionError(log_path.read_text())
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/contents', timeout=.5) as response:
                    contents = json.load(response)
                break
            except (OSError, TimeoutError):
                time.sleep(.2)
        assert contents is not None
        names = [item['name'] for item in contents['content']]
        assert '01' in names
        emit('W1_jupyterlab_launch', argv=argv, root_listing_contains_01='01' in names,
             server_log_path=str(log_path), scope='Linux local server; no GUI interaction')
    finally:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

# The CLI default is a local gradient dry run, not training.
output = EXTERNAL / 'dry-run-must-not-exist.pt'
assert not output.exists()
argv = [sys.executable, 'scripts/train.py', '--device', 'cpu', '--output', str(output)]
stdout, stderr = run(argv)
report = json.loads(stdout.splitlines()[-1])
assert report['mode'] == 'dry-run-no-weight-update' and len(report['history']) == 1
assert not output.exists()
emit('W7_default_dry_run', argv=argv, stdout=stdout, stderr=stderr,
     checkpoint_created=output.exists(), scope='One CPU forward/backward; no --train and no optimizer step')
emit('finished', passed=True, checked_fences=sum(len(v) for v in blocks.values()),
     finished_at=datetime.now(timezone.utc).isoformat())
