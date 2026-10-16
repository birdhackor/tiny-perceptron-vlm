"""Independent bounded CPU checks for frozen lesson 19.19; no training/test run."""

import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

ROOT = Path('/workspace/tiny-perceptron-vlm')
WORK = Path('/workspace/work/curriculum-20261016/technical-19.19')
OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/backfill/19.19-execution.json'
PYTHON = str(ROOT / '.venv/bin/python')
STATE = '兩盒追蹤;A=2;B=6;事件=交換;查詢=B'
result = {
    'reviewer_task': '/root/backfill_tech_19_19',
    'scope': 'Fresh anonymous download, bounded CPU CLI/offline/HTTP/manifest checks and read-only checkpoint inspection; no training or formal test execution.',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
    'runs': [],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, offline=False):
    if offline:
        wrapper = """import runpy,socket,sys,urllib.request
attempts=[]
def denied(*args,**kwargs):
    attempts.append(str(args[0]) if args else 'network')
    raise RuntimeError('19.19 independent offline guard: network disabled')
urllib.request.urlopen=denied
socket.create_connection=denied
socket.socket.connect=denied
sys.argv=['tiny_perceptron.lesson_products']+sys.argv[1:]
try:
    runpy.run_module('tiny_perceptron.lesson_products',run_name='__main__')
finally:
    print('OFFLINE_NETWORK_ATTEMPTS='+str(len(attempts)),file=sys.stderr)
"""
        command = [PYTHON, '-c', wrapper, *args]
    else:
        command = [PYTHON, '-m', 'tiny_perceptron.lesson_products', *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=55)
    entry = {'command': command, 'cwd': str(ROOT), 'exit_code': completed.returncode, 'stdout': completed.stdout, 'stderr': completed.stderr, 'offline_guard': offline}
    result['runs'].append(entry)
    return completed


WORK.mkdir(parents=True, exist_ok=True)
fresh = WORK / 'fresh-products'
if fresh.exists():
    raise RuntimeError('Fresh download destination already exists; preserve prior evidence')
fresh.mkdir()
result['empty_starting_directory'] = not any(fresh.iterdir())
for product in ('b', 'c'):
    completed = run(['download', '--product', product, '--destination', str(fresh / product)])
    assert completed.returncode == 0, completed.stderr

b_args = ['--b', str(fresh / 'b')]
c_args = ['--c', str(fresh / 'c')]
for args in (
    ['solve', *b_args, '--state', STATE, '--loops', '3'],
    ['judge', *c_args, '--state', '卡片進度;申請編號=?', '--question', 'info'],
    ['auto', *b_args, *c_args, '--state', STATE, '--effort', 'auto'],
    ['auto', *b_args, *c_args, '--state', STATE, '--effort', 'medium', '--max-loops', '2'],
    ['judge', *c_args, '--state', '卡片進度;申請編號=?', '--question', 'info', '--candidates', 'no,yes'],
):
    completed = run(args, offline=True)
    assert completed.returncode == 0, completed.stderr
    assert 'OFFLINE_NETWORK_ATTEMPTS=0' in completed.stderr
result['b_effort_parser_probe'] = run(['solve', *b_args, '--state', STATE, '--effort', 'medium']).returncode

from tiny_perceptron.lesson_products.inference import Product

torch.set_num_threads(2)
result['loaded_products'] = {}
for product in ('b', 'c'):
    obj = Product(fresh / product, product)
    result['loaded_products'][product] = {
        'manifest_sha256': sha(fresh / product / 'artifact-manifest.json'),
        'verified_files': len(obj.manifest['files']),
        'config': vars(obj.config),
        'tokenizer_size': len(obj.tokenizer),
        'all_modules_eval': all(not module.training for module in obj.model.modules()),
        'weight_requires_grad': any(p.requires_grad for p in obj.model.parameters()),
        'public_model_keys': list(torch.load(fresh / product / 'model.pt', map_location='cpu', weights_only=True)),
    }
    full = ROOT / 'outputs/lesson-products/products' / product
    receipt = json.loads((full / 'training-receipt.json').read_bytes())
    last = torch.load(full / 'last.pt', map_location='cpu', weights_only=True)
    result['loaded_products'][product]['training_receipt'] = {
        k: receipt[k] for k in ('random_initialization', 'seed', 'last_step', 'selected_step', 'data_manifest_sha256', 'initial_weights_sha256', 'selected_weights_sha256', 'environment')
    }
    result['loaded_products'][product]['full_checkpoint'] = {
        'keys': list(last),
        'optimizer_state_items': len(last['optimizer']['state']),
        'sampler_rng_bytes': last['sampler_rng'].numel(),
        'torch_rng_bytes': last['torch_rng'].numel(),
        'cuda_rng_devices': len(last['cuda_rng']),
        'metadata': last['metadata'],
    }

tampered = WORK / 'tampered-b'
shutil.copytree(fresh / 'b', tampered)
(tampered / 'auto-policy.json').write_text('{}\n')
try:
    Product(tampered, 'b')
except ValueError as error:
    result['tampered_policy_rejection'] = str(error)
else:
    raise AssertionError('Modified policy passed manifest validation')

with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
command = [PYTHON, '-m', 'tiny_perceptron.lesson_products', 'serve', *b_args, *c_args, '--port', str(port)]
server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
result['http'] = {'command': command, 'host': '127.0.0.1', 'requests': []}
try:
    for _ in range(100):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                result['http']['health'] = json.load(response)
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError('Local HTTP server did not become ready')
    for endpoint, payload in (
        ('solve', {'state': STATE, 'loops': 3}),
        ('judge', {'state': '卡片進度;申請編號=?', 'question': 'info'}),
        ('auto', {'state': STATE, 'effort': 'medium', 'max_loops': 2}),
    ):
        request = urllib.request.Request(f'http://127.0.0.1:{port}/{endpoint}', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            result['http']['requests'].append({'endpoint': endpoint, 'payload': payload, 'status': response.status, 'output': json.load(response)})
finally:
    server.terminate()
    stdout, stderr = server.communicate(timeout=10)
    result['http']['stdout'] = stdout
    result['http']['stderr'] = stderr

result['source_sha256'] = {
    str(path.relative_to(ROOT)): sha(path)
    for path in (
        ROOT / 'tiny_perceptron/lesson_products/release.json',
        ROOT / 'tiny_perceptron/lesson_products/delivery.py',
        ROOT / 'tiny_perceptron/lesson_products/__main__.py',
        ROOT / 'tiny_perceptron/lesson_products/inference.py',
        ROOT / 'tiny_perceptron/lesson_products/model.py',
        ROOT / 'tiny_perceptron/lesson_products/train.py',
        ROOT / 'tiny_perceptron/lesson_products/training.py',
        ROOT / 'tiny_perceptron/lesson_products/data.py',
        ROOT / 'scripts/lesson_products/train_local.py',
        ROOT / 'docs/lesson-products/README.md',
        ROOT / 'docs/lesson-products/results/public-cpu-acceptance-v1.json',
        ROOT / 'docs/selftrained/v2-public-cpu-commands.md',
    )
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'artifact': str(OUT.relative_to(ROOT)), 'runs': len(result['runs']), 'offline_runs': sum(r['offline_guard'] for r in result['runs']), 'http_requests': len(result['http']['requests']), 'b_effort_parser_exit': result['b_effort_parser_probe']}, ensure_ascii=False))
