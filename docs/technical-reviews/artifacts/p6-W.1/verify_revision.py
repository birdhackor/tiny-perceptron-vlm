"""Recheck only the W.1 success-criteria correction; never run GPU computations."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tomllib

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
initial = json.loads((OUT / 'initial-report.json').read_bytes())
version_checks = []
for source in initial['sources']:
    if source['kind'] == 'repository_code':
        observed = digest(ROOT / source['path'])
        assert observed == source['sha256'], (source['path'], observed)
        version_checks.append({'path': source['path'], 'sha256': observed, 'unchanged': True})
for artifact in initial['artifacts']:
    assert digest(ROOT / artifact['path']) == artifact['sha256'], artifact['path']
for name, expected in initial['figure_sha256'].items():
    assert digest(ROOT / name) == expected, name

raw = (ROOT / 'course/first-steps.md').read_bytes()
start = raw.index(b'## W.1 ')
end = raw.index(b'## W.2 ', start)
section, intro = raw[start:end], raw[:start]
old_section = (OUT / 'section-W.1.md').read_bytes()
old_sentence = ('看到 `compute` 那列為 `OK（cpu）`，且套件沒有「未安裝」或最後的紅色錯誤，'
                '才表示 CPU 計算檢查完成。')
new_sentence = ('看到 `compute` 那列為 `OK`（括號中的裝置名稱依電腦而異），且套件沒有'
                '「未安裝」或最後的紅色錯誤，就表示計算環境可以使用。')
assert old_sentence.encode() in old_section
assert new_sentence.encode() in section
# The only substantive edit is the success condition; a final blank line was also added.
assert old_section.decode().replace(old_sentence, new_sentence).rstrip() == section.decode().rstrip()
assert hashlib.sha256(intro).hexdigest() == initial['intro_sha256']

script = ROOT / 'scripts/check_env.py'
tree = ast.parse(script.read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
backward = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'backward')
compute = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == 'show'
               and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'compute')
assert backward < compute.lineno
assert isinstance(compute.args[1], ast.JoinedStr)
assert any(isinstance(value, ast.FormattedValue) and ast.unparse(value.value) == 'device.type'
           for value in compute.args[1].values)

spec = importlib.util.spec_from_file_location('revision_check_env', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
branches = []
for cuda_available, mps_available, expected in [(False, False, 'cpu'), (False, True, 'mps'),
                                               (True, True, 'cuda')]:
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available,
            get_device_properties=lambda number: SimpleNamespace(name='bounded mock CUDA',
                total_memory=2**30, major=8, minor=0),
            is_bf16_supported=lambda **kwargs: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        device=torch.device, get_num_threads=lambda: 1, version=SimpleNamespace(cuda=None))
    stream = io.StringIO()
    with patch.object(module.shutil, 'which', return_value=None), contextlib.redirect_stdout(stream):
        selected = module.pick_device(fake)
        assert selected.type == expected
        # This is the real script's output formatter; the GPU computation is not performed.
        module.show('compute', f'OK（{selected.type}）')
    branches.append({'cuda_available_mock': cuda_available, 'mps_available_mock': mps_available,
                     'selected_device': selected.type, 'formatter_output': stream.getvalue(),
                     'valid_under_revised_sentence': True, 'GPU_computation_measured': False})

lock = tomllib.loads((ROOT / 'uv.lock').read_text())
locked = [{'name': p['name'], 'version': p['version'], 'source': p['source']}
          for p in lock['package'] if p['name'] in {'torch', 'ipykernel', 'jupyterlab', 'nbclient', 'safetensors'}]
prior_workflow = json.loads((OUT / 'workflow-results.json').read_bytes())
assert len(prior_workflow['notebooks']) == 4
assert all(n['fresh_kernel'] and n['device'] == 'cpu' for n in prior_workflow['notebooks'])
assert prior_workflow['jupyter_server']['notebook_api_status'] == 200
assert prior_workflow['jupyter_server']['lab_status'] == 200
prior_cpu = (OUT / 'check-env.txt').read_text()
assert 'OK（cpu）' in prior_cpu and '未安裝！' not in prior_cpu and 'Traceback' not in prior_cpu

results = {'reviewer_task': '/root/p6_fact_w1', 'source_sha256': hashlib.sha256(section).hexdigest(),
    'intro_sha256': hashlib.sha256(intro).hexdigest(), 'only_substantive_edit': 'c11 success criterion',
    'unchanged_repository_sources': version_checks, 'prior_artifacts_rehashed': len(initial['artifacts']),
    'figure_unchanged': True, 'prior_cpu_workflow_reused': True, 'prior_notebook_kernel_count': 4,
    'new_kernel_or_figure_runs': 0, 'new_device_branch_checks': branches,
    'main_compute_print_follows_backward': True, 'main_compute_field': ast.unparse(compute.args[1]),
    'current_lock_checked': locked,
    'limitation': 'Device-availability flags are mocked; no native Apple/Windows or GPU numerical computation.'}
(OUT / 'revision-check-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
