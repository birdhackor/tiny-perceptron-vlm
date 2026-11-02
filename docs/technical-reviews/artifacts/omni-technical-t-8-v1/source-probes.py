"""Small CPU/source checks for T.8; no formal GPU recipes are executed."""

import ast
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_experiments.architecture import run_flash_probe, run_efficiency, run_precision
from scripts.course_experiments.common import Context

ROOT = Path('/workspace/tiny-perceptron-vlm')
CACHE = Path('/workspace/work/reference/omni-technical-t-8-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-technical-t-8-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    directory = CACHE / 'flash-cpu-fixture'
    directory.mkdir(exist_ok=True)
    ctx = Context('cpu', directory, CACHE / 'absent-dependencies', ROOT / 'assets/training')
    assert not ctx.dependencies.exists()
    probes = {'created_at': datetime.now(UTC).isoformat(),
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
              'dependencies': {}}
    for name, fn in [('efficiency', run_efficiency), ('precision', run_precision)]:
        try:
            fn(ctx)
        except FileNotFoundError as error:
            assert 'sft/dataset.json' in str(error)
            probes['dependencies'][name] = {'expected_failure': str(error), 'training_executed': False}
        else:
            raise AssertionError('Missing prerequisite was not rejected')
    report = run_flash_probe(ctx)
    assert report['status'] == 'not_run' and report['verification_passed'] is False
    assert report['schedule_completed'] is False and report['routes'] == {}
    probes['flash_cpu'] = {key: report[key] for key in ['status', 'verification_passed', 'schedule_completed', 'reason', 'configuration']}
    fixture = directory / 'fixture.pt'
    probes['flash_cpu']['fixture'] = {'path': str(fixture), 'sha256': sha(fixture), 'bytes': fixture.stat().st_size}

    # Execute the exact finalization AST from the reviewed function on synthetic
    # status dictionaries. This checks Boolean semantics, not CUDA performance.
    code = ROOT / 'scripts/course_experiments/architecture.py'
    tree = ast.parse(code.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'run_flash_probe')
    start = next(i for i, x in enumerate(fn.body) if isinstance(x, ast.Assign)
                 and isinstance(x.targets[0], ast.Subscript)
                 and isinstance(x.targets[0].slice, ast.Constant)
                 and x.targets[0].slice.value == 'schedule_completed')
    fragment = ast.Module(body=fn.body[start:start + 3], type_ignores=[])
    compiled = compile(ast.fix_missing_locations(fragment), str(code), 'exec')
    probes['flash_status_semantics'] = []
    cases = [({'fp16': 'completed', 'bf16': 'runtime_error'}, ['fp16'], True, False, 'failed_verification'),
             ({'fp16': 'completed', 'bf16': 'unsupported'}, ['fp16'], True, True, 'completed_with_unsupported_routes'),
             ({'fp16': 'unsupported', 'bf16': 'unsupported'}, [], False, True, 'unsupported'),
             ({'fp16': 'completed', 'bf16': 'budget_exhausted'}, ['fp16', 'bf16'], False, False, 'budget_exhausted')]
    for statuses, supported, verified, completed, status in cases:
        obj = {'routes': {k: {'status': v} for k, v in statuses.items()}, 'supported_routes': supported}
        exec(compiled, {'report': obj})
        assert (obj['verification_passed'], obj['schedule_completed'], obj['status']) == (verified, completed, status)
        probes['flash_status_semantics'].append(obj)
    probes['source_fragment'] = {'path': str(code.relative_to(ROOT)), 'source_sha256': sha(code),
                                 'locator': 'run_flash_probe final schedule/verification/status AST nodes',
                                 'ast_sha256': hashlib.sha256(ast.dump(fragment).encode()).hexdigest()}
    probes['checks_passed'] = True
    path = OUT / 'source-probe-result.json'
    assert not path.exists()
    path.write_text(json.dumps(probes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'checks_passed': True}))


if __name__ == '__main__':
    main()
