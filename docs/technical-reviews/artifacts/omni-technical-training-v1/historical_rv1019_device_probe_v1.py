"""training groups/9 的有界 CPU 歷史覆核；不訓練、不下載、不覆寫既有證據。"""

import argparse
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[4]
FORMAL = Path(__file__).resolve().parent
SCRATCH = Path('/workspace/work/omni-technical-training-historical-rv1019-v1')
REQUEST = ROOT / 'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json'
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
TASK = '/root/omni_technical_coordinator/technical_training'
FIXTURE = Path('/workspace/work/omni-technical-training-v1/checkpoints/attributes.pt')
OWN_REPORTS = [
    ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_training/technical-original-v1.json',
    ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_training/technical-current-v1.json',
]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(UTC).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': sha(path.read_bytes())}


def read_ref(ref):
    raw = (ROOT / ref['path']).read_bytes()
    assert sha(raw) == ref['sha256'], ref
    return json.loads(raw)


def node(value, pointer):
    for segment in pointer.split('/')[1:]:
        segment = segment.replace('~1', '/').replace('~0', '~')
        value = value[int(segment)] if isinstance(value, list) else value[segment]
    return value


def save_new(path, document):
    with path.open('x', encoding='utf-8') as out:
        out.write(json.dumps(document, ensure_ascii=False, indent=2) + '\n')


def run(label, argv, expected):
    before = now()
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True,
                          text=True, timeout=40, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    row = {'label': label, 'argv': [sys.executable, *argv], 'started_at': before,
           'completed_at': now(), 'exit_code': proc.returncode,
           'stdout': proc.stdout, 'stderr': proc.stderr, 'expected_exit_code': expected}
    assert proc.returncode == expected, row
    return row


def capture_defaults(module_name, argv):
    """實際 parser 解析後立刻停下，避免固定實驗或下載副作用。"""
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args
    captured = {}

    class ParsedOnly(Exception):
        pass

    def observe(parser, *args, **kwargs):
        result = original(parser, *args, **kwargs)
        captured.update({k: str(v) if isinstance(v, Path) else v for k, v in vars(result).items()})
        raise ParsedOnly()

    with patch.object(sys, 'argv', [module_name, *argv]), patch.object(argparse.ArgumentParser, 'parse_args', observe):
        try:
            module.main()
        except ParsedOnly:
            pass
        else:
            raise AssertionError('parser not intercepted: ' + module_name)
    return captured


def main():
    started = now()
    assert not SCRATCH.exists(), 'use a fresh scratch path'
    assert FIXTURE.is_file()
    own_before = [reference(p) for p in OWN_REPORTS]
    fixture_before = reference(FIXTURE)
    request = json.loads(REQUEST.read_bytes())
    group = request['groups'][9]
    assert group['page_id'] == 'training' and group['old_owner'] == '/root/rv1019_technical_training'
    assert group['old_role'] == 'technical' and len(group['items']) == 2
    frozen = json.loads(FREEZE.read_bytes())
    page = next(p for p in frozen['pages'] if p['page_id'] == 'training')
    current_raw = (ROOT / page['snapshot']).read_bytes()
    assert sha(current_raw) == page['source_sha256'] == sha((ROOT / page['source']).read_bytes())
    body = current_raw.decode()
    device_quote = next(line for line in body.splitlines() if line.startswith('- **Python 怎麼寫。**'))
    pointer = read_ref(group['prior_active_pointer'])
    manifest_ref = {'path': pointer['manifest'], 'sha256': pointer['manifest_sha256']}
    prior_manifest = read_ref(manifest_ref)
    prior_freeze = read_ref(prior_manifest['source_freeze'])
    prior_page = next(p for p in prior_freeze['pages'] if p['page_id'] == 'training')
    prior_raw = (ROOT / prior_page['snapshot']).read_bytes()
    assert sha(prior_raw) == prior_page['source_sha256']
    prior_quote = next(line for line in prior_raw.decode().splitlines() if line.startswith('- **Python 怎麼寫。**'))
    assert device_quote == prior_quote
    dispositions = read_ref(prior_manifest['dispositions'])
    verification = read_ref(group['items'][0]['historical_fix_verification']['report'])
    old_rows = []
    for item in group['items']:
        old = node(dispositions, item['disposition_ref']['json_pointer'])
        issue_report = read_ref(item['original_report'])
        issue = node(issue_report, item['json_pointer'])
        assert old['report'] == item['original_report'] and old['json_pointer'] == item['json_pointer']
        assert old['issue_id'] == item['issue_id'] and old['final_necessity'] == item['necessity'] == 'necessary'
        assert old['action'] == 'fixed' and old['affected_pages'] == item['affected_pages'] == ['training']
        assert item['historical_fix_verification'] in old['verification']
        assert issue['necessary'] is True and issue['claim_id'] == 'device_generalization'
        original_freeze = read_ref(issue_report['source_freeze'])
        original_page = next(p for p in original_freeze['pages'] if p['page_id'] == 'training')
        original_raw = (ROOT / original_page['snapshot']).read_bytes()
        assert sha(original_raw) == original_page['source_sha256'] == issue_report['source_sha256']
        original_quote = next(line for line in original_raw.decode().splitlines() if line.startswith('- **Python 怎麼寫。**'))
        old_rows.append({'request_item': item, 'original_issue': issue,
                         'original_disposition_node_sha256': sha(json.dumps(old, ensure_ascii=False, sort_keys=True,
                                                                            separators=(',', ':')).encode()),
                         'original_snapshot': reference(ROOT / original_page['snapshot']),
                         'original_source_quote': original_quote})
    assert verification['reviewer_task'] == group['old_owner']
    assert verification['source_sha256'] == prior_page['source_sha256']
    assert verification['figure_sha256'] == prior_page['figures_sha256'] == {}
    assert verification['verdict'] == 'pass' and 'device_generalization' in verification['resolved_issue_ids']
    unavailable = read_ref(group['unavailability']['report'])
    record = node(unavailable, group['unavailability']['json_pointer'])
    assert record['reviewer_task'] == group['old_owner'] and record['unavailable'] is True

    SCRATCH.mkdir(parents=True)
    from tiny_perceptron.training import choose_device
    device_cases = []
    for cuda, mps, expected in [(True, True, 'cuda'), (True, False, 'cuda'),
                                (False, True, 'mps'), (False, False, 'cpu')]:
        with patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
            auto = str(choose_device())
            cpu = str(choose_device('cpu'))
        assert auto == expected and cpu == 'cpu'
        device_cases.append({'mock_cuda_available': cuda, 'mock_mps_available': mps,
                             'auto': auto, 'explicit_cpu': cpu, 'scope': 'selection logic only; no GPU/MPS computation'})
    parser_defaults = {
        'train': capture_defaults('scripts.train', ['--task', 'text']),
        'evaluate': capture_defaults('scripts.evaluate', [str(FIXTURE)]),
        'infer': capture_defaults('scripts.infer', [str(FIXTURE)]),
        'fixed_experiment': capture_defaults('scripts.course_experiments.run', ['--experiment', 'simple_models']),
        'quantize': capture_defaults('scripts.quantize', [str(FIXTURE)]),
        'prepare': capture_defaults('scripts.prepare_data', []),
        'fetch': capture_defaults('scripts.fetch_training_assets', ['--list']),
    }
    assert [parser_defaults[k]['device'] for k in ['train', 'evaluate', 'infer']] == ['auto'] * 3
    assert parser_defaults['fixed_experiment']['device'] == 'cpu'
    assert all('device' not in parser_defaults[k] for k in ['quantize', 'prepare', 'fetch'])

    runs = []
    for name, args in [
        ('reject-quantize-device', ['scripts/quantize.py', str(FIXTURE), '--output', str(SCRATCH / 'must-not-exist.pt'), '--device', 'cpu']),
        ('reject-prepare-device', ['scripts/prepare_data.py', '--output', str(SCRATCH / 'must-not-generate'), '--device', 'cpu']),
        ('reject-fetch-device', ['scripts/fetch_training_assets.py', '--asset', 'tinystories', '--output', str(SCRATCH / 'must-not-download'), '--device', 'cpu']),
        ('reject-fixed-mps', ['-m', 'scripts.course_experiments.run', '--experiment', 'simple_models', '--device', 'mps', '--output', str(SCRATCH / 'must-not-run-mps')]),
    ]:
        row = run(name, args, 2)
        if name != 'reject-fixed-mps':
            assert 'unrecognized arguments: --device cpu' in row['stderr']
        runs.append(row)
    assert not any((SCRATCH / s).exists() for s in ['must-not-exist.pt', 'must-not-generate', 'must-not-download', 'must-not-run-mps'])
    assert not torch.cuda.is_available(), 'CPU-only proof expected in this environment'
    row = run('reject-unavailable-cuda-no-fallback', ['-m', 'scripts.course_experiments.run', '--experiment', 'simple_models',
              '--device', 'cuda', '--output', str(SCRATCH / 'must-not-run-cuda')], 1)
    assert 'must not silently fall back to CPU' in row['stderr']
    runs.append(row)
    assert not (SCRATCH / 'must-not-run-cuda').exists()
    dry_reports = []
    for label, extra in [('dry-text-auto', []), ('dry-text-explicit-cpu', ['--device', 'cpu'])]:
        row = run(label, ['scripts/train.py', '--task', 'text', '--output', str(SCRATCH / 'must-not-train.pt'), *extra], 0)
        report = json.loads(row['stdout'].splitlines()[-1])
        assert report['device'] == 'cpu' and report['mode'] == 'dry-run-no-weight-update'
        dry_reports.append(report)
        runs.append(row)
    assert not (SCRATCH / 'must-not-train.pt').exists()
    assert dry_reports[0]['history'] == dry_reports[1]['history']
    infer_reports = []
    for label, extra in [('infer-auto', []), ('infer-explicit-cpu', ['--device', 'cpu'])]:
        row = run(label, ['scripts/infer.py', str(FIXTURE), '--chat', '--prompt', 'color=blue;shape=square;pitch=low;color?',
                          '--tokens', '2', '--json', *extra], 0)
        infer_reports.append(json.loads(row['stdout']))
        runs.append(row)
    assert infer_reports[0] == infer_reports[1]
    for label, extra in [('evaluate-auto', []), ('evaluate-explicit-cpu', ['--device', 'cpu'])]:
        runs.append(run(label, ['scripts/evaluate.py', str(FIXTURE), '--mode', 'sft', '--tokens', '2', '--limit', '1',
                                '--output', str(SCRATCH / (label + '.json')), *extra], 0))
    eval_auto = json.loads((SCRATCH / 'evaluate-auto.json').read_bytes())
    eval_cpu = json.loads((SCRATCH / 'evaluate-explicit-cpu.json').read_bytes())
    assert eval_auto == eval_cpu and eval_auto['declared_split'] == 'train-diagnostic'

    quantize = importlib.import_module('scripts.quantize')
    original_loader = quantize.load_checkpoint
    loader_calls = []

    def observe_loader(*args, **kwargs):
        model, payload = original_loader(*args, **kwargs)
        effective = kwargs.get('device', args[1] if len(args) > 1 else inspect.signature(original_loader).parameters['device'].default)
        devices = sorted({str(p.device) for p in model.parameters()} | {str(b.device) for b in model.buffers()})
        loader_calls.append({'argc': len(args), 'kwargs': kwargs, 'effective_device': str(effective), 'loaded_tensor_devices': devices})
        assert effective == 'cpu' and devices == ['cpu']
        return model, payload

    buffer = io.StringIO()
    quant_output = SCRATCH / 'quantized-cpu.pt'
    quant_start = now()
    quant_argv = ['scripts/quantize.py', str(FIXTURE), '--bits', '4', '--output', str(quant_output)]
    with patch.object(sys, 'argv', quant_argv), patch.object(quantize, 'load_checkpoint', observe_loader), contextlib.redirect_stdout(buffer):
        quantize.main()
    assert len(loader_calls) == 1
    qstate = torch.load(quant_output, map_location='cpu', weights_only=True)['model']
    assert sorted({str(v.device) for v in qstate.values()}) == ['cpu']
    quantization = {'argv': quant_argv, 'started_at': quant_start, 'completed_at': now(), 'exit_code': 0,
                    'stdout': buffer.getvalue(), 'load_observation': loader_calls,
                    'output': reference(quant_output), 'output_tensor_devices': ['cpu'],
                    'scope': 'actual main/load/pack/save on CPU; observer wraps loader without altering its return'}
    assert reference(FIXTURE) == fixture_before and [reference(p) for p in OWN_REPORTS] == own_before
    paths = ['scripts/train.py', 'scripts/evaluate.py', 'scripts/infer.py', 'scripts/quantize.py',
             'scripts/prepare_data.py', 'scripts/fetch_training_assets.py', 'scripts/course_experiments/run.py',
             'tiny_perceptron/training.py']
    receipts = {'schema_version': 1, 'reviewer_task': TASK, 'created_at': now(),
                'request': reference(REQUEST), 'request_pointer': '/groups/9',
                'prior_generation': group['prior_generation'], 'old_owner': group['old_owner'], 'old_role': group['old_role'],
                'current_freeze': reference(FREEZE), 'current_page': page, 'complete_source_read': True,
                'current_source_quote': device_quote, 'prior_accepted_source_quote': prior_quote,
                'unchanged_device_paragraph': True, 'prior_active_pointer': group['prior_active_pointer'],
                'prior_manifest': manifest_ref, 'prior_freeze': prior_manifest['source_freeze'],
                'prior_page': prior_page, 'prior_gate': pointer['actual_completed_gate'],
                'prior_dispositions': prior_manifest['dispositions'], 'items': old_rows,
                'original_closure': group['items'][0]['historical_fix_verification'],
                'unavailability': group['unavailability'], 'unavailability_record': record,
                'current_code': [reference(ROOT / p) for p in paths], 'preserved_own_reports': own_before,
                'scope_excluded': ['groups/8 continuity items', 'reader answer contents', 'GPU/MPS execution',
                                   'training updates', 'formal experiment reruns', 'network downloads', 'whole-stage gate']}
    save_new(FORMAL / 'historical-rv1019-source-receipts-v1.json', receipts)
    result = {'schema_version': 1, 'reviewer_task': TASK, 'started_at': started, 'completed_at': now(),
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'platform': platform.platform(), 'device': 'cpu',
                              'cuda_available': str(torch.cuda.is_available()), 'mps_available': str(torch.backends.mps.is_available())},
              'source_receipts': reference(FORMAL / 'historical-rv1019-source-receipts-v1.json'),
              'fixture': fixture_before, 'fixture_scope': 'known two-update CPU fixture from own original; reused only for CLI/device mechanics, no quality claim',
              'mock_device_cases': device_cases, 'actual_parser_defaults': parser_defaults,
              'runs': runs, 'quantization_loader_observation': quantization,
              'summary': {'subprocess_commands': len(runs), 'quantize_main_calls': 1,
                          'dry_no_update': 2, 'new_training_updates': 0,
                          'hardware_training_measurements': 0, 'all_expected_results': True},
              'not_claimed': 'No GPU/MPS hardware validation, runtime cross-device equality, model ability, full fixed recipe success, fullstage or release acceptance.'}
    save_new(FORMAL / 'historical-rv1019-device-execution-v1.json', result)
    print(json.dumps({'execution': reference(FORMAL / 'historical-rv1019-device-execution-v1.json'),
                      'sources': reference(FORMAL / 'historical-rv1019-source-receipts-v1.json'),
                      'summary': result['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
