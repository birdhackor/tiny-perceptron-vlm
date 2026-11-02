"""Audit preserved T.8 results and their data/code fingerprints without training."""

import ast
import hashlib
import json
import platform
import statistics
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_experiments.common import records_sha256, split_records

ROOT = Path('/workspace/tiny-perceptron-vlm')
CACHE = Path('/workspace/work/reference/omni-technical-t-8-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-technical-t-8-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    text = path.read_text()
    lines = text.splitlines(True)
    return {n.name: ''.join(lines[n.lineno - 1:n.end_lineno])
            for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def main():
    audit = {'created_at': datetime.now(UTC).isoformat(),
             'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
             'scope': 'CPU audit/recalculation of preserved historical L4 records. No new GPU measurements.',
             'historical': {}}
    source = 'scripts/course_experiments/architecture.py'
    current = functions(ROOT / source)
    expected_seconds = {'modern': '44.6', 'moe': '65.7', 'efficiency': '32.3', 'precision': '15.0', 'flash_probe': '4.5'}
    selected = {'modern': ['run_modern', '_text_dataset', '_train'],
                'moe': ['run_moe', '_text_dataset', '_train'],
                'efficiency': ['run_efficiency', '_sft_dataset', '_benchmark'],
                'precision': ['run_precision', '_sft_dataset', '_train'],
                'flash_probe': ['run_flash_probe', '_flash_profile', '_flash_measure']}
    for name, expected in expected_seconds.items():
        path = ROOT / f'docs/course-experiments/results/{name}.json'
        result = json.loads(path.read_text())
        assert result['gpu'] == 'NVIDIA L4' and result['device'] == 'cuda'
        assert result['step_scale'] == 1 and result['evidence_status'] == 'complete_run'
        assert not result['unfinished_schedules'] and result['modal']['run_id']
        assert f"{result['elapsed_seconds']:.1f}" == expected
        old_path = CACHE / f'{name}-original-architecture.py'
        assert sha(old_path) == result['code_sha256'][source]
        old = functions(old_path)
        assert all(old[fn] == current[fn] for fn in selected[name])
        row = {k: result[k] for k in ['revision', 'gpu', 'device', 'torch_version', 'python_version',
                                     'seed', 'elapsed_seconds', 'timing_scope', 'evidence_status']}
        row.update(result_path=str(path.relative_to(ROOT)), result_sha256=sha(path),
                   rounded_seconds=expected, modal_run_id=result['modal']['run_id'],
                   original_code_sha256=sha(old_path), original_code_cache=str(old_path),
                   checked_functions=selected[name], checked_function_text_identical_to_current=True)
        data = result['results']
        row['dataset'] = data.get('dataset')
        if name in ['modern', 'moe', 'precision']:
            row['variants'] = {}
            for variant, value in data['variants'].items():
                train = value['training']
                assert train['steps'] == train['requested_steps'] == train['optimizer_updates']
                row['variants'][variant] = {'config': value['model']['config'],
                    'training': {k: train[k] for k in ['steps', 'batch_size', 'effective_tokens', 'auxiliary_weight']},
                    'heldout': {s: {k: v[k] for k in ['records', 'effective_tokens', 'nll']}
                                for s, v in value['heldout'].items()}}
        if name == 'modern':
            assert len(row['variants']) == 6
            assert set(row['variants']) == {'baseline', 'rope', 'rmsnorm', 'relu2', 'swiglu', 'tied'}
            assert all(v['training']['steps'] == 240 for v in row['variants'].values())
        if name == 'moe':
            assert len(row['variants']) == 7
            assert [row['variants'][n]['config']['width'] for n in ['dense_active_top1', 'dense_active_top2', 'dense_total']] == [64, 80, 104]
            assert all(v['training']['steps'] == 180 for v in row['variants'].values())
            row['teacher_variant'] = data['teacher_variant']
            assert row['teacher_variant'] == 'top2_aux0.01'
        if name == 'flash_probe':
            row['probe'] = {k: data[k] for k in ['status', 'schedule_completed', 'verification_passed', 'configuration', 'supported_routes']}
            row['routes'] = {}
            for route, values in data['routes'].items():
                assert values['status'] == 'completed'
                assert values['all_checks_finite'] and values['all_comparisons_within_tolerance']
                assert all(p['flash_cuda_verified'] for p in values['profiles'].values())
                measured = {}
                for mode, backends in values['measurements'].items():
                    for backend, m in backends.items():
                        assert m['warmup_calls'] == 3 and m['measured_calls'] == 9
                        assert statistics.median(m['samples_seconds']) == m['median_seconds']
                        assert m['peak_allocated_bytes'] - m['allocated_before_bytes'] == m['additional_peak_allocated_bytes']
                        measured[f'{mode}/{backend}'] = {k: m[k] for k in ['warmup_calls', 'measured_calls', 'median_seconds',
                            'allocated_before_bytes', 'peak_allocated_bytes', 'additional_peak_allocated_bytes']}
                row['routes'][route] = {'status': values['status'], 'profiler_flash_verified': True,
                                        'finite_and_within_tolerance': True, 'measurements': measured}
        audit['historical'][name] = row
    archive = ROOT / 'assets/training/tinystories-v1.tar.gz'
    assert sha(archive) == '1ac125d427040eb3311c237fbbc65102adfefe62e0a4ecafd78c395a5db59e23'
    with tarfile.open(archive) as tar:
        name = next(n for n in tar.getnames() if n.endswith('tinystories-train-512.jsonl'))
        raw = tar.extractfile(name).read()
    assert hashlib.sha256(raw).hexdigest() == '7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0'
    records = [json.loads(line) for line in raw.decode().splitlines()]
    for record in records:
        record['family'] = record.get('text_sha256', records_sha256([{'text': record['text']}]))
    data = split_records(records, 42)
    report = {name: {'records': len(rows), 'sha256': records_sha256(rows)} for name, rows in data.items()}
    assert report == audit['historical']['modern']['dataset'] == audit['historical']['moe']['dataset']
    audit['reconstructed_tinystories'] = {'archive_sha256': sha(archive), 'records': len(records), 'split': report}
    audit['checks_passed'] = True
    path = OUT / 'evidence-audit.json'
    assert not path.exists()
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'checks_passed': True}))


if __name__ == '__main__':
    main()
