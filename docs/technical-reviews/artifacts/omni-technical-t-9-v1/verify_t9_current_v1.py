"""T.9 v6 callback: exact commands, current split, immutable original evidence."""

import difflib
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).parent
CACHE = Path('/workspace/work/reference/omni-technical-t-9-v1')
WORK = CACHE / 'current-v1'
ORIGINAL = ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_t_9/technical-original-v1.json'
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v6.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    torch.set_num_threads(1)
    original = json.loads(ORIGINAL.read_text())
    assert sha(ORIGINAL) == 'fa8859b0a136bfec61612f564da3e8c61753d3e07f37b7d4e6e4797eaf3b1665'
    original_execution = json.loads((ART / 'execution-summary-v1.json').read_text())
    page = next(p for p in json.loads(FREEZE.read_text())['pages'] if p['page_id'] == 'T.9')
    snapshot = ROOT / page['snapshot']
    assert sha(snapshot) == page['source_sha256']
    text = snapshot.read_text()
    before = (ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.9.md').read_text()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in ['scripts', '.venv']:
        (WORK / name).symlink_to(ROOT / name, target_is_directory=True)
    (WORK / 'checkpoints').mkdir()
    old_fixture = CACHE / 'execution-v3/checkpoints/attributes.pt'
    assert sha(old_fixture) == original_execution['fixture']['checkpoint_sha256']
    (WORK / 'checkpoints/attributes.pt').symlink_to(old_fixture)
    commands = []

    def run(command, label):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run(shlex.split(command), cwd=WORK, env=env,
                                text=True, capture_output=True, timeout=35)
        raw = WORK / 'raw' / f'{label}.txt'
        save(raw, result.stdout + result.stderr)
        receipt = {'command': command, 'cwd': str(WORK), 'returncode': result.returncode,
                   'raw_path': str(raw), 'raw_sha256': sha(raw)}
        commands.append(receipt)
        if result.returncode:
            raise RuntimeError(receipt)
        return result.stdout

    run('.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42', 'prepare')
    parts = {split: [json.loads(line) for line in (WORK / f'data/generated/attributes-sft/{split}.jsonl').read_text().splitlines()]
             for split in ['train', 'validation', 'test']}
    spec = importlib.util.spec_from_file_location('current_t9_prepare', ROOT / 'scripts/prepare_data.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    records = generator.generate_records('attributes-sft')
    families = sorted({record['family'] for record in records})
    random.Random(42).shuffle(families)
    cut1, cut2 = int(0.8 * len(families)), int(0.9 * len(families))
    assignment = {family: 'train' if i < cut1 else 'validation' if i < cut2 else 'test'
                  for i, family in enumerate(families)}
    for split, actual in parts.items():
        expected = [dict(record, split=split) for record in records if assignment[record['family']] == split]
        assert actual == expected
    assert len(records) == 60
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not {r['family'] for r in parts[left]} & {r['family'] for r in parts[right]}

    probes = []
    validation_commands = 0
    settings_boundary = None
    inline_stdout = None
    for kind, block in re.findall(r'```(bash|python)\n(.*?)```', text, re.S):
        if kind == 'python':
            inline_path = WORK / 'outputs/size_check.py'
            save(inline_path, block)
            inline_stdout = run('.venv/bin/python outputs/size_check.py', 'inline-size')
            continue
        for command in block.strip().splitlines():
            if not any(f'scripts/{tool}.py' in command for tool in ['quantize', 'infer', 'evaluate']):
                continue
            if 'scripts/evaluate.py' in command and '/test.jsonl' in command and settings_boundary is None:
                assert validation_commands == 3 and len(probes) == 2
                boundary = {'created_at': datetime.now(UTC).isoformat(), 'validation_commands_completed': 3,
                            'prior_probe_splits': [p['matches'][0]['split'] for p in probes],
                            'precommitted_variants': ['fp32', 'int4', 'int8'],
                            'selection_scope': 'Software smoke only. No trained-quality bit selection claimed; all requested variants and generation settings fixed before test output.'}
                boundary_path = WORK / 'settings-boundary-before-test.json'
                save(boundary_path, json.dumps(boundary, ensure_ascii=False, indent=2) + '\n')
                settings_boundary = {'path': str(boundary_path), 'sha256': sha(boundary_path), **boundary}
            stdout = run(command, f'command-{len(commands):02}')
            if 'scripts/infer.py' in command:
                args = shlex.split(command)
                prompt = args[args.index('--prompt') + 1]
                matches = [{'split': split, 'row': i, 'family': record['family'],
                            'target': record['messages'][-1]['content']}
                           for split, rows in parts.items() for i, record in enumerate(rows)
                           if record['messages'][0]['content'] == prompt]
                assert len(matches) == 1
                assert matches[0]['split'] == 'train' and matches[0]['family'] == 'blue:square:low'
                assert matches[0]['target'] == 'blue'
                generated = json.loads(stdout)
                assert len(generated['generated_ids']) <= 24
                probes.append({'command': command, 'prompt': prompt, 'matches': matches,
                               'generated_token_count': len(generated['generated_ids']),
                               'generated_ids': generated['generated_ids'], 'generation_status': generated['generation_status'],
                               'eos': generated['eos']})
            if 'scripts/evaluate.py' in command and '/validation.jsonl' in command:
                validation_commands += 1
    assert len(probes) == 2 and probes[0]['prompt'] == probes[1]['prompt']
    assert settings_boundary is not None and inline_stdout is not None
    assert len(inline_stdout.strip().splitlines()) == 3
    assert sha(old_fixture) == original_execution['fixture']['checkpoint_sha256']
    run('.venv/bin/python -m scripts.course_experiments.run --help', 'fixed-recipe-help')
    reports = {}
    for split in ['validation', 'test']:
        reports[split] = {}
        for variant in ['fp32', 'int4', 'int8']:
            path = WORK / f'outputs/attributes-{variant}-{split}.json'
            report = json.loads(path.read_text())
            assert report['declared_split'] == split
            assert report['limit'] == 'all' and report['unselected_records'] == 0
            assert report['generation_evaluated_records'] == len(parts[split])
            assert report['generation_skipped_records'] == 0
            assert [sample['row'] for sample in report['samples']] == list(range(len(parts[split])))
            reports[split][variant] = {'path': str(path), 'sha256': sha(path),
                                      **{k: report[k] for k in ['declared_split', 'records_read', 'generation_evaluated_records',
                                          'generation_skipped_records', 'effective_tokens', 'metric_denominators']}}
    identities = []
    for source in original['sources']:
        if source['kind'] == 'repository_code':
            path = ROOT / source['path']
            assert sha(path) == source['sha256']
            identities.append({'source_id': source['id'], 'path': source['path'], 'sha256': sha(path), 'unchanged': True})
        elif source.get('original_cache_path'):
            assert sha(source['original_cache_path']) == source['original_sha256']
            identities.append({'source_id': source['id'], 'path': source['original_cache_path'],
                               'sha256': source['original_sha256'], 'unchanged': True})
    source_lines = text.splitlines()
    validation_line = next(i for i, line in enumerate(source_lines, 1) if '先用validation選位數' in line)
    test_lines = [i for i, line in enumerate(source_lines, 1) if 'scripts/evaluate.py' in line and '/test.jsonl' in line]
    probe_lines = [i for i, line in enumerate(source_lines, 1) if 'scripts/infer.py' in line]
    assert max(probe_lines) < validation_line < min(test_lines)
    summary = {'created_at': datetime.now(UTC).isoformat(),
               'reviewer_task': '/root/omni_technical_coordinator/technical_t_9',
               'original_report_sha256': sha(ORIGINAL), 'source_snapshot': page,
               'freeze_sha256': sha(FREEZE),
               'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                               'torch_git_version': torch.version.git_version, 'device': 'cpu',
                               'torch_num_threads': '1', 'platform': platform.platform()},
               'scope': 'Callback CPU software recheck using unchanged one-update fixture. No new optimizer updates, long training, GPU, paid execution or quality conclusions.',
               'text_diff': ''.join(difflib.unified_diff(before.splitlines(True), text.splitlines(True), fromfile='v5T9', tofile='v6T9')),
               'identity_checks': identities, 'commands': commands, 'probes': probes,
               'generator_function_matches_cli': True,
               'data': json.loads((WORK / 'data/generated/attributes-sft/manifest.json').read_text())['splits'],
               'source_order': {'probe_command_lines': probe_lines, 'validation_selection_line': validation_line,
                                'test_command_lines': test_lines},
               'settings_boundary': settings_boundary,
               'inline_size_stdout': inline_stdout, 'evaluations': reports}
    output = ART / 'current-execution-summary-v1.json'
    save(output, json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summary': str(output), 'sha256': sha(output),
                      'probes': [{'matches': p['matches'], 'generated_tokens': p['generated_token_count']} for p in probes],
                      'commands': len(commands), 'all_identities_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
