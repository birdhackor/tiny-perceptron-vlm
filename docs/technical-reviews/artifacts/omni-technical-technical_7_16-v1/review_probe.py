"""7.16 reviewer: exact embedded example, existing result audit and tiny CPU inference.

No training is performed. Third-party files, weights and detailed generations stay
under /workspace/work/reference. Run this file from the repository root.
"""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import evaluate_lm, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.training import load_checkpoint

OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-technical_7_16-v1')
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
RESULT = ROOT / 'docs/course-experiments/results/sft_ablation.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def section(path, identifier):
    raw = path.read_bytes()
    pattern = rb'^## ' + re.escape(identifier.encode()) + rb' .+$'
    start = re.search(pattern, raw, re.M).start()
    following = re.search(rb'^## ', raw[start + 1:], re.M)
    end = start + 1 + following.start() if following else len(raw)
    return raw[start:end]


def audit():
    torch.set_num_threads(2)
    report = {
        'reviewer_task': '/root/omni_technical_coordinator/technical_7_16',
        'started_at': datetime.now(UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_7_16-v1/review_probe.py',
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'device': 'cpu', 'threads': str(torch.get_num_threads()), 'platform': platform.platform()},
        'no_training': True,
    }
    frozen = json.loads(FREEZE.read_bytes())
    page_map = {p['page_id']: p for p in frozen['pages']}
    report['freeze_sha256'] = sha(FREEZE.read_bytes())
    report['source_checks'] = {}
    for identifier in ['7.16', '7.15', '7.10', '5.16']:
        page = page_map[identifier]
        original = (ROOT / page['snapshot']).read_bytes()
        current = section(ROOT / page['source'], identifier)
        assert original == current
        assert sha(original) == page['source_sha256']
        report['source_checks'][identifier] = {
            'snapshot': page['snapshot'], 'source': page['source'],
            'source_sha256': sha(original), 'current_exact_slice_equal': True,
        }
    assert page_map['7.16']['figures_sha256'] == {}
    nb = ROOT / page_map['7.16']['notebook']
    assert sha(nb.read_bytes()) == page_map['7.16']['notebook_sha256']
    lesson = (ROOT / page_map['7.16']['snapshot']).read_text()
    embedded = re.findall(r'```python\n(.*?)\n```', lesson, re.S)
    assert len(embedded) == 1
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(embedded[0], '7.16 embedded Python', 'exec'), namespace)
    assert [r['source'] for r in namespace['recipe']] == ['A', 'A', 'B', 'B']
    assert sum(r['answer_tokens'] for r in namespace['recipe']) == 8
    nb_cells = [c for c in json.loads(nb.read_text())['cells'] if c['cell_type'] == 'code']
    assert any(embedded[0].strip() in ''.join(c['source']) for c in nb_cells)
    a, b = namespace['a'], namespace['b']
    exercise = [a[0], b[0], b[1], b[0]]
    report['embedded'] = {
        'sha256': sha(embedded[0].encode()), 'stdout': output.getvalue(),
        'notebook_sha256': sha(nb.read_bytes()), 'notebook_contains_same_example': True,
        'exercise_counts': dict(Counter(r['source'] for r in exercise)),
        'exercise_effective_targets': sum(r['answer_tokens'] for r in exercise),
        'unique_B_records': len({id(r) for r in exercise if r['source'] == 'B'}),
    }
    for record in b:
        record['answer_tokens'] = 4
    targets = {key: sum(r['answer_tokens'] for r in exercise if r['source'] == key) for key in ['A', 'B']}
    report['embedded']['exercise_B_length_4_targets'] = targets
    report['embedded']['exercise_B_length_4_target_shares'] = {k: v / sum(targets.values()) for k, v in targets.items()}
    assert targets == {'A': 2, 'B': 12}
    report['hypothetical_budgets'] = {
        'fixed_four_B_only': {'records': 4, 'effective_targets': 4 * 2, 'B_targets': 8},
        'fixed_four_replay': {'records': 4, 'effective_targets': 2 * 2 + 2 * 2, 'B_targets': 4},
        'keep_two_B_then_add_two_A': {'B_only_records': 2, 'B_only_targets': 4,
                                      'replay_records': 4, 'replay_targets': 8},
        'scope': 'Equal records and declared targets are not a measured FLOP or wall-clock equality.',
    }

    original = json.loads(RESULT.read_bytes())
    report['recorded_result_sha256'] = sha(RESULT.read_bytes())
    report['recorded_environment'] = {k: original[k] for k in [
        'revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'elapsed_seconds',
        'timing_scope', 'step_scale', 'evidence_status']}
    assert round(original['elapsed_seconds'], 1) == 23.5
    report['code_checks'] = {}
    for name in ['scripts/course_experiments/text.py', 'scripts/course_experiments/common.py',
                 'scripts/prepare_data.py', 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']:
        current = (ROOT / name).read_bytes()
        old = subprocess.run(['git', 'show', original['revision'] + ':' + name], check=True, capture_output=True).stdout
        assert current == old
        if name in original['code_sha256']:
            assert sha(current) == original['code_sha256'][name]
        report['code_checks'][name] = {'sha256': sha(current), 'identical_to_recorded_revision': True}

    parts = {'A_attributes': split_records(generate_records('attributes-sft'), seed=42),
             'B_arithmetic': split_records(arithmetic_records(), seed=42)}
    report['dataset_checks'] = {}
    for task, task_parts in parts.items():
        manifest_name = 'attributes' if task == 'A_attributes' else 'arithmetic'
        result = {}
        families = {split: {r['family'] for r in rows} for split, rows in task_parts.items()}
        assert not (families['train'] & families['validation'] or families['train'] & families['test'] or families['validation'] & families['test'])
        for split, rows in task_parts.items():
            raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
            registered = original['results']['data'][manifest_name][split]
            assert sha(raw) == registered['sha256']
            assert len(rows) == registered['records']
            examples = text_examples(rows, mode='sft')
            result[split] = {'records': len(rows), 'families': len(families[split]), 'sha256': sha(raw),
                             'effective_targets': sum(int((y != IGNORE).sum()) for _, y in examples),
                             'input_length_range': [min(len(x) for x, _ in examples), max(len(x) for x, _ in examples)]}
        result['no_family_overlap'] = True
        report['dataset_checks'][task] = result

    report['training_exposure_reconstruction'] = {}
    for name in ['b-only', 'replay']:
        rows = parts['B_arithmetic']['train'] if name == 'b-only' else parts['A_attributes']['train'] + parts['B_arithmetic']['train']
        examples = text_examples(rows, mode='sft')
        assert records_sha256(rows) == original['results']['runs'][name]['training']['records_sha256']
        rng = random.Random(42)
        count, input_count, padded_count = 0, 0, 0
        source_count, source_targets = Counter(), Counter()
        for _ in range(500):
            indices = rng.choices(range(len(examples)), k=16)
            x, y, _ = pad_batch([examples[i] for i in indices])
            count += int((y != IGNORE).sum())
            input_count += sum(len(examples[i][0]) for i in indices)
            padded_count += x.numel()
            for i in indices:
                source = 'A' if name == 'replay' and i < 45 else 'B'
                source_count[source] += 1
                source_targets[source] += int((examples[i][1] != IGNORE).sum())
        assert count == original['results']['runs'][name]['training']['effective_tokens']
        report['training_exposure_reconstruction'][name] = {
            'steps': 500, 'batch_size': 16, 'effective_targets': count,
            'nonpadding_input_positions': input_count, 'padded_input_positions': padded_count,
            'source_draws': dict(source_count), 'source_targets': dict(source_targets),
            'method': 'Replay the original random.Random(42).choices schedule and pad_batch; no optimizer or forward call.',
        }
    report['target_exposure_ratio_replay_over_B_only'] = (
        report['training_exposure_reconstruction']['replay']['effective_targets'] /
        report['training_exposure_reconstruction']['b-only']['effective_targets'])

    tokenizer = ByteTokenizer()
    report['full_recorded_sample_audit'] = {}
    detailed = {}
    weight_receipts = []
    for state, experiment, filename in [('before', 'sft', 'model.pt'), ('b-only', 'sft_ablation', 'b-only.pt'),
                                         ('replay', 'sft_ablation', 'replay.pt')]:
        manifest_path = ROOT / f'docs/course-experiments/public-releases/{experiment}.json'
        manifest = json.loads(manifest_path.read_bytes())['release']['public_manifest']
        file_entry = next(f for f in manifest['files'] if f['output'] == filename)
        weight_path = CACHE / (experiment + '-' + filename)
        assert sha(weight_path.read_bytes()) == file_entry['sha256']
        assert weight_path.stat().st_size == file_entry['bytes']
        weight_receipts.append({'state': state, 'local_path': str(weight_path),
                                'url': f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{file_entry['path']}",
                                'sha256': file_entry['sha256'], 'bytes': file_entry['bytes'],
                                'public_manifest': str(manifest_path.relative_to(ROOT)),
                                'public_manifest_sha256': sha(manifest_path.read_bytes())})
        model, payload = load_checkpoint(weight_path, device='cpu')
        assert sum(p.numel() for p in model.parameters()) == 141568
        recorded = original['results']['before'] if state == 'before' else original['results']['runs'][state]
        state_summary = {}
        detailed[state] = {}
        for task, task_parts in parts.items():
            detailed[state][task] = {}
            for split in ['validation', 'test']:
                records = task_parts[split]
                old_eval = recorded[task][split]
                assert len(old_eval['samples']) == len(records) == old_eval['records']
                matches, eos = 0, 0
                for sample, row in zip(old_eval['samples'], records, strict=True):
                    assert sample['messages'] == row['messages'][:-1]
                    assert sample['expected'] == row['messages'][-1]['content']
                    ids = sample['generated_ids']
                    raw = ids[:ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids
                    exact = raw == tokenizer.encode(sample['expected'])
                    ended = tokenizer.eos_id in ids
                    assert exact == sample['exact'] and ended == sample['eos']
                    matches += int(exact)
                    eos += int(ended)
                assert matches == old_eval['matches']
                assert matches / len(records) == old_eval['exact_match']
                assert eos / len(records) == old_eval['eos_rate']
                current = evaluate_lm(model, records, mode='sft', tokens=32)
                detailed[state][task][split] = current
                assert current['records'] == old_eval['records']
                assert current['matches'] == old_eval['matches']
                assert current['eos_rate'] == old_eval['eos_rate']
                assert [s['generated_ids'] for s in current['samples']] == [s['generated_ids'] for s in old_eval['samples']]
                assert current['effective_tokens'] == old_eval['effective_tokens']
                assert abs(current['nll'] - old_eval['nll']) < 1e-5
                state_summary[task + '/' + split] = {
                    'matches': matches, 'records': len(records), 'eos_count': eos,
                    'raw_ids_recomputed_from_all_saved_samples': True,
                    'CPU_published_weights_same_generated_ids_for_every_row': True,
                    'effective_targets': current['effective_tokens'],
                    'cpu_nll': current['nll'], 'recorded_nll': old_eval['nll'],
                }
        report['full_recorded_sample_audit'][state] = state_summary
    report['weight_receipts'] = weight_receipts
    raw_path = CACHE / 'cpu-evaluations-v1.json'
    dump_new(raw_path, detailed)
    report['detailed_cpu_evaluations'] = {'path': str(raw_path), 'sha256': sha(raw_path.read_bytes()), 'bytes': raw_path.stat().st_size}
    help_result = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'], check=True, capture_output=True, text=True)
    assert '--experiment' in help_result.stdout and '{cpu,cuda}' in help_result.stdout
    spec = experiment_spec('sft_ablation')
    report['command_interface'] = {
        'command': 'uv run --extra cpu python -m scripts.course_experiments.run --help',
        'returncode': help_result.returncode, 'stdout': help_result.stdout,
        'registered_experiment_id': spec['id'], 'module': spec['module'], 'function': spec['function'],
        'cuda_training_command_executed': False,
        'dependency_checked_in_source': 'Context.dependency("sft") defaults to sft/model.pt; run_sft_ablation loads it then deep-copies the base for each branch.',
    }
    report['finished_at'] = datetime.now(UTC).isoformat()
    report['result'] = 'All assertions passed; no training, GPU use, large model or paid execution.'
    return report


if __name__ == '__main__':
    result = audit()
    output_path = OUT / 'probe-result-v1.json'
    dump_new(output_path, result)
    print(json.dumps({'result': result['result'], 'output': str(output_path.relative_to(ROOT)),
                      'sha256': sha(output_path.read_bytes()), 'environment': result['environment']}, ensure_ascii=False))
