"""Bounded CPU audit of existing records, not model evaluation or training."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import (
    TOK, build_dataset, encode_record, expected_final, parse_action, prompt_ids,
)
from tiny_perceptron.data import IGNORE

HERE = Path(__file__).resolve().parent
PINNED = HERE / 'sources/pinned'
VERSION = '1df335318bda03fd771807f66976953231d5a00b'

def load(name):
    path = PINNED / name
    raw = path.read_bytes()
    value = json.loads(raw)
    shapes[name] = {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'top_level_types': {k: type(v).__name__ for k, v in value.items()},
    }
    return value

shapes = {}
data_name = 'docs/course-experiments/capstone-evidence/deployment/data.json'
data = load(data_name)
splits, manifest = build_dataset()
assert splits == data['splits'], 'Current deterministic data differs from pinned original'
assert manifest['sha256'] == data['manifest']['sha256']
rows = data['splits']['validation']
by_id = {r['id']: r for r in rows}
assert len(by_id) == len(rows)
tasks4 = ('style', 'missing', 'unavailable', 'safety')
text_tasks = ('calculator', 'unavailable', 'tool_return', 'concept', 'rag', 'missing', 'safety', 'style')
other_tasks = ('calculator', 'tool_return', 'concept', 'rag')
stage_results = {}
pointer_log = {data_name: ['/splits/validation', '/manifest/sha256', '/manifest/version', '/manifest/seed']}
all_stage_records = {}

for stage in ('sft', 'joint', 'dpo'):
    name = f'docs/course-experiments/capstone-evidence/{stage}/validation.json'
    values = load(name)
    pointers = ['/count', '/action_correct', '/end_to_end_correct', '/by_task']
    counters = defaultdict(Counter)
    selected = []
    assert len(values['records']) == len(rows)
    assert {r['id'] for r in values['records']} == set(by_id)
    for i, rec in enumerate(values['records']):
        pointers += [f'/records/{i}/{k}' for k in (
            'id', 'task', 'expected_action', 'expected_final', 'action_trace',
            'parsed_action', 'runtime', 'final_trace', 'answer', 'action_correct',
            'end_to_end_correct')]
        row = by_id[rec['id']]
        trace = rec['action_trace']
        assert rec['task'] == row['task']
        assert rec['expected_action'] == row['answer']
        assert rec['expected_final'] == expected_final(row)
        assert trace['prompt_ids'] == prompt_ids(row)
        assert TOK.decode(trace['generated_ids']) == trace['raw']
        assert trace['eos'] == (trace['stop_reason'] == 'eos')
        assert trace['eos'] == (trace['generated_ids'][-1] == TOK.eos_id)
        parsed = parse_action(trace)
        assert parsed == rec['parsed_action']
        action_correct = trace['eos'] and trace['raw'] == row['answer']
        answer = parsed.get('content') if parsed['status'] in ('direct', 'ask') else None
        if rec['final_trace'] is not None:
            final = rec['final_trace']
            assert TOK.decode(final['generated_ids']) == final['raw']
            final_parsed = parse_action(final)
            if final_parsed['status'] == 'direct':
                answer = final_parsed['content']
        assert answer == rec['answer']
        end_correct = action_correct and answer == rec['expected_final']
        assert action_correct == rec['action_correct']
        assert end_correct == rec['end_to_end_correct']
        counters[row['task']].update(count=1, action_correct=int(action_correct), end_to_end_correct=int(end_correct))
        if row['task'] in tasks4 or row['task'] == 'rag':
            selected.append({
                'id': row['id'], 'task': row['task'], 'user': row['user'],
                'system': row['system'], 'expected': row['answer'],
                'generated': trace['raw'], 'eos': trace['eos'], 'correct': end_correct,
            })
    totals = dict(counters)
    assert totals == values['by_task']
    assert sum(t['count'] for t in totals.values()) == values['count']
    assert sum(t['action_correct'] for t in totals.values()) == values['action_correct']
    assert sum(t['end_to_end_correct'] for t in totals.values()) == values['end_to_end_correct']
    text_count = sum(totals[t]['count'] for t in text_tasks)
    text_correct = sum(totals[t]['end_to_end_correct'] for t in text_tasks)
    other_count = sum(totals[t]['count'] for t in other_tasks)
    other_correct = sum(totals[t]['end_to_end_correct'] for t in other_tasks)
    assert (other_count, other_correct) == (23, 23)
    assert all(totals[t]['count'] == totals[t]['end_to_end_correct'] for t in tasks4)
    stage_results[stage] = {
        'by_task_recomputed': totals,
        'text_all': {'correct': text_correct, 'count': text_count},
        'other_text': {'correct': other_correct, 'count': other_count, 'tasks': list(other_tasks)},
        'selected_raw_rows': selected,
    }
    all_stage_records[stage] = {r['id']: r for r in values['records']}
    pointer_log[name] = pointers
    report_name = f'docs/course-experiments/capstone-evidence/{stage}/train-report.json'
    report = load(report_name)
    keys = ['data_version', 'stage', 'requested_steps', 'steps', 'schedule_completed',
            'seed', 'device', 'code_sha256', 'parent_checkpoint_sha256', 'inference_export']
    pointer_log[report_name] = ['/' + k for k in keys]
    selected_provenance = {k: report[k] for k in keys}
    assert report['stage'] == stage and report['schedule_completed']
    assert report['steps'] == report['requested_steps']
    for name_code, digest in report['code_sha256'].items():
        assert hashlib.sha256((PINNED / name_code).read_bytes()).hexdigest() == digest
    stage_results[stage]['training_provenance'] = selected_provenance

constant_payloads = {}
for task in ('missing', 'unavailable', 'safety', 'rag'):
    subset = [r for r in rows if r['task'] == task]
    labels = Counter(r['answer'] for r in subset)
    constant_payloads[task] = {
        'labels': dict(labels), 'count': len(subset),
        'constant_answer_best_correct': max(labels.values()),
    }
    assert len(labels) == 1
assert constant_payloads['rag']['labels'] == {'DIRECT:書櫃': 3}
example_id = '691c9de656c01fa2df60'
example = by_id[example_id]
assert example['user'] == '照抄數字15，只要答案。'
assert example['system'] == '計算器=開；風格=短。'
assert all_stage_records['sft'][example_id]['action_trace']['raw'] == 'DIRECT:15'
assert len([r for r in rows if r['task'] == 'missing' and '14元' in r['user']]) == 0

# Variants exercise encoding and action grammar only; there is no model involved.
variants = []
for task in tasks4:
    row = next(r for r in splits['train'] if r['task'] == task)
    x, y = encode_record(row)
    prefix = prompt_ids(row)
    assert y[-1].item() == TOK.eos_id
    assert int((y != IGNORE).sum()) == len(TOK.encode(row['answer'])) + 1
    assert torch.all(y[:len(prefix)-1] == IGNORE)
    correct_trace = {'raw': row['answer'], 'eos': True}
    stopped_trace = {'raw': row['answer'], 'eos': False}
    assert parse_action(correct_trace)['status'] in ('direct', 'ask')
    assert parse_action(stopped_trace)['status'] == 'invalid'
    variants.append({'task': task, 'supervised_token_count': int((y != IGNORE).sum()),
                     'answer_utf8_byte_count': len(TOK.encode(row['answer'])),
                     'final_label': int(y[-1]), 'without_eos': parse_action(stopped_trace)})
available = next(r for r in splits['train'] if r['task'] == 'calculator')
disabled = next(r for r in splits['train'] if r['task'] == 'unavailable' and r['user'] == available['user'])
assert available['answer'].startswith('TOOL:calculator:')
assert disabled['answer'] == 'ASK:計算器未開'
assert available['system'] != disabled['system']
assert available['system'] in ('計算器=開；風格=短。', '計算器=關；風格=短。')

joint_to_dpo = []
for row in rows:
    before = all_stage_records['joint'][row['id']]
    after = all_stage_records['dpo'][row['id']]
    if before['end_to_end_correct'] != after['end_to_end_correct']:
        joint_to_dpo.append({'id': row['id'], 'task': row['task'],
                             'joint_correct': before['end_to_end_correct'],
                             'dpo_correct': after['end_to_end_correct'],
                             'joint_raw': before['action_trace']['raw'],
                             'dpo_raw': after['action_trace']['raw']})
assert joint_to_dpo and {r['task'] for r in joint_to_dpo} == {'joint'}

output = {
    'purpose': 'Existing raw-record verification and bounded encoding/parser variants; no new model scores.',
    'version': VERSION,
    'environment': {'python': sys.version, 'torch': str(torch.__version__),
                    'torch_git_version': str(torch.version.git_version), 'device': 'cpu'},
    'input_shapes': shapes, 'actual_json_pointers': pointer_log,
    'stage_results': stage_results, 'constant_payloads': constant_payloads,
    'encoding_variants': variants, 'availability_counterfactual': {
        'user': available['user'], 'available_system': available['system'],
        'available_answer': available['answer'], 'disabled_system': disabled['system'],
        'disabled_answer': disabled['answer']},
    'joint_to_dpo_changed_correctness': joint_to_dpo,
    'all_assertions_passed': True,
}
(HERE / 'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    'all_assertions_passed': True,
    'total_validation_rows': len(rows),
    'stage_summaries': {stage: {k: v[k] for k in ('by_task_recomputed', 'text_all', 'other_text')}
                        for stage, v in stage_results.items()},
    'constant_payloads': constant_payloads,
    'joint_to_dpo_changes': len(joint_to_dpo),
    'encoding_variants': variants,
}, ensure_ascii=False, indent=2))
