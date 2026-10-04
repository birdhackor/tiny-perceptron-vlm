import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron import capstone as c

PROOF = Path(__file__).resolve().parent
started = time.perf_counter()
torch.set_num_threads(2)
def read(path):
    return json.loads((ROOT / path).read_text())
def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
def parse(trace):
    if not trace['eos']:
        return {'status': 'invalid', 'reason': 'unterminated_generation'}
    text = trace['raw']
    for label, offset in [('DIRECT:', 7), ('ASK:', 4)]:
        if text.startswith(label) and len(text) > offset:
            return {'status': label[:-1].lower(), 'content': text[offset:]}
    match = re.fullmatch(r'TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})', text)
    if match:
        return dict(status='tool', name=match[1], a=int(match[2]), b=int(match[3]))
    return {'status': 'invalid', 'reason': 'malformed_action'}
def raw(ids):
    return bytes(i - 8 for i in ids if i >= 8).decode('utf-8', errors='replace')
def check_trace(trace, row):
    ids = trace['generated_ids']
    assert trace['raw'] == raw(ids)
    assert trace['eos'] == (bool(ids) and ids[-1] == 2)
    assert (trace['stop_reason'] == 'eos') == trace['eos']
    assert all(i >= 8 for i in ids[:-1])
    assert trace['prompt_ids'] == c.prompt_ids(row)

body = (PROOF / 'source-original.md').read_text()
example = re.search(r'```python\n(.*?)\n```', body, re.S)[1]
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(example, '19.7-literal-code', 'exec'), {})
assert "'result': '3'" in printed.getvalue()
assert 'calculator_unavailable' in printed.getvalue()
request = 'TOOL:calculator:1+2'
assert c.TOK.encode(request) == [x + 8 for x in request.encode('utf-8')]
unknown = c.parse_action({'raw': request.replace('calculator', 'unknown'), 'eos': True})
assert unknown == {'status': 'tool', 'name': 'unknown', 'a': 1, 'b': 2}
assert c.calculator_runtime(unknown) == {'status': 'error', 'reason': 'tool_not_allowlisted'}
boundary = []
for a, b, expected in [(0, 0, 'ok'), (999, 999, 'ok'), (-1, 0, 'error'), (1000, 0, 'error'), (True, 2, 'error'), (1.0, 2, 'error')]:
    result = c.calculator_runtime(dict(status='tool', name='calculator', a=a, b=b))
    assert result['status'] == expected
    boundary.append({'a': a, 'b': b, 'result': result})

data_path = 'docs/course-experiments/capstone-evidence/deployment/data.json'
data = read(data_path)
splits, manifest = c.build_dataset()
assert splits == data['splits'] and manifest == data['manifest']
families = {name: {r['family'] for r in rows} for name, rows in splits.items()}
assert not any(families[a] & families[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
assert 'numbers:1:2' in families['test'] and 'numbers:1:2' not in families['train'] | families['validation']
for name, rows in splits.items():
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == manifest['sha256'][name]

stages = {}
for name, split in [('pretrain', 'validation'), ('sft', 'validation'), ('joint', 'validation'), ('dpo', 'validation'), ('test-joint', 'test')]:
    path = f'docs/course-experiments/capstone-evidence/{name}/validation.json' if split == 'validation' else 'docs/course-experiments/capstone-evidence/deployment/test-joint.json'
    report = read(path)
    inputs = {row['id']: row for row in splits[split]}
    assert {r['id'] for r in report['records']} == set(inputs)
    assert len(report['records']) == len(inputs) == report['count']
    counts = {}
    examples = []
    for record in report['records']:
        row = inputs[record['id']]
        trace = record['action_trace']
        check_trace(trace, row)
        parsed = parse(trace)
        assert parsed == record['parsed_action']
        assert row['answer'] == record['expected_action']
        correct_action = trace['eos'] and trace['raw'] == row['answer']
        answer = None
        if parsed['status'] in ('direct', 'ask'):
            answer = parsed['content']
            assert record['runtime'] is None and record['final_trace'] is None
        elif parsed['status'] == 'tool':
            assert parsed['name'] == 'calculator' and row['available']
            actual_result = str(parsed['a'] + parsed['b'])
            assert record['runtime'] == {'status': 'ok', 'result': actual_result}
            followup = dict(row, image=None, audio=None, user=f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{actual_result}。請回答。")
            check_trace(record['final_trace'], followup)
            final = parse(record['final_trace'])
            if final['status'] == 'direct':
                answer = final['content']
        if row['task'] == 'calculator':
            op = re.fullmatch(r'TOOL:calculator:(\d+)\+(\d+)', row['answer'])
            expected = str(int(op[1]) + int(op[2]))
        else:
            expected = row['answer'].split(':', 1)[1]
        end_correct = correct_action and answer == expected
        assert expected == record['expected_final'] and answer == record['answer']
        assert correct_action == record['action_correct'] and end_correct == record['end_to_end_correct']
        stat = counts.setdefault(row['task'], dict(count=0, action_correct=0, runtime_ok=0, final_eos_and_correct=0, end_to_end_correct=0, no_runtime=0))
        stat['count'] += 1
        stat['action_correct'] += int(correct_action)
        stat['end_to_end_correct'] += int(end_correct)
        stat['no_runtime'] += int(record['runtime'] is None)
        stat['runtime_ok'] += int(record['runtime'] is not None and record['runtime']['status'] == 'ok')
        stat['final_eos_and_correct'] += int(record['final_trace'] is not None and record['final_trace']['eos'] and answer == expected)
        if row['task'] in ('calculator', 'tool_return') and (row['family'] in ('numbers:4:4', 'numbers:0:1', 'numbers:1:2')):
            examples.append(dict(id=row['id'], task=row['task'], user=row['user'], expected_action=row['answer'], expected_final=expected, action=trace['raw'], action_eos=trace['eos'], runtime=record['runtime'], final=None if record['final_trace'] is None else record['final_trace']['raw'], final_eos=None if record['final_trace'] is None else record['final_trace']['eos'], correct=end_correct))
    for task, stat in counts.items():
        assert {k:stat[k] for k in ['count','action_correct','end_to_end_correct']} == report['by_task'][task]
    assert sum(s['action_correct'] for s in counts.values()) == report['action_correct']
    assert sum(s['end_to_end_correct'] for s in counts.values()) == report['end_to_end_correct']
    stages[name] = dict(path=path, sha256=sha(path), audited_records=len(inputs), tool_counts={k:counts[k] for k in ['calculator','unavailable','tool_return']}, examples=examples)

model = object()
row = next(r for r in splits['test'] if r['task'] == 'calculator' and r['family'] == 'numbers:1:2')
calls = []
def injected_generation(actual_model, actual_row, max_new_tokens):
    assert actual_model is model
    calls.append(actual_row['user'])
    return {'raw': row['answer'] if len(calls) == 1 else 'DIRECT:4', 'eos': True}
with patch.object(c, 'generate_trace', injected_generation):
    probe = c.run_assistant(model, row)
assert calls[1] == '原題：1+2。計算器回報：3。請回答。'
assert probe['runtime']['result'] == '3' and probe['answer'] == '4'

training = {}
previous = None
for name in ['pretrain', 'sft', 'joint', 'preference']:
    path = f'docs/course-experiments/results/capstone_{name}.json'
    full = read(path); result = full['results']
    assert result['schedule_completed'] and result['steps'] == result['requested_steps']
    assert result['data_manifest'] == manifest and result['seed'] == 42
    assert result['parent_checkpoint_sha256'] == previous
    previous = result['inference_export']['sha256']
    assert all(sha(p) == h for p,h in result['code_sha256'].items())
    training[name] = {'path':path, 'sha256':sha(path), **{k:full[k] for k in ['revision','device','seed','torch_version','python_version','gpu','elapsed_seconds','timing_scope']}, **{k:result[k] for k in ['stage','requested_steps','steps','seconds','effective_tokens','objective','parent_checkpoint_sha256','inference_export','test_evaluated']}}
assert read(stages['test-joint']['path'])['checkpoint_sha256'] == training['joint']['inference_export']['sha256']
proof = dict(environment={'python':sys.version.split()[0], 'torch':torch.__version__, 'device':'cpu', 'cuda_available':torch.cuda.is_available(), 'torch_threads':torch.get_num_threads()}, literal_example_stdout=printed.getvalue(), ascii_request={'text':request,'utf8_bytes':len(request.encode()),'byte_tokens':len(c.TOK.encode(request))}, unknown_request={'parsed':unknown,'runtime':c.calculator_runtime(unknown)}, runtime_boundaries=boundary, dataset={'path':data_path,'sha256':sha(data_path),'version':manifest['version'],'seed':manifest['seed'],'counts':manifest['counts'],'split_sha256':manifest['sha256'],'family_intersections':0,'held_out_1_2':True}, stages=stages, training_records=training, orchestration_probe={'generation':'explicit scripted trace injection; no trained-model generation','same_model_object':True,'calls':calls,'runtime':probe['runtime'],'answer':probe['answer']}, limitations='Independent CPU audit of original fixed-seed L4 records, not GPU training or benchmark replication; no new model/data downloads, no optimizer updates.', audit_seconds=time.perf_counter()-started)
(PROOF / 'audit-results.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'environment':proof['environment'],'audited_records':sum(s['audited_records'] for s in stages.values()),'tool_counts':{k:v['tool_counts'] for k,v in stages.items()},'literal_example_stdout':proof['literal_example_stdout'],'ascii_request':proof['ascii_request'],'orchestration_probe':proof['orchestration_probe'],'audit_seconds':proof['audit_seconds'],'assertions':'all passed'},ensure_ascii=False,indent=2))
