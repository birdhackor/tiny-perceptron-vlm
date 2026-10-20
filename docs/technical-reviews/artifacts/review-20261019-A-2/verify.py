"""獨立重算公告實驗紀錄，執行教材變例與資料處理，不進行訓練。"""
import ast
import contextlib
import copy
import difflib
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import Context, new_lm, records_sha256, split_records, text_examples
from tiny_perceptron.data import ByteTokenizer, IGNORE


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_code(code):
    out = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(out):
        exec(compile(code, '<announcement>', 'exec'), namespace)
    return {'stdout': out.getvalue(), 'source': namespace['source'], 'prompt': namespace['prompt']}


result_path = ROOT / 'docs/course-experiments/results/rag.json'
record = json.loads(result_path.read_bytes())
r = record['results']
freeze = ROOT / 'docs/course-review-20261019/freeze-01/sources/A.2.md'
body = freeze.read_text()
code = body.split('```python\n', 1)[1].split('```', 1)[0]
base = run_code(code)
red = run_code(code.replace('青街8號', '紅街9號'))
renumbered = run_code(code.replace('公告1', '公告2'))
assert base['source'] == {'id': '公告1', 'date': '2027年3月1日', 'address': '青街8號'}
assert '青街8號' in base['prompt'] and base['stdout'].endswith('由資料確定的答案 青街8號\n')
assert red['source']['id'] == '公告1' and red['source']['date'] == base['source']['date']
assert red['stdout'] == base['stdout'].replace('青街8號', '紅街9號')
assert renumbered['stdout'] == base['stdout'].replace('公告1', '公告2')
assert {n.func.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} == {'print'}

rows = r['samples']
tok = ByteTokenizer()
modes = ['correct_context', 'changed_address_context', 'changed_source_context']
by_mode = {mode: {x['family']: x for x in rows if x['mode'] == mode} for mode in modes}
keys = set(by_mode[modes[0]])
assert len(keys) == 12 and all(set(by_mode[m]) == keys for m in modes)
recomputed = {}
compact = []
for mode in modes:
    correct = set()
    for key in sorted(keys):
        row = by_mode[mode][key]
        sample = row['generation']['samples'][0]
        ids = sample['generated_ids']
        raw = ids[:-1] if ids and ids[-1] == tok.eos_id else ids
        decoded = bytes(t - 8 for t in raw if t >= 8).decode('utf-8', errors='replace')
        invalid = [t for t in raw if t < 8]
        assert decoded == sample['generated'] and invalid == sample['invalid_special_tokens']
        fact = row['context_fact']
        expected = fact['address'] + '[' + fact['source'] + ']'
        assert expected == row['expected']
        passed = decoded == expected and not invalid
        assert passed == row['fact_answer_correct']
        assert row['documents'] == [{'id': fact['source'], 'text': fact['key'] + ' address=' + fact['address']}]
        expected_prompt = 'Docs:\n[' + fact['source'] + '] ' + fact['key'] + ' address=' + fact['address'] + '\nFind:' + key + '\nReply:address[source] or UNKNOWN'
        assert row['generation']['messages'] == [{'role': 'user', 'content': expected_prompt}]
        baseline_fact = by_mode['correct_context'][key]['context_fact']
        different_fields = [f for f in baseline_fact if baseline_fact[f] != fact[f]]
        assert different_fields == ({'correct_context': [], 'changed_address_context': ['address'], 'changed_source_context': ['source']}[mode])
        assert row['generation']['candidate_count'] == 1 and row['generation']['temperature'] == 0.0
        if passed:
            correct.add(key)
        compact.append({'family': key, 'mode': mode, 'expected': expected, 'generated': decoded, 'correct': passed})
    metric = r['metrics'][mode]['fact_answer_correct']
    assert metric['numerator'] == len(correct) and metric['denominator'] == len(keys)
    assert metric['rate'] == len(correct) / len(keys)
    recomputed[mode] = {'correct': len(correct), 'total': len(keys), 'correct_families': sorted(correct)}
for mode in modes[1:]:
    both = set(recomputed[mode]['correct_families']) & set(recomputed['correct_context']['correct_families'])
    metric = r['paired_counterfactuals'][mode]['both_answers_correct']
    assert metric == {'numerator': len(both), 'denominator': 12, 'rate': len(both) / 12}
    recomputed[mode]['both_correct'] = len(both)
    recomputed[mode]['both_correct_families'] = sorted(both)
assert [recomputed[m]['correct'] for m in modes] == [5, 7, 5]
assert [recomputed[m]['both_correct'] for m in modes[1:]] == [5, 2]
for key, mode, expected in [('K017', 'changed_address_context', 'C3[D1]'), ('K028', 'changed_source_context', 'B9[D0]')]:
    assert by_mode[mode][key]['generation']['samples'][0]['generated'] == expected
assert by_mode['correct_context']['K017']['generation']['samples'][0]['generated'] == 'C2[D1]'
assert by_mode['correct_context']['K028']['generation']['samples'][0]['generated'] == 'B9[D9]'

splits = app._rag_splits(record['seed'])
summary = {}
for name, records in splits.items():
    families = {row['family'] for row in records}
    assert all(Counter(row['family'] for row in records)[f] == 8 for f in families)
    assert all(doc['text'].split()[0] in families for row in records for doc in row['documents'])
    summary[name] = {'records': len(records), 'families': len(families), 'sha256': records_sha256(records)}
    assert summary[name] == r['split'][name]
assert not ({x['family'] for x in splits['train']} & {x['family'] for x in splits['test']})
assert not ({x['family'] for x in splits['validation']} & {x['family'] for x in splits['test']})
assert not ({x['family'] for x in splits['train']} & {x['family'] for x in splits['validation']})
rng = random.Random(record['seed'] + 100)
facts = []
for key in sorted(keys):
    source = f'D{rng.randrange(10)}'
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    fact = {'family': key, 'key': key, 'address': address, 'source': source}
    assert fact == by_mode['correct_context'][key]['fact']
    facts.append(fact)
assert records_sha256(facts) == r['test_facts_sha256']
original_splits = records_sha256(splits)
for fact in facts:
    prior = copy.deepcopy(fact)
    cases = app._rag_counterfactual_contexts(fact)
    assert fact == prior
    for mode, case in cases.items():
        assert case['fact'] == by_mode[mode][fact['key']]['context_fact']
assert original_splits == records_sha256(splits)
icl = split_records(app._icl_records(), record['seed'])
training_records = splits['train'] + icl['train']
assert len(training_records) == r['training']['records'] == 894
assert records_sha256(training_records) == r['training']['records_sha256']
examples = text_examples(training_records, mode='sft', max_length=160)
for row, (x, labels) in zip(training_records, examples, strict=True):
    target = tok.encode(row['messages'][-1]['content']) + [tok.eos_id]
    assert labels[labels != IGNORE].tolist() == target
    assert x.dtype == labels.dtype == torch.int64 and len(x) == len(labels) <= 160

ctx = Context('cpu', Path(__file__).parent, Path(__file__).parent, ROOT / 'assets/training', seed=42)
model = new_lm(ctx, width=64, layers=2, max_length=160, heads=2, backend='sdpa')
def state_hash():
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
before = state_hash()
probe = []
for mode in ['correct_context', 'changed_address_context', 'changed_source_context']:
    generated = app._sample(model, by_mode[mode]['K017']['generation']['messages'], ctx, tokens=2)
    probe.append({'mode': mode, 'generated': generated['samples'][0]['generated']})
after = state_hash()
assert before == after
assert model.description()['parameters'] == r['model']['parameters'] == 143616

help_run = subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/course_experiments/run.py', '--help'], cwd=ROOT, capture_output=True, text=True, check=True)
assert '--experiment' in help_run.stdout and '--device {cpu,cuda}' in help_run.stdout
artifact_names = sorted(a['path'] for a in record['artifacts'])
expected_artifacts = ['dataset.json', 'icl-dataset.json', 'retrieval-corpus.json', 'generations.json', 'model.pt']
checkpoint_steps = [i for i in range(1, 1001) if i % max(1, 1000 // 4) == 0]
expected_artifacts += [f'model-step-{i}.pt' for i in checkpoint_steps]
assert all(name in artifact_names for name in expected_artifacts)
code_paths = ['scripts/course_experiments/applications.py', 'scripts/course_experiments/common.py', 'scripts/course_experiments/run.py', 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']
code_hashes = {p: digest((ROOT / p).read_bytes()) for p in code_paths}
code_version_comparison = {p: {'current_sha256': code_hashes[p], 'recorded_sha256': record['code_sha256'][p], 'same': code_hashes[p] == record['code_sha256'][p]} for p in code_paths}
application_path = 'scripts/course_experiments/applications.py'
revision = 'dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17'
historical = subprocess.run(['git', 'show', revision + ':' + application_path], capture_output=True, check=True).stdout
assert digest(historical) == record['code_sha256'][application_path]
current = (ROOT / application_path).read_bytes()
def functions(raw):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
old_functions, current_functions = functions(historical), functions(current)
changed_functions = [name for name in sorted(old_functions.keys() | current_functions.keys()) if old_functions.get(name) != current_functions.get(name)]
assert changed_functions == ['_sample']
old_sample = next(node for node in ast.parse(historical).body if isinstance(node, ast.FunctionDef) and node.name == '_sample')
new_sample = next(node for node in ast.parse(current).body if isinstance(node, ast.FunctionDef) and node.name == '_sample')
old_return, new_return = old_sample.body[-1].value, new_sample.body[-1].value
assert isinstance(old_return, ast.Dict) and isinstance(new_return, ast.Dict)
assert ast.dump(old_return.keys[0]) == ast.dump(new_return.keys[0]) == "Constant(value='messages')"
assert isinstance(old_return.values[0], ast.Name) and old_return.values[0].id == 'messages'
assert ast.unparse(new_return.values[0]) == 'copy.deepcopy(messages)'
old_return.values[0] = new_return.values[0]
assert ast.dump(old_sample, include_attributes=False) == ast.dump(new_sample, include_attributes=False)
provenance = {'recorded_source_found_in_commit': revision, 'recorded_source_sha256': digest(historical), 'changed_functions': changed_functions, 'rag_functions_identical': True, 'sample_semantic_change': '回傳 messages 改用 copy.deepcopy(messages)；生成與評分路徑其餘 AST 相同。RAG 呼叫未在回傳後變更傳入的 messages。', 'diff': list(difflib.unified_diff(historical.decode().splitlines(), current.decode().splitlines(), fromfile='recorded', tofile='current', lineterm=''))}
render_record = next(x for x in json.loads((ROOT / 'docs/course-review-20261019/figure-renders-04.json').read_bytes())['records'] if x['page_id'] == 'A.2')
assert digest((ROOT / render_record['png']['path']).read_bytes()) == render_record['png']['sha256']
print(json.dumps({
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-2/verify.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'cpu'},
    'status': 'all_assertions_passed',
    'application_code_provenance': provenance, 'figure_render_receipt': render_record,
    'record_path': str(result_path.relative_to(ROOT)), 'record_sha256': digest(result_path.read_bytes()),
    'source_sha256': digest(freeze.read_bytes()), 'code_sha256': code_hashes, 'code_version_comparison': code_version_comparison,
    'announcement_examples': {'original': base, 'changed_address': red, 'changed_source': renumbered},
    'recomputed': recomputed, 'per_question': compact, 'split_summary': summary,
    'training_records': len(training_records), 'mask_checks': len(examples),
    'fresh_test_facts_sha256': records_sha256(facts), 'split_unchanged_after_interventions': True,
    'random_initialization_inference_probe': {'before_parameters_sha256': before, 'after_parameters_sha256': after, 'samples': probe, 'scope': '僅驗證無更新推論；未訓練模型的輸出不作準確率證據。'},
    'cli_help': help_run.stdout, 'recorded_artifact_names': artifact_names, 'checkpoint_steps': checkpoint_steps,
    'recorded_training': {'seed': record['seed'], 'steps': r['training']['steps'], 'effective_tokens': r['training']['effective_tokens'], 'gpu': record['gpu'], 'python': record['python_version'], 'torch': record['torch_version'], 'elapsed_seconds': record['elapsed_seconds'], 'timing_scope': record['timing_scope']},
    'scope': '從保存的逐題生成與 token 重算分數；未載入歷史權重、未重跑訓練或 GPU 計時；檔案清單以原始實驗紀錄及現行產生程式核對，未讀取被限制的 outputs。'
}, ensure_ascii=False, indent=2))
