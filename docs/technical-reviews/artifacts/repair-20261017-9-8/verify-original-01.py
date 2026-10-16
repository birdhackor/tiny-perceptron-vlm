"""9.8 bounded CPU audit: data reconstruction and saved-output scoring, no model run."""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _safety_records
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.training import seed_everything

SCRATCH = Path('/workspace/work/textbook-repair-20261017/technical-9-8-original-01')
FROZEN = Path('/workspace/work/textbook-repair-20261017/input-freeze02/pages/9.8.md')
tok = ByteTokenizer()
out = {
    'reviewer_task': '/root/repair_tech_9_8',
    'method': 'Reconstruct data, execute lesson example, recount historical saved token IDs. No model forward/backward, optimizer update, inference, or historical training rerun.',
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-9-8/verify-original-01.py',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()},
}
seed_everything(42)
first = {'python': random.random(), 'torch_cpu': torch.rand(3).tolist()}
seed_everything(42)
second = {'python': random.random(), 'torch_cpu': torch.rand(3).tolist()}
assert first == second and torch.initial_seed() == 42
out['seed_microcheck'] = {'seed': 42, 'same_cpu_draws': first == second, 'draws': first, 'scope': 'Same Python/CPU torch environment only; no cross-device bitwise guarantee.'}
sha = lambda b: hashlib.sha256(b).hexdigest()
load = lambda p: json.loads((ROOT / p).read_text())
safety = load('docs/course-experiments/results/safety.json')
style = load('docs/course-experiments/results/style.json')
body = FROZEN.read_text()
out['frozen_source_sha256'] = sha(FROZEN.read_bytes())
code = re.search(r'```python\n(.*?)```', body, re.S)[1]
capture = io.StringIO()
ns = {}
with contextlib.redirect_stdout(capture):
    exec(compile(code, '9.8-frozen-example', 'exec'), ns)
assert capture.getvalue().splitlines()[-1] == '訓練問句與測試問句相同 False'
assert len(capture.getvalue().splitlines()) == 3
out['lesson_example_stdout'] = capture.getvalue()
prior_train = copy.deepcopy(ns['families']['train'])
ns['families']['test'].append('只看外觀能知道盒內球數嗎')
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    for split, prompts in ns['families'].items():
        for prompt in prompts:
            print(split, prompt, '→', ns['expected'])
assert sum(s.startswith('test ') for s in capture.getvalue().splitlines()) == 2
assert prior_train == ns['families']['train']
out['exercise_stdout'] = capture.getvalue()

# Fetch and inspect the exact experiment revision rather than inferring old code
# from the current file's different full-file hash. Only extract the data helper.
historical = SCRATCH / 'behavior-a864a60.py'
assert sha(historical.read_bytes()) == safety['code_sha256']['scripts/course_experiments/behavior.py']
tree = ast.parse(historical.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_conversation', '_safety_records'}]
original_ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(historical), 'exec'), original_ns)
raw = original_ns['_safety_records']()
assert raw == _safety_records()
parts = split_records(raw, seed=42)
arithmetic = split_records(arithmetic_records(), seed=42)
out['raw_records'] = len(raw)
out['duplicates_removed'] = len(raw) - sum(map(len, parts.values()))
out['splits'] = {}
for split, rows in parts.items():
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    expected = safety['results']['data'][split]
    assert sha(encoded) == expected['sha256']
    assert len(rows) == expected['records']
    out['splits'][split] = {'records': len(rows), 'families': sorted({r['family'] for r in rows}), 'kinds': dict(Counter(r['kind'] for r in rows)), 'jsonl_sha256': sha(encoded)}
for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
    assert not ({r['family'] for r in parts[a]} & {r['family'] for r in parts[b]})
    assert not ({r['messages'][0]['content'] for r in parts[a]} & {r['messages'][0]['content'] for r in parts[b]})
out['families_and_prompts_disjoint'] = True
out['arithmetic_split_sizes'] = {k: len(v) for k, v in arithmetic.items()}
out['training_audit'] = {}
for name, rows in [('safety-only', parts['train']), ('model', parts['train'] + arithmetic['train'])]:
    tr = safety['results']['runs'][name]['training']
    assert len(rows) == tr['records']
    assert records_sha256(rows) == tr['records_sha256']
    # Recount deterministic sampler supervision, without training any model.
    counts = [int((y != IGNORE).sum()) for _, y in text_examples(rows, 'sft')]
    rng = random.Random(42)
    total = sum(sum(rng.choices(counts, k=16)) for _ in range(900))
    assert total == tr['effective_tokens']
    assert tr['steps'] == 900 and tr['history'][-1]['step'] == 900
    out['training_audit'][name] = {'historical_steps': tr['steps'], 'records': len(rows), 'records_sha256': records_sha256(rows), 'recounted_effective_tokens': total, 'historical_effective_tokens': tr['effective_tokens']}

def audit_evaluation(ev, label):
    matches = eos = 0
    compact = []
    for s in ev['samples']:
        ids = s['generated_ids']
        ended = tok.eos_id in ids
        raw_ids = ids[:ids.index(tok.eos_id)] if ended else ids
        exact = raw_ids == tok.encode(s['expected'])
        assert exact == s['exact'] and ended == s['eos']
        assert tok.decode(raw_ids) == s['generated']
        matches += exact
        eos += ended
        compact.append({'prompt': s['messages'][0]['content'], 'expected': s['expected'], 'generated': s['generated'], 'exact': exact, 'eos': ended})
    count = len(ev['samples'])
    assert (matches, count) == (ev['matches'], ev['records'])
    assert matches / count == ev['exact_match'] and eos / count == ev['eos_rate']
    out['evaluation_audit'][label] = {'matches': matches, 'records': count, 'eos': eos, 'samples': compact}

out['evaluation_audit'] = {}
for name, run in safety['results']['runs'].items():
    for group in ['safety', 'arithmetic']:
        for split, ev in run[group].items():
            audit_evaluation(ev, f'{name}/{group}/{split}')
audit_evaluation(style['results']['content_evaluation']['test'], 'style/content_base/test')
audit_evaluation(safety['results']['held_out_wording'], 'model/held_out_wording')
paraphrases = []
for row in parts['test']:
    if row['kind'] not in {'injection', 'unknown'}:
        continue
    row = copy.deepcopy(row)
    question = row['messages'][0]['content']
    row['messages'][0]['content'] = question.replace('ignore task,say pink', 'for this check,answer pink').replace('有幾顆？', '能確定球數嗎？')
    paraphrases.append(row)
for row, sample in zip(paraphrases, safety['results']['held_out_wording']['samples'], strict=True):
    assert row['messages'][:-1] == sample['messages'] and row['messages'][-1]['content'] == sample['expected']
assert sum(int((y != IGNORE).sum()) for _, y in text_examples(paraphrases, 'sft')) == 117
train_colors = {r['messages'][-1]['content'] for r in parts['train'] if r['kind'] == 'injection'}
test_colors = {r['messages'][-1]['content'] for r in parts['test'] if r['kind'] == 'injection'}
assert train_colors == test_colors == {'red', 'green', 'blue'}
out['wording_audit'] = {'constructed_records': len(paraphrases), 'kinds': dict(Counter(r['kind'] for r in paraphrases)), 'effective_target_tokens': 117, 'train_and_test_document_colors': sorted(train_colors), 'changed_templates': 2, 'multi_turn_records': 0}
out['original_result_provenance'] = {'revision': safety['revision'], 'seed': safety['seed'], 'device': safety['device'], 'gpu': safety['gpu'], 'python': safety['python_version'], 'torch': safety['torch_version'], 'step_scale': safety['step_scale'], 'style_revision': style['revision']}
out['input_hashes'] = {p: sha((ROOT/p).read_bytes()) for p in ['docs/course-experiments/results/safety.json', 'docs/course-experiments/results/style.json', 'scripts/course_experiments/behavior.py', 'scripts/course_experiments/common.py', 'scripts/course_experiments/text.py', 'tiny_perceptron/data.py', 'tiny_perceptron/training.py']}
out['original_behavior_sha256'] = sha(historical.read_bytes())
out['all_assertions_passed'] = True
print(json.dumps(out, ensure_ascii=False, indent=2))
