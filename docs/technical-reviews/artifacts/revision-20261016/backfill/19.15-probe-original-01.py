"""19.15 independent bounded CPU inspection; no training or formal test entrypoint."""
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.inference import EFFORT, Product, check_manifest
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded
from tiny_perceptron.lesson_products.training import state_digest

ROOT = Path.cwd()
OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/backfill/19.15-execution-original-01.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_bytes())

def gold(world):
    if world['task'] == 'sum':
        return sum(world[k] for k in ['x', 'y', 'z'])
    state = {'A': world['a'], 'B': world['b']}
    for target, value in world['events']:
        if target == 'swap':
            state = {'A': state['B'], 'B': state['A']}
        else:
            state[target] = value
    return state[world['query']]

def record_metric(counter, key, answer, truth, family):
    counter[key]['count'] += 1
    counter[key]['correct'] += int(answer == truth)
    counter[key]['families'].add(family)

torch.set_num_threads(2)
freeze = load('docs/course-revision-20261016/freeze-14.json')
page = next(x for x in freeze['pages'] if x['page_id'] == '19.15')
body = lesson_slices(Path('course/chapters/19.md').read_bytes().decode())['19.15']
assert hashlib.sha256(body.encode()).hexdigest() == page['source_sha256'] == sha(page['snapshot'])
formal = load('docs/lesson-products/results/final-report-v1.json')
settings = load('docs/lesson-products/results/settings-freeze-v1.json')
pretest = load('docs/lesson-products/results/pretest-freeze-record-v1.json')
started = load('docs/lesson-products/results/final-report-v1.started.json')
assert sha('docs/lesson-products/results/settings-freeze-v1.json') == formal['settings_freeze_sha256'] == started['settings_freeze_sha256'] == pretest['settings_freeze_sha256']
assert settings['recorded_at'] < pretest['recorded_at'] < started['recorded_at'] < formal['recorded_at']
assert settings['test_outcomes_observed'] is False and pretest['test_marker_absent'] is True

sources = [
 page['snapshot'], 'tiny_perceptron/lesson_products/model.py', 'tiny_perceptron/lesson_products/data.py',
 'tiny_perceptron/lesson_products/training.py', 'tiny_perceptron/lesson_products/train.py',
 'tiny_perceptron/lesson_products/inference.py', 'tiny_perceptron/lesson_products/__main__.py',
 'tiny_perceptron/lesson_products/evaluation.py',
 'docs/lesson-products/data-v2-manifest.json', 'docs/lesson-products/data-v2-token-audit.json',
 'docs/lesson-products/jobs/train-b-v1.json', 'docs/lesson-products/results/initial-b-execution.json',
 'docs/lesson-products/results/settings-freeze-v1.json', 'docs/lesson-products/results/pretest-freeze-record-v1.json',
 'docs/lesson-products/results/final-report-v1.started.json', 'docs/lesson-products/results/final-report-v1.json',
 'docs/lesson-products/results/b-artifact-manifest.json',
 'docs/lesson-products/results/mechanism-fixed3-runner-result.json',
 'docs/lesson-products/results/mechanism-odd135-runner-result.json'
]
identity = {}
for name, expected in settings['source_sha256s'].items():
    path = 'tiny_perceptron/lesson_products/' + name
    identity[path] = {'expected': expected, 'observed': sha(path)}
    assert identity[path]['expected'] == identity[path]['observed']

# Inspect only B data and existing predictions; do not call evaluate_frozen.
data_path = Path('outputs/lesson-products/data-v2/b.jsonl')
all_rows = [json.loads(line) for line in data_path.open()]
data_manifest = load('docs/lesson-products/data-v2-manifest.json')
assert sha('docs/lesson-products/data-v2-manifest.json') == sha('outputs/lesson-products/data-v2/manifest.json') == settings['data_manifest_sha256']
assert sha(data_path) == data_manifest['files']['b.jsonl']['sha256']
assert len(all_rows) == data_manifest['files']['b.jsonl']['rows']
core_splits = {'train', 'calibration', 'policy', 'test'}
families = defaultdict(set)
lookup = {}
for row in all_rows:
    assert row['gold'] == gold(row['world'])
    if row['split'] in core_splits:
        families[row['family_id']].add(row['split'])
    assert (row['split'], row['id']) not in lookup
    lookup[row['split'], row['id']] = row
    if row['task'] == 'boxes':
        assert 0 <= row['world']['a'] <= 7 and 0 <= row['world']['b'] <= 7
        assert all(target == 'swap' or 0 <= value <= 7 for target, value in row['world']['events'])
        assert row['strata']['repeated_swap'] == (sum(target == 'swap' for target, _ in row['world']['events']) >= 2)
    else:
        assert all(0 <= row['world'][k] <= 6 for k in ['x', 'y', 'z'])
assert all(len(splits) == 1 for splits in families.values())
raw = Path('outputs/lesson-products/final-report-v1.predictions.jsonl')
assert sha(raw) == formal['predictions']['sha256']
metrics = defaultdict(lambda: {'count': 0, 'correct': 0, 'families': set()})
seen = Counter()
for line in raw.open():
    item = json.loads(line)
    if item['product'] != 'b':
        continue
    split = item['split']
    row = lookup[split, item['id']]
    assert item['gold'] == row['gold'] and item['family_id'] == row['family_id']
    seen[split, item['id']] += 1
    if split == 'test':
        scores = item['scores_by_depth']
    else:
        scores = {'5': item['raw_scores']}
    for depth, values in scores.items():
        assert len(values) == 19
        answer = max(range(len(values)), key=values.__getitem__)
        if split == 'test':
            assert answer == item['answers_by_depth'][depth]
        record_metric(metrics, (split, depth, row['task']), answer, row['gold'], row['family_id'])
        if row['task'] == 'boxes' and row['strata']['repeated_swap']:
            record_metric(metrics, (split, depth, 'repeated_swap'), answer, row['gold'], row['family_id'])
assert all(n == 1 for n in seen.values())
assert set(seen) == {key for key in lookup if key[0] in ['test', 'wording_shift', 'entity_shift', 'long_shift']}
recomputed = {}
for (split, depth, task), value in sorted(metrics.items()):
    observed = {'count': value['count'], 'correct': value['correct'], 'families': len(value['families']), 'percentage': value['correct']/value['count']*100}
    expected = formal['b'][depth][task] if split == 'test' else formal['shifts'][split][task]
    assert all(observed[k] == expected[k] for k in ['count', 'correct', 'families'])
    recomputed[f'{split}/depth{depth}/{task}'] = observed

bdir = Path('outputs/lesson-products/products/b')
b = Product(bdir, 'b', 'cpu')
manifest = check_manifest(bdir, 'b')
assert manifest == load('docs/lesson-products/results/b-artifact-manifest.json')
for name, expected in settings['products']['b'].items():
    assert sha(bdir / name) == expected
parameters = sum(p.numel() for p in b.model.parameters())
assert parameters == 170003
initial = {}
for seed, expected in [(7421, 'ecb489e7c102c2c6049e35e4fbfaa3e9e3d6cad9753c0566d9d96fdc88625bf3'), (8159, '4ad5df4b5f48c0250dad6f4493e0704c58bee694ad2c20ea61e6fa8aa7feccde')]:
    torch.manual_seed(seed)
    initial[str(seed)] = state_digest(LoopedSolver(b.config).state_dict())
    assert initial[str(seed)] == expected

checkpoints = {}
for mode, directory, receipt_path in [
 ('multi', bdir, 'docs/lesson-products/results/initial-b-execution.json'),
 ('fixed3', Path('outputs/lesson-products/mechanisms/fixed3'), 'docs/lesson-products/results/mechanism-fixed3-runner-result.json'),
 ('odd', Path('outputs/lesson-products/mechanisms/odd'), 'docs/lesson-products/results/mechanism-odd135-runner-result.json')
]:
    receipt = load(receipt_path)
    if 'execution' in receipt:
        receipt = receipt['execution']
    training = receipt['training']
    checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
    assert {'model','optimizer','sampler_rng','torch_rng','cuda_rng','config','metadata'} <= set(checkpoint)
    assert checkpoint['optimizer']['state'] and checkpoint['cuda_rng']
    assert checkpoint['metadata']['step'] == training['last_step']
    assert checkpoint['metadata']['mode'] == mode
    assert checkpoint['metadata']['initial_weights_sha256'] == initial[str(training['seed'])]
    for name in ['last.pt', 'best.pt', 'model.pt', 'training-receipt.json', 'tokenizer.json', 'config.json']:
        assert sha(directory / name) == receipt['client_artifacts'][name]['sha256']
    assert training['resume_verification']['exact_next_step'] is True
    checkpoints[mode] = {'last_step': checkpoint['metadata']['step'], 'seed': training['seed'], 'keys': list(checkpoint), 'optimizer_state_entries': len(checkpoint['optimizer']['state']), 'cuda_rng_states': len(checkpoint['cuda_rng']), 'gpu': receipt['gpu_name'], 'runner_environment': training['environment'], 'resume_receipt': training['resume_verification'], 'source': receipt_path}
    if mode != 'multi':
        assert sha(directory / 'last.pt') == settings['mechanisms'][mode]['last.pt']
    else:
        best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        assert best['metadata']['step'] == training['selected_step'] == 1000
        assert state_digest(best['model']) == state_digest(b.model.state_dict())
        assert training['random_initialization'] is True and training['resume_source'] is None
assert checkpoints['fixed3']['last_step'] == checkpoints['odd']['last_step'] == 1500
assert EFFORT == {'low':1,'medium':3,'high':5}

example = '兩盒追蹤;A=2;B=6;事件=交換;查詢=B'
encoded = b.tokenizer.encode(example)
assert Tokenizer.load(bdir / 'tokenizer.json').encode(example) == encoded
calls = {str(d): b.solve(example, d) for d in [1,3,5]}
assert all(value['loops'] == int(depth) for depth,value in calls.items())
with torch.inference_mode():
    logits = b.model(padded([encoded], b.config.max_length), loops=5, all_depths=True)
assert list(logits.shape) == [1,5,19]
labels = torch.tensor([2])
losses = F.cross_entropy(logits.flatten(0,1), labels[:,None].expand(-1,5).reshape(-1), reduction='none').view(-1,5)
mean = (losses * torch.tensor([1,1,1,1,1])).sum(-1).mean()/5
assert torch.allclose(mean, losses.mean(), rtol=0, atol=0)
leaf = logits.clone().requires_grad_(True)
F.cross_entropy(leaf.flatten(0,1), labels[:,None].expand(-1,5).reshape(-1)).backward()
gradient_nonzero = [bool(torch.any(leaf.grad[:,d] != 0)) for d in range(5)]
assert all(gradient_nonzero)

result = {
 'reviewer_task':'/root/backfill_tech_19_15', 'command':'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.15-probe-original-01.py',
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','platform':platform.platform()},
 'source_sha256':{p:sha(p) for p in sources}, 'source_identity_against_pretest_freeze':identity,
 'frozen_section_sha256':page['source_sha256'],
 'recomputed_from_existing_predictions':recomputed,
 'raw_predictions':{'path':str(raw),'sha256':sha(raw),'bytes':raw.stat().st_size},
 'b_data':{'path':str(data_path),'sha256':sha(data_path),'rows':len(all_rows),'gold_matches_independent_interpreter':True,'core_family_overlap':0,'scored_rows':len(seen)},
 'settings_protocol':settings['protocol'],
 'chronology':{'settings':settings['recorded_at'],'pretest':pretest['recorded_at'],'test_started':started['recorded_at'],'final':formal['recorded_at']},
 'parameters':parameters,'initial_state_digests':initial,'checkpoints':checkpoints,
 'inference_state_dict_keys':list(torch.load(bdir / 'model.pt', map_location='cpu', weights_only=True)),
 'cpu_example':{'state':example,'gold':2,'calls':calls,'token_count':len(encoded)},
 'equal_multidepth_ce':{'logits_shape':list(logits.shape),'labels':[2]*5,'losses':losses.tolist(),'mean':float(mean),'gradient_nonzero_by_depth':gradient_nonzero,'optimizer_updates_performed':0},
 'scope':'Original bounded CPU inspection and independent recount of saved raw predictions. Existing L4 receipts establish historical training/resume observations; reviewer did not rerun GPU, training, formal test, or network publication.'
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'saved':str(OUT.relative_to(ROOT)), 'parameters':parameters,'scored_rows':len(seen),'raw_sha256':sha(raw),'cpu_calls':calls,'status':'all assertions passed'},ensure_ascii=False))
