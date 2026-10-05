"""Bounded CPU review: fixed arithmetic, shape changes, counts, recorded NLLs.

No optimizer, training, corpus download, checkpoint load, GPU, or model evaluation.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
records = []
result = json.loads((OUT / 'moe-original-result.json').read_bytes())
pointers = []


def raw(pointer):
    obj = result
    for key in pointer.strip('/').split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    pointers.append(pointer)
    return obj


def log(name, **values):
    row = {'name': name, **values}
    records.append(row)
    print(json.dumps(row, ensure_ascii=False))


scope = {}
exec(compile((OUT / 'original-execution/fence-1.py').read_bytes(), 'raw-original-fence-1', 'exec'), scope)
ffn, x, up, down = [scope[name] for name in ('ffn', 'x', 'up', 'down')]


def scalar_ffn(row):
    hidden = [max(0, row[0] * a + row[1] * b) for a, b in zip([1, 0, 1], [0, 1, 1])]
    return [hidden[0] + hidden[2], hidden[1] + hidden[2]]


for repeats in (1, 2, 3):
    batch = torch.cat([x] * repeats, dim=0)
    expected = [scalar_ffn(row) for row in batch.tolist()]
    actual = ffn(batch)
    assert actual.tolist() == expected
    assert tuple(actual.shape) == (2 * repeats, 2)
    assert up.numel() + down.numel() == 12
    log('repetition', repeats=repeats, tokens=len(batch), output=actual.tolist(), weights=12,
        multiplications=len(batch) * 12, additions=len(batch) * 7, relu_elements=len(batch) * 3)

negative = torch.tensor([[-1.0, 2.0], [-2.0, -1.0], [0.0, 0.0]])
expected = [scalar_ffn(row) for row in negative.tolist()]
assert ffn(negative).tolist() == expected == [[1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]
log('relu-and-row-independence', input=negative.tolist(), pre_relu=(negative @ up).tolist(), output=expected)
changed = x.clone(); changed[1, 0] = -2
assert torch.equal(ffn(changed)[0], ffn(x)[0])
assert not torch.equal(ffn(changed)[1], ffn(x)[1])
log('one-row-change', output=ffn(changed).tolist(), unchanged_first_row=True)
wide_up = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
wide_down = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
assert torch.equal(torch.relu(x @ wide_up) @ wide_down, ffn(x))
assert wide_up.numel() + wide_down.numel() == 16
log('hidden-width-change', up_shape=list(wide_up.shape), down_shape=list(wide_down.shape), output_shape=[2, 2], weights=16)

base = '/results/variants/dense_active_top1'
config = raw(base + '/model/config')
model = TinyLM(ModelConfig(**config))
per_ffn = [sum(p.numel() for p in block.ffn.parameters()) for block in model.blocks]
total = sum(p.numel() for p in model.parameters())
formula_ffn = 64 * 256 + 256 + 256 * 64 + 64
formula_total = 264 * 64 + 128 * 64 + 2 * (4 * 64 * 64 + 4 * 64 + formula_ffn) + 2 * 64 + 264 * 64
assert per_ffn == [33088, 33088] and total == formula_total == raw(base + '/model/parameters') == 141568
log('dense-parameter-count', config=config, ffn_hidden=[block.ffn.up.out_features for block in model.blocks], per_layer_ffn=per_ffn, two_layer_ffn=sum(per_ffn), total=total, formula_ffn=formula_ffn, formula_total=formula_total)

training = {key: raw(base + '/training/' + key) for key in ('steps', 'requested_steps', 'optimizer_updates', 'skipped_updates', 'effective_tokens', 'all_requested_attempts_completed', 'status')}
assert training['steps'] == training['requested_steps'] == training['optimizer_updates'] == 180
assert training['skipped_updates'] == 0 and training['all_requested_attempts_completed'] is True and training['status'] == 'completed'
log('recorded-training-updates', **training)
for split, target_records, target_text in [('validation', 51, '2.29999'), ('test', 52, '2.32093')]:
    measure = {key: raw(base + '/heldout/' + split + '/' + key) for key in ('nll', 'nll_sum', 'effective_tokens', 'examples', 'records')}
    denominator = raw('/results/dataset/' + split)
    recalculated = measure['nll_sum'] / measure['effective_tokens']
    assert abs(recalculated - measure['nll']) <= 1e-12
    assert format(recalculated, '.5f') == target_text
    assert measure['records'] == denominator['records'] == target_records
    assert measure['effective_tokens'] > measure['examples'] > measure['records']
    assert abs(float(target_text) - recalculated) <= 5e-6
    log('recorded-heldout-' + split, **measure, dataset_sha256=denominator['sha256'], recomputed_nll=recalculated, prose_rounded=target_text, nll_units='natural-log cross-entropy per non-ignored next-token target; not per story')

provenance = {key: raw('/' + key) for key in ('schema_version', 'experiment_id', 'revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu')}
original_code = OUT / 'repository/architecture-experiment-revision.py'
original_sha = hashlib.sha256(original_code.read_bytes()).hexdigest()
assert original_sha == raw('/code_sha256/scripts~1course_experiments~1architecture.py')
current_matches = hashlib.sha256((ROOT / 'scripts/course_experiments/architecture.py').read_bytes()).hexdigest() == original_sha
log('recorded-provenance', **provenance, experiment_architecture_sha256=original_sha, current_architecture_matches=current_matches,
    explanation='Used git-show at recorded experiment revision; no result commentary or correction summaries inspected.')

environment = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'device': 'cpu', 'torch_threads': str(torch.get_num_threads()), 'reviewer_task': '/root/phase4_factual_coordinator/factual_15_1', 'scope': 'parameter initialization/count only; no model forward or reevaluation beyond tiny matrix examples'}
audit = {'environment': environment, 'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-15_1-independent/verify_cpu_and_measurements.py', 'inspected_json_pointers': sorted(set(pointers)), 'raw_json_sha256': hashlib.sha256((OUT / 'moe-original-result.json').read_bytes()).hexdigest(), 'checks': records, 'all_assertions_passed': True}
(OUT / 'cpu-verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print('All bounded CPU arithmetic, parameter-count, update-record and heldout-denominator assertions passed.')
