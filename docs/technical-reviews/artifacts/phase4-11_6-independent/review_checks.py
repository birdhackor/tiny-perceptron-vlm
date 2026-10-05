"""Independent CPU checks: original numeric fence, counts and raw historical samples.

Does not train, load model checkpoints, save neural weights or fetch model/data assets.
Only the named raw JSON pointers below are accessed; annotation fields are excluded.
"""
import contextlib
import hashlib
import io
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.modalities import _sequence, _vision_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene

A = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
data_path = ROOT / 'docs/course-experiments/results/vqa.json'
original = data_path.read_bytes()
d = json.loads(original)
pointers = []

def read(pointer):
    pointers.append(pointer)
    value = d
    for key in pointer.strip('/').split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value

result = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
          'torch_git_version': str(torch.version.git_version), 'device': 'cpu',
          'cuda_build': str(torch.version.cuda), 'cuda_available': str(torch.cuda.is_available())},
          'original_vqa_sha256': hashlib.sha256(original).hexdigest()}
namespace = {}
stdout = io.StringIO()
fence = (A / 'original-execution/fence-1.py').read_bytes()
with contextlib.redirect_stdout(stdout):
    exec(compile(fence, '11.6:original-fence-1', 'exec'), namespace)
assert [sum(x.values()) for x in namespace['plans'].values()] == [10000, 10000]
only_changed = {'描述對齊': 2000, '問答微調': 6000}
balanced = {'描述對齊': 2000, '問答微調': 8000}
assert sum(only_changed.values()) != 10000 and sum(balanced.values()) == 10000
result['budget_fence'] = {'original_stdout': stdout.getvalue(), 'only_alignment_changed': sum(only_changed.values()),
                          'changed_budget_check': sum(only_changed.values()) == 10000,
                          'balanced': sum(balanced.values()), 'balanced_check': sum(balanced.values()) == 10000}

ctx = SimpleNamespace(device='cpu', seed=read('/seed'))
splits = _vision_records(('shape?', 'color?'))
raw_splits = {}
for name in ['train', 'validation', 'test']:
    base = '/results/data/splits/' + name
    rows = read(base + '/records')
    count = read(base + '/count')
    sha = read(base + '/sha256')
    assert rows == splits[name] and count == len(rows)
    assert hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == sha
    raw_splits[name] = rows
families = {k: {r['family'] for r in v} for k, v in raw_splits.items()}
assert all(not families[a] & families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
result['data'] = {'counts': {k: len(v) for k,v in raw_splits.items()},
                  'family_counts': {k: len(v) for k,v in families.items()}, 'family_overlap': False}

def expected_batch_counts(records, steps, replay_gate):
    return [count_batch(records, i, replay_gate) for i in range(steps)]

def count_batch(records, i, replay_gate):
    rng = random.Random(ctx.seed + i)
    total = 0
    for _ in range(4):
        # The all branch passes a nonempty replay list even with ratio=0.
        # The implementation consumes this Bernoulli draw before choosing QA.
        if replay_gate:
            rng.random()
        total += len(rng.choice(records)['answer'].encode('utf-8')) + 1
    return total

training = {}
for name, base, records in [
    ('alignment', '/results/two_stage_alignment_training', _vision_records()['train']),
    ('two_stage_qa', '/results/variants/all/training', raw_splits['train']),
    ('direct_qa', '/results/variants/direct_vqa/training', raw_splits['train']),
]:
    selected = {k: read(base + '/' + k) for k in ['history','steps','effective_tokens','effective_targets',
               'trainable_parameters','weights_changed','nonzero_gradient_seen','config','modal_config']}
    history = selected.pop('history')
    counts = [r['effective_targets'] for r in history]
    assert len(history) == selected['steps']
    assert [r['step'] for r in history] == list(range(1, len(history)+1))
    assert sum(counts) == selected['effective_tokens'] == selected['effective_targets']
    assert counts == expected_batch_counts(records, selected['steps'], name == 'two_stage_qa')
    assert selected['weights_changed'] and selected['nonzero_gradient_seen']
    training[name] = {**selected, 'recomputed_target_total': sum(counts), 'last_batch': counts[-1],
                      'deterministic_batch_counts_match_every_step': True}
target = training['alignment']['effective_tokens'] + training['two_stage_qa']['effective_tokens']
direct = training['direct_qa']['effective_tokens']
assert target == read('/results/direct_vs_two_stage_budget/target_effective_tokens')
assert direct == read('/results/direct_vs_two_stage_budget/direct_effective_tokens')
assert target == read('/results/variants/direct_vqa/training/requested_effective_tokens')
assert direct - target == read('/results/variants/direct_vqa/training/budget_excess')
assert direct - training['direct_qa']['last_batch'] < target <= direct
assert 0 <= direct - target <= 40
assert read('/results/direct_vs_two_stage_budget/matched_to_within_final_batch') is True
result['training'] = training
result['budget_history'] = {'two_stage_effective_targets': target, 'direct_effective_targets': direct,
                           'excess': direct-target, 'relative_excess': (direct-target)/target,
                           'two_stage_updates': training['alignment']['steps'] + training['two_stage_qa']['steps'],
                           'direct_updates': training['direct_qa']['steps'],
                           'direct_before_last_batch': direct-training['direct_qa']['last_batch'],
                           'unit': 'repeated supervised UTF-8-byte targets plus answer EOS, excluding ignored prompts/modal slots',
                           'compute_and_update_counts_equal': False}

tok = ByteTokenizer()
result['samples'] = {}
for name in ['all', 'direct_vqa']:
    base = '/results/variants/' + name + '/test'
    samples = read(base + '/samples')
    fields = {k: read(base + '/' + k) for k in ['examples','correct','exact_match','effective_tokens','eos_rate','generation_errors','skipped']}
    assert len(samples) == len(raw_splits['test']) == fields['examples'] == 12
    recalculated = []
    for row, s in zip(raw_splits['test'], samples, strict=True):
        assert (s['family'],s['question'],s['target']) == (row['family'],row['question'],row['answer'])
        ids = s['generated_ids']
        raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        match = raw == tok.encode(row['answer'])
        eos = tok.eos_id in ids
        assert match == s['exact_match'] and eos == s['eos']
        assert tok.decode(raw) == s['generated']
        recalculated.append({'row': s['row'], 'question': s['question'], 'target': s['target'],
                             'generated': s['generated'], 'exact_match': match, 'eos': eos})
    correct = sum(x['exact_match'] for x in recalculated)
    effective = sum(len(tok.encode(x['answer'])) + 1 for x in raw_splits['test'])
    assert correct == fields['correct'] and correct/12 == fields['exact_match']
    assert effective == fields['effective_tokens']
    assert sum(x['eos'] for x in recalculated)/12 == fields['eos_rate']
    assert fields['generation_errors'] == 0 and fields['skipped'] == []
    result['samples'][name] = {'correct':correct,'examples':12, 'answer_target_denominator':effective,
                              'eos_correct': sum(x['eos'] for x in recalculated), 'rows':recalculated}
direct_fields = read('/results/variants/direct_vqa')
assert not any(k in direct_fields for k in ['text_before','text_after','text_validation'])
result['direct_text_holdout_fields_present'] = False
left = result['samples']['all']['rows']
right = result['samples']['direct_vqa']['rows']
paired = {'all_only_correct':0, 'direct_only_correct':0, 'both_correct':0, 'both_incorrect':0}
for a,b in zip(left,right,strict=True):
    key = ('both_correct' if a['exact_match'] else 'direct_only_correct') if b['exact_match'] else (
        'all_only_correct' if a['exact_match'] else 'both_incorrect')
    paired[key] += 1
assert paired == {'all_only_correct':0, 'direct_only_correct':1, 'both_correct':9, 'both_incorrect':2}
result['paired_outcomes'] = paired

# Bounded, random-initialized forward only: check target masks and the actual denominator.
torch.manual_seed(0)
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
before = {k:v.detach().clone() for k,v in model.state_dict().items()}
forward = []
for row in [_vision_records()['train'][0], raw_splits['train'][1]]:
    ids, labels, count = _sequence(row, ctx)
    with torch.no_grad():
        output = model(ids, labels, image=scene(row['color'], row['shape'], offset=row['offset']))
        y = output['labels']; logits = output['logits']
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=-100, reduction='sum')
        expected = ce / count
        observed = masked_loss(logits, y)
    assert int((y != -100).sum()) == count == len(tok.encode(row['answer'])) + 1
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-7)
    forward.append({'question':row['question'], 'answer':row['answer'], 'effective_targets':count,
                    'ids_shape':list(ids.shape), 'expanded_labels_shape':list(y.shape),
                    'masked_loss':float(observed), 'sum_cross_entropy_div_targets':float(expected)})
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
result['bounded_forward'] = {'cases':forward, 'parameters_updated':False,
                             'scope':'random width-8 model target-mask and loss-denominator check only; no capability evaluation'}
for path in ['scripts/course_experiments/modalities.py','tiny_perceptron/model.py','tiny_perceptron/data.py',
             'tiny_perceptron/multimodal.py','tiny_perceptron/training.py']:
    actual = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    assert actual == read('/code_sha256/' + path.replace('~','~0').replace('/','~1'))
read('/revision'); read('/device'); read('/torch_version'); read('/python_version'); read('/step_scale')
result['inspected_json_pointers'] = sorted(set(pointers))
result['annotation_fields_read'] = []
result['boundary'] = 'No training or existing-model re-evaluation. This validates raw measurements and CPU mechanics; it does not establish universal stage superiority or text retention.'
(A/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['environment','budget_fence','data','budget_history','direct_text_holdout_fields_present','bounded_forward']},ensure_ascii=False,indent=2))
print('Historical samples independently recounted: all 9/12, direct_vqa 10/12; every stored per-step target count reproduced; all assertions passed.')
