"""Bounded CPU arithmetic and existing-result audit; no model evaluation or training."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


inspected = []


def pointer(value, path):
    inspected.append(path)
    for part in path.strip('/').split('/'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


result = json.loads((HERE / 'inputs/docs/course-experiments/results/distillation.json').read_text())
base = '/results/tasks/attributes'


def field(path):
    return pointer(result, base + '/' + path)


def original_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in chosen} == set(names)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), namespace)
    return [{'file': str(path.relative_to(HERE)), 'function': n.name,
             'first_line': n.lineno, 'last_line': n.end_lineno} for n in chosen]


spec = importlib.util.spec_from_file_location('original_byte_data', HERE / 'versioned-code/tiny_perceptron/data.py')
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)
namespace = dict(torch=torch, F=F, IGNORE=data_module.IGNORE,
                 ByteTokenizer=data_module.ByteTokenizer, render_chat=data_module.render_chat,
                 pad_batch=data_module.pad_batch, json=json, hashlib=hashlib, Path=Path,
                 _sha=sha, extract_asset=lambda ctx, identifier: HERE / 'original-data')
method_locations = original_functions(
    HERE / 'versioned-code/scripts/course_experiments/compression.py',
    ['_chat', '_example', '_prompt', '_answer', '_challenge'], namespace)
method_locations += original_functions(HERE / 'versioned-code/tiny_perceptron/model.py',
                                       ['loss_sum', 'masked_loss'], namespace)
tok = data_module.ByteTokenizer()

# Execute the unmodified original fence and the requested accounting variant.
fence_namespace = {}
exec(compile((HERE / 'fence-1.py').read_bytes(), 'original-fence-1.py', 'exec'), fence_namespace)
runs = fence_namespace['runs']
assert [(x['correct'] / x['total'], x['student_seconds'] + x['teacher_seconds'])
        for x in runs] == [(0.75, 10), (1.0, 50)]
variant = copy.deepcopy(runs)
variant[1]['teacher_seconds'] = 0
assert variant[1]['correct'] / variant[1]['total'] == 1.0
assert variant[1]['student_seconds'] + variant[1]['teacher_seconds'] == 10
print('Accounting variant: distill accuracy=1.0; total_seconds=10; student data unchanged')

# A teacher-agreement counterexample is a logical sample, not a model result.
gold, teacher, before, after = [0, 1], [0, 0], [0, 1], [0, 0]
agreement = lambda x: sum(a == b for a, b in zip(x, teacher)) / len(gold)
accuracy = lambda x: sum(a == b for a, b in zip(x, gold)) / len(gold)
assert (agreement(before), accuracy(before), agreement(after), accuracy(after)) == (0.5, 1, 1, 0.5)
print('Constructed counterexample: teacher agreement 0.5->1.0 while truth accuracy 1.0->0.5')

# Candidate axis is last; labels are [batch, position]. Unequal lengths and IGNORE.
probabilities = torch.tensor([[[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]],
                              [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]], dtype=torch.float64)
labels = torch.tensor([[0, 1, -100], [1, -100, -100]])
summed, count = namespace['loss_sum'](probabilities.log(), labels)
manual = -math.log(0.8) - math.log(0.4) - math.log(0.8)
assert int(count) == 3 and abs(summed.item() - manual) < 1e-12
left, left_count = namespace['loss_sum'](probabilities[:1].log(), labels[:1])
right, right_count = namespace['loss_sum'](probabilities[1:].log(), labels[1:])
assert abs((left + right).item() - manual) < 1e-12
assert abs(namespace['masked_loss'](probabilities.log(), labels).item() - manual / 3) < 1e-12
print(f'Toy answer NLL: sum={summed.item():.12f}; valid_positions=3; mean={manual / 3:.12f}; unit=nats/position')

dataset_path = HERE / 'original-data/dataset.json'
hard_path = HERE / 'original-data/sft-hard-targets.json'
assert sha(dataset_path) == field('data/sha256')
assert sha(hard_path) == field('hard_target_generation/sha256')
dataset = json.loads(dataset_path.read_text())
hard = json.loads(hard_path.read_text())
families = {s: {x['family'] for x in rows} for s, rows in dataset.items()}
assert not any(families[a] & families[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
assert {s: len(v) for s, v in dataset.items()} == field('data/counts')
assert {s: len(v) for s, v in families.items()} == field('data/families')
assert field('data/family_intersections') == 0 and len(dataset['train']) == field('data/student_training_records') == 45
assert len(hard['records']) == len(hard['audit']) == 45
for original, generated, audit in zip(dataset['train'], hard['records'], hard['audit'], strict=True):
    answer = original['messages'][-1]['content']
    assert original['family'] == generated['family'] == audit['family']
    assert original['messages'][-2]['content'] == audit['question']
    assert audit['gold_answer'] == audit['teacher_answer'] == answer
    assert generated['_hard_ids'] == audit['teacher_ids'] == tok.encode(answer) + [tok.eos_id]
    assert audit['teacher_correct'] and audit['eos'] and generated['_hard_eos']
    assert audit['valid_target_tokens'] == len(audit['teacher_ids'])
assert field('hard_target_generation/correct') == 45 and field('hard_target_generation/wrong') == 0
assert field('hard_target_generation/zero_target_records') == 0

seed = pointer(result, '/seed')
assert seed == 42
plan_rng = random.Random(seed)
plan = [[plan_rng.randrange(45) for _ in range(16)] for _ in range(400)]
plan_hash = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
gold_examples = [namespace['_example'](x, 128) for x in dataset['train']]
hard_examples = [namespace['_example'](x, 128) for x in hard['records']]
assert all(torch.equal(g[0], h[0]) and torch.equal(g[1], h[1])
           for g, h in zip(gold_examples, hard_examples, strict=True))
effective = sum(int((gold_examples[i][1] != data_module.IGNORE).sum()) for batch in plan for i in batch)
assert effective == 44985
heldout_tokens = {s: sum(int((namespace['_example'](x, 128)[1] != data_module.IGNORE).sum()) for x in dataset[s])
                  for s in ['validation', 'test']}
assert heldout_tokens == {'validation': 36, 'test': 69}

audit_metrics = []


def check_metrics(metrics, label, rows=None):
    assert metrics['examples'] == len(metrics['generated_samples'])
    if rows is not None:
        assert metrics['examples'] == len(rows)
    correct = completed = eos_count = 0
    for i, sample in enumerate(metrics['generated_samples']):
        ids = sample['generated_ids']
        ended = tok.eos_id in ids
        raw = ids[:ids.index(tok.eos_id)] if ended else ids
        exact = raw == tok.encode(sample['expected'])
        assert exact == sample['exact'] and ended == sample['ended_with_eos']
        assert tok.decode(raw) == sample['generated']
        if rows is not None:
            assert sample['family'] == rows[i]['family']
            assert sample['expected'] == namespace['_answer'](rows[i])
        correct += exact
        completed += exact and ended
        eos_count += ended
    assert correct == metrics['correct'] and completed == metrics['completed_correct'] and eos_count == metrics['eos_count']
    assert abs(correct / len(metrics['generated_samples']) - metrics['exact_match']) < 1e-12
    assert abs(metrics['nll_sum'] / metrics['supervised_tokens'] - metrics['answer_nll']) < 1e-12
    audit_metrics.append(dict(label=label, examples=metrics['examples'], valid_tokens=metrics['supervised_tokens'],
                              correct=correct, nll=metrics['answer_nll']))


teacher_tests = field('teacher_test')
for split in ['validation', 'test']:
    m = field('teacher_' + split)
    assert m['supervised_tokens'] == heldout_tokens[split]
    check_metrics(m, 'teacher_' + split, dataset[split])
assert field('teacher_frozen_and_unchanged') is True
assert field('teacher_provenance/config')['vocab_size'] == 264
assert field('teacher_provenance/config')['max_length'] == 128
assert field('teacher_provenance/steps') == 900
assert field('teacher_cache/seconds') > 0 and field('teacher_cache/file_bytes') > 0
assert field('hard_target_generation/seconds') > 0
for width in [16, 32]:
    initialization = set()
    params = set()
    for method in ['ce', 'teacher_hard', 'ce_kl']:
        run = f'w{width}_{method}'
        training = field('runs/' + run + '/training')
        initialization.add(training['initialization_sha256'])
        params.add(field('runs/' + run + '/storage/parameter_count'))
        assert training['batch_plan_sha256'] == plan_hash
        assert training['steps'] == training['optimizer_updates'] == 400
        assert training['effective_supervised_tokens'] == effective
        assert training['training_examples'] == training['training_sequence_chunks'] == 45
        assert training['weights_changed'] and training['initialization_sha256'] != training['final_sha256']
        assert training['seconds'] > 0
        for split in ['validation', 'test']:
            m = field('runs/' + run + '/' + split)
            assert m['supervised_tokens'] == heldout_tokens[split]
            check_metrics(m, run + '/' + split, dataset[split])
        agreement_value = sum(s['generated_ids'] == t['generated_ids'] for s, t in
                              zip(m['generated_samples'], teacher_tests['generated_samples'], strict=True)) / 10
        assert agreement_value == field('runs/' + run + '/teacher_agreement')
    assert len(initialization) == len(params) == 1
    assert params == {13744 if width == 16 else 33632}
print('Attribute audit: split sizes=45/5/10; family counts=9/1/2; no family overlap; heldout tokens=36/69')
print(f'Matched training: seed={seed}; 400 updates; batch=16; effective targets={effective}; plan_sha256={plan_hash}')

# Execute the original selection method with an offline artifact-root resolver.
selection_saved = field('out_of_domain_gsm8k/selection')
assert sha(HERE / 'original-data/gsm8k-train-first200.jsonl') == selection_saved['source_file_sha256']
selected, selection = namespace['_challenge'](SimpleNamespace(), 128, reserved_tokens=24)
for key in ['status', 'source_rows', 'source_file_sha256', 'eligible_rows', 'selected_rows', 'skipped_rows',
            'unselected_eligible_rows', 'selected_source_rows', 'skipped', 'max_length', 'reserved_generation_tokens']:
    assert selection[key] == selection_saved[key], key
assert selection['selected_source_rows'] == [14, 94]
assert len(selected) == 2 and selection['skipped_rows'] == 198
for row in selected:
    assert len(namespace['_prompt'](row)) + 24 <= 128
    assert len(tok.encode(row['answer'])) + 1 <= 24
    assert row['question'] not in {x['messages'][-2]['content'] for x in dataset['train']}
check_metrics(field('out_of_domain_gsm8k/teacher'), 'OOD/teacher', selected)
for run in ['w16_ce', 'w16_teacher_hard', 'w16_ce_kl', 'w16_ce_kl_packed4',
            'w32_ce', 'w32_teacher_hard', 'w32_ce_kl', 'w32_ce_kl_packed4']:
    check_metrics(field('runs/' + run + '/out_of_domain_gsm8k'), 'OOD/' + run, selected)
print('Original OOD selector reproduced: 200 original rows; 2 eligible/selected; 198 skipped; source rows=[14,94]')

output = dict(environment=dict(python=sys.version, torch=torch.__version__, torch_git_version=torch.version.git_version,
                               device='cpu', cuda_build=str(torch.version.cuda), cuda_available=str(torch.cuda.is_available())),
              status='pass', original_method_locations=method_locations,
              result_json_sha256=sha(HERE / 'inputs/docs/course-experiments/results/distillation.json'),
              inspected_pointers=sorted(set(inspected)), metric_recomputations=audit_metrics,
              arithmetic=dict(table_accuracy=[0.75, 1.0], table_seconds=[10, 50], variant_seconds=10,
                              toy_nll_sum=manual, toy_nll_count=3, toy_nll_mean=manual / 3),
              data=dict(split_sizes={s: len(v) for s,v in dataset.items()}, heldout_tokens=heldout_tokens,
                        effective_supervised_tokens=effective, batch_plan_sha256=plan_hash,
                        ood_selected_rows=selection['selected_source_rows']))
(HERE / 'cpu-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(f'PASS: {len(audit_metrics)} existing metric groups audited; no training or full model reevaluation')
