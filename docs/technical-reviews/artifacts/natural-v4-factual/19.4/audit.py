"""Independent bounded CPU/record audit for section 19.4; no model training."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import (
    CapstoneModel, build_dataset, default_config, prepare_batch, encode_record,
    prompt_ids, preference_pairs, save_capstone, load_capstone, TOK,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss

torch.set_num_threads(2)
BASE = Path(__file__).parent
RESEARCH = ROOT / 'outputs/natural-v4/factual-research/19.4'
splits, manifest = build_dataset()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
result = {
    'scope': 'Fresh CPU loss/count/checkpoint probes and independent fixed-record recomputation; no optimizer update, GPU, checkpoint/model download, or score replication.',
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'torch_git': torch.version.git_version, 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'threads': str(torch.get_num_threads())},
    'dataset': {'version': manifest['version'], 'seed': manifest['seed'], 'counts': manifest['counts'], 'sha256': manifest['sha256']},
    'inputs_sha256': {},
}
assert torch.__version__ == '2.14.1+cpu' and not torch.cuda.is_available()
assert manifest['counts'] == {'train': 552, 'validation': 84, 'test': 90}
for name in splits:
    families = {r['family'] for r in splits[name]}
    for other in splits:
        if name != other:
            assert not families & {r['family'] for r in splits[other]}

# Execute the lesson's single-random-Dense forward/loss example and exercise.
examples = []
chosen = []
for task, expected in [('style', (43, 10)), ('concept', (53, 29))]:
    torch.manual_seed(42)
    row = next(row for row in splits['train'] if row['task'] == task)
    chosen.append(row)
    model = CapstoneModel(default_config(dense=True))
    assert all(type(b.ffn).__name__ == 'DenseFFN' for b in model.language.blocks)
    record = {'task': task, 'id': row['id'], 'user': row['user'], 'answer': row['answer'], 'user_utf8_bytes': len(row['user'].encode()), 'answer_utf8_bytes': len(row['answer'].encode()), 'objectives': []}
    for pretrain, target_count in zip((True, False), expected, strict=True):
        batch, labels = prepare_batch([row], pretrain=pretrain)
        output = model(**batch)
        loss = masked_loss(output['logits'], labels)
        count = int((labels != IGNORE).sum())
        independent_count = len((row['user'] + '\n' + row['answer']).encode()) + 1 if pretrain else len(row['answer'].encode()) + 1
        assert count == target_count == independent_count
        assert loss.isfinite()
        assert batch['ids'].dtype == torch.int64 and labels.dtype == torch.int64
        assert output['logits'].shape == (*labels.shape, 264)
        if not pretrain:
            prefix = prompt_ids(row)
            assert batch['ids'][0, :len(prefix)].tolist() == prefix
            assert labels[0, :len(prefix)-1].eq(IGNORE).all()
            assert labels[0, len(prefix)-1:].tolist() == TOK.encode(row['answer']) + [TOK.eos_id]
        else:
            assert batch['ids'][0, 0] == TOK.bos_id
            assert batch['ids'][0, 1:].tolist() == TOK.encode(row['user'] + '\n' + row['answer'])
        record['objectives'].append({'pretrain': pretrain, 'targets': count, 'loss': float(loss.detach()), 'finite': True, 'input_shape': list(batch['ids'].shape), 'logit_shape': list(output['logits'].shape), 'valid_inputs': int(batch['valid'].sum()), 'ignored_targets': int(labels.eq(IGNORE).sum())})
    examples.append(record)
result['examples'] = examples

# Independent gathered-log-probability denominator and ignored-gradient probe.
batch, labels = prepare_batch(chosen)
torch.manual_seed(1704)
scores = torch.randn(*labels.shape, 264, dtype=torch.float64, requires_grad=True)
active = labels.ne(IGNORE)
logp = scores.log_softmax(-1)
expected_sum = -logp[active].gather(1, labels[active, None]).sum()
expected_mean = expected_sum / active.sum()
actual = masked_loss(scores, labels)
assert torch.allclose(actual, expected_mean, atol=1e-12, rtol=0)
actual.backward()
assert scores.grad[~active].eq(0).all()
altered = scores.detach().clone()
altered[~active] = 9999
assert torch.equal(masked_loss(altered, labels), actual.detach())
result['batch_denominator'] = {'samples': 2, 'active_targets': int(active.sum()), 'padding_inputs': int((~batch['valid']).sum()), 'ignored_targets': int((~active).sum()), 'sum': float(expected_sum.detach()), 'mean': float(actual.detach()), 'independent_mean': float(expected_mean.detach()), 'ignored_gradient_max': float(scores.grad[~active].abs().max()), 'ignored_logit_perturbation_exactly_unchanged': True}

# Save/reload only self-created random weights, proving state and byte identity mechanism.
with tempfile.TemporaryDirectory(dir=RESEARCH) as directory:
    path = Path(directory) / 'same-name.pt'
    save_capstone(path, model, stage='pretrain', step=0, inference_only=True)
    before = sha(path)
    loaded, payload = load_capstone(path)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert payload['tokenizer'] == TOK.state()
    assert len(before) == 64
    changed = bytearray(path.read_bytes()); changed[-1] ^= 1
    assert hashlib.sha256(changed).hexdigest() != before
    result['checkpoint_probe'] = {'stage': payload['stage'], 'step': payload['step'], 'tokenizer': payload['tokenizer'], 'all_state_tensors_equal': True, 'sha256': before, 'hex_characters': len(before), 'one_byte_change_changes_hash': True, 'scope': 'Untrained self-created checkpoint roundtrip; does not prove historical training or authenticates arbitrary external claims.'}

# Rebuild sampled CE target budgets without forwarding or updating a model.
stages = [('pretrain', 300, 364409), ('sft', 1400, 647067), ('joint', 600, 249100), ('dpo', 100, 41403)]
budgets = []
for index, (stage, steps, expected) in enumerate(stages):
    rows = [r for r in splits['train'] if r['image'] is None and r['audio'] is None] if index < 2 else splits['train']
    by_task = defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)
    names = sorted(by_task)
    rng = random.Random(42 + index * 1000)
    total = 0
    sampled = Counter()
    preference_response_targets = 0
    pairs = preference_pairs(splits['train']) if stage == 'dpo' else None
    for _ in range(steps):
        selected = [rng.choice(by_task[rng.choice(names)]) for _ in range(24)]
        for row in selected:
            n = len((row['user'] + '\n' + row['answer']).encode()) + 1 if stage == 'pretrain' else len(row['answer'].encode()) + 1
            total += n
            sampled[row['id']] += 1
        if stage == 'dpo':
            picked = rng.choices(pairs, k=12)
            preference_response_targets += sum(2 * (len(p['chosen'].encode()) + len(p['rejected'].encode()) + 2) for p in picked)
    assert total == expected
    budgets.append({'stage': stage, 'steps': steps, 'batch_size': 24, 'sampler_seed': 42+index*1000, 'candidate_records': len(rows), 'sample_occurrences': 24*steps, 'distinct_sampled_records': len(sampled), 'ce_targets_recomputed': total, 'expected': expected, 'dpo_pairs_per_step': 12 if pairs is not None else 0, 'dpo_pair_occurrences': 12*steps if pairs is not None else 0, 'dpo_policy_reference_response_targets_excluded_from_ce_count': preference_response_targets})
result['budgets'] = budgets

# Independently reconstruct each saved validation action, runtime and final answer.
audit = []
recorded_hashes = []
for stage, steps, tokens in stages:
    directory = ROOT / f'docs/course-experiments/capstone-evidence/{stage}'
    paths = [directory / 'validation.json', directory / 'train-report.json', directory / 'data-manifest.json', ROOT / f'docs/course-experiments/results/capstone_{"preference" if stage=="dpo" else stage}.json']
    for p in paths:
        result['inputs_sha256'][str(p.relative_to(ROOT))] = sha(p)
    valid, report, recorded_manifest, original = [json.loads(p.read_text()) for p in paths]
    assert report == original['results']
    assert recorded_manifest == manifest == report['data_manifest']
    assert original['gpu'] == 'NVIDIA L4' and original['seed'] == 42
    assert original['torch_version'] == '2.14.1+cu126'
    assert report['stage'] == stage and report['steps'] == report['requested_steps'] == steps
    assert report['effective_tokens'] == tokens and report['schedule_completed'] and not report['budget_exhausted']
    assert report['test_evaluated'] is False
    assert len(valid['records']) == 84
    by_task = defaultdict(lambda: {'count': 0, 'action_correct': 0, 'end_to_end_correct': 0})
    failed = Counter()
    def trace_check(trace, expected_prompt):
        assert trace['prompt_ids'] == expected_prompt
        assert trace['raw'] == TOK.decode(trace['generated_ids'])
        assert trace['eos'] == (trace['generated_ids'][-1] == TOK.eos_id)
        assert trace['eos'] == (trace['stop_reason'] == 'eos')
    for row, rec in zip(splits['validation'], valid['records'], strict=True):
        assert rec['id'] == row['id'] and rec['family'] == row['family'] and rec['task'] == row['task']
        assert rec['expected_action'] == row['answer']
        trace = rec['action_trace']; trace_check(trace, prompt_ids(row))
        raw, eos = trace['raw'], trace['eos']
        action_correct = bool(eos and raw == row['answer'])
        expected_final = str(sum(map(int, row['answer'].rsplit(':',1)[1].split('+')))) if row['answer'].startswith('TOOL:') else row['answer'].split(':',1)[1]
        assert rec['expected_final'] == expected_final
        answer = None
        if eos and raw.startswith(('DIRECT:', 'ASK:')) and raw.split(':',1)[1]:
            answer = raw.split(':',1)[1]
        match = re.fullmatch(r'TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})', raw) if eos else None
        if match:
            name, a, b = match.group(1), int(match.group(2)), int(match.group(3))
            if name == 'calculator' and row['available']:
                assert rec['runtime'] == {'status':'ok','result':str(a+b)}
                followup = dict(row, image=None, audio=None)
                followup['user'] = f'原題：{a}+{b}。計算器回報：{a+b}。請回答。'
                final = rec['final_trace']; trace_check(final, prompt_ids(followup))
                if final['eos'] and final['raw'].startswith('DIRECT:') and final['raw'][7:]:
                    answer = final['raw'][7:]
            else:
                assert rec['runtime']['status'] == 'error'
        assert answer == rec['answer']
        end_correct = bool(action_correct and answer == expected_final)
        assert action_correct == rec['action_correct'] and end_correct == rec['end_to_end_correct']
        task = by_task[row['task']]
        task['count'] += 1; task['action_correct'] += action_correct; task['end_to_end_correct'] += end_correct
        if not end_correct: failed[row['task']] += 1
    counts = {'count': sum(v['count'] for v in by_task.values()), 'action_correct': sum(v['action_correct'] for v in by_task.values()), 'end_to_end_correct': sum(v['end_to_end_correct'] for v in by_task.values()), 'by_task': dict(by_task)}
    for k,v in counts.items(): assert valid[k] == report['validation_summary'][k] == v
    modal_tasks = {'image_color','image_shape','joint','audio'}
    modality_n = sum(v['count'] for k,v in by_task.items() if k in modal_tasks)
    modality_c = sum(v['end_to_end_correct'] for k,v in by_task.items() if k in modal_tasks)
    auxiliary_error = max(abs(h['loss_before_update'] - ((h['dpo_loss'] + .2*h['ce_before_update']) if stage == 'dpo' else h['ce_before_update']) - .01*h['auxiliary_before_update']) for h in report['history'])
    assert auxiliary_error < 3e-7
    record = {'stage':stage, 'gpu_record_environment': {k:original[k] for k in ['python_version','torch_version','gpu','device','seed','timing_scope']}, 'training_seconds_recorded': report['elapsed_training_seconds'], 'steps':steps, 'targets':tokens, 'validation_recomputed':counts, 'text_correct':counts['end_to_end_correct']-modality_c, 'text_count':84-modality_n, 'modality_correct':modality_c, 'modality_count':modality_n, 'failures_by_task':dict(failed), 'loss_formula_max_abs_residual':auxiliary_error, 'parent_sha256':report['parent_checkpoint_sha256'], 'export_sha256':report['inference_export']['sha256'], 'test_evaluated':report['test_evaluated']}
    if stage == 'pretrain': record['quoted_4_plus_4_trace'] = valid['records'][0]['action_trace']['raw']
    audit.append(record)
    recorded_hashes.append((report['parent_checkpoint_sha256'],report['inference_export']['sha256']))
assert recorded_hashes[0][0] is None
for index in range(1,4): assert recorded_hashes[index][0] == recorded_hashes[index-1][1]
result['stage_record_audit'] = audit
result['parent_chain'] = {'all_three_full_64_character_links_equal':True, 'pretrain_parent_is_null':True, 'limit':'Verified original recorded linkage and matching training code; historical model tensor/file execution was not independently replayed.'}

# Check full-file hashes against exact original run code, then AST compare used APIs.
result['code_sha256'] = {}
for current, historical in [('tiny_perceptron/capstone.py','historical-capstone.py'), ('scripts/course_experiments/capstone.py','historical-training.py')]:
    p, q = ROOT/current, RESEARCH/historical
    result['code_sha256'][current] = sha(p)
    assert sha(p) == sha(q)
    assert all(json.loads((ROOT/f'docs/course-experiments/capstone-evidence/{s}/train-report.json').read_text())['code_sha256'][current] == sha(p) for s,_,_ in stages)
for current in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/modern.py','tiny_perceptron/alignment.py']:
    result['code_sha256'][current] = sha(ROOT/current)
result['all_checks_passed'] = True
(BASE/'audit-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
