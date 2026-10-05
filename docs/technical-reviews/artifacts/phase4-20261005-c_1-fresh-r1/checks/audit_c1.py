"""Independent C.1 audit: bytes, data, sampler accounting and saved samples only."""
import ast
import copy
import hashlib
import json
import platform
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from scripts.course_experiments.applications import (
    _reasoning_records, _reasoning_sft, _verify_reasoning, _prompt_ids,
)
from scripts.course_experiments.common import new_lm, split_records, records_sha256, text_examples

BASE = Path(__file__).resolve().parents[1]
raw = (BASE / 'frozen-input/reasoning.json').read_bytes()
report = json.loads(raw)
result = {
    'scope': 'No training, downloads, GPU, checkpoint loading or model generation. Existing raw samples only.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'device': 'cpu', 'cuda_build': str(torch.version.cuda),
                    'cuda_available': str(torch.cuda.is_available())},
    'raw_result_sha256': hashlib.sha256(raw).hexdigest(),
    'checked_json_pointers': [
        '/revision', '/device', '/seed', '/torch_version', '/python_version', '/step_scale',
        '/code_sha256/{scripts/course_experiments/applications.py,scripts/course_experiments/common.py,tiny_perceptron/data.py,tiny_perceptron/model.py}',
        '/results/seed', '/results/split', '/results/model_config',
        '/results/comparison/{direct,steps}/base_state_sha256',
        '/results/comparison/{direct,steps}/training/{steps,planned_steps,step_scale,effective_tokens,records,records_sha256,parameters,trainable_parameters}',
        '/results/comparison/{direct,steps}/budgets/0/{candidate_count,candidate_final_accuracy,generated_tokens,forward_input_tokens}',
        '/results/comparison/{direct,steps}/budgets/0/samples/0..23/{family,a,b,c,question,truth,generation,candidates}',
        '/hf/{repo,revision,prefix,result_revision}',
    ],
}
# Save literal RFC 6901 pointers, rather than compressed display notation.
result['checked_json_pointers'] = [
    '/revision', '/device', '/seed', '/torch_version', '/python_version', '/step_scale',
    '/results/seed', '/results/split', '/results/model_config',
    '/hf/repo', '/hf/revision', '/hf/prefix', '/hf/result_revision',
]
result['checked_json_pointers'] += [
    '/code_sha256/' + name.replace('~', '~0').replace('/', '~1')
    for name in ['scripts/course_experiments/applications.py', 'scripts/course_experiments/common.py',
                 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']
]
for mode in ['direct', 'steps']:
    prefix = '/results/comparison/' + mode
    result['checked_json_pointers'].append(prefix + '/base_state_sha256')
    result['checked_json_pointers'] += [prefix + '/training/' + key for key in [
        'steps', 'planned_steps', 'step_scale', 'effective_tokens', 'records',
        'records_sha256', 'parameters', 'trainable_parameters']]
    result['checked_json_pointers'] += [prefix + '/budgets/0/' + key for key in [
        'candidate_count', 'candidate_final_accuracy', 'generated_tokens', 'forward_input_tokens']]
    for index in range(24):
        result['checked_json_pointers'] += [prefix + '/budgets/0/samples/' + str(index) + '/' + key
            for key in ['family', 'a', 'b', 'c', 'question', 'truth', 'generation', 'candidates']]
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
tok = ByteTokenizer()
answers = ['5', '從2開始數三次：3、4、5，所以是5。', '2+3=5']
counts = []
for answer, expected in zip(answers, [1, 47, 5], strict=True):
    ids = tok.encode(answer)
    assert len(ids) == expected == len(answer.encode('utf-8'))
    assert all(x >= 8 for x in ids) and tok.decode(ids) == answer
    x, y = render_chat([{'role': 'user', 'content': '2+3=?'}, {'role': 'assistant', 'content': answer}])
    labels = y[y != IGNORE].tolist()
    assert labels == ids + [tok.eos_id]
    assert len(labels) == expected + 1
    counts.append({'answer': answer, 'python_characters': len(answer), 'raw_bytes': list(answer.encode('utf-8')),
                   'encoded_ids': ids, 'encode_length': len(ids), 'chat_supervised_labels': labels,
                   'chat_supervised_count': len(labels), 'encode_contains_special_ids': False})
assert tok.encode('2+3=5') == [58, 51, 59, 69, 61]
result['manual_materials'] = counts

rows = _reasoning_records()
splits = split_records(rows, report['seed'])
assert len(rows) == 216 and len({r['family'] for r in rows}) == 56
families = {name: {r['family'] for r in records} for name, records in splits.items()}
assert not (families['train'] & families['validation'] or families['train'] & families['test'] or families['validation'] & families['test'])
assert all(r['family'] == ':'.join(map(str, sorted((r['a'], r['b'], r['c'])))) and
           r['truth'] == r['a'] + r['b'] + r['c'] and
           all(0 <= r[k] <= 5 for k in ['a','b','c']) for r in rows)
result['split_reconstruction'] = {}
for name, records in splits.items():
    observed = {'records': len(records), 'families': len(families[name]), 'sha256': records_sha256(records)}
    assert observed == report['results']['split'][name]
    result['split_reconstruction'][name] = observed

for name in ['scripts/course_experiments/applications.py', 'scripts/course_experiments/common.py', 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']:
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == report['code_sha256'][name]
ctx = SimpleNamespace(seed=report['seed'], device='cpu')
base = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend='sdpa')
base_state = copy.deepcopy(base.state_dict())
base_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in base_state.values())).hexdigest()
result['initialization'] = {'cpu_reconstructed_state_sha256': base_hash, 'saved_branch_state_sha256': {}}

def independent_score(candidate, row, mode):
    text = candidate['generated'].strip()
    clean = not candidate['invalid_special_tokens']
    if mode == 'direct':
        parsed = int(text) if clean and re.fullmatch(r'-?[0-9]+', text) else None
        return {'final_correct': parsed == row['truth'], 'fully_verified': parsed == row['truth']}
    match = re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)', text)
    if not match or not clean:
        final = re.search(r';answer=(-?[0-9]+)$', text) if clean else None
        return {'final_correct': bool(final and int(final[1]) == row['truth']), 'fully_verified': False}
    a,b,s,p,c,t,f = map(int, match.groups())
    return {'final_correct': f == row['truth'],
            'fully_verified': a+b == s and p+c == t and p == s and t == f and
                              (a,b,c) == (row['a'],row['b'],row['c']) and f == row['truth']}

result['training_accounting'] = {}
result['existing_single_candidate_audit'] = {}
draws_by_mode = {}
for mode in ['direct', 'steps']:
    branch = report['results']['comparison'][mode]
    assert base_hash == branch['base_state_sha256']
    result['initialization']['saved_branch_state_sha256'][mode] = branch['base_state_sha256']
    cloned = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend='sdpa')
    cloned.load_state_dict(base_state)
    assert all(torch.equal(v, cloned.state_dict()[k]) for k,v in base_state.items())
    train = _reasoning_sft(splits['train'], mode)
    examples = text_examples(train, mode='sft', max_length=128)
    token_lengths = [int((y != IGNORE).sum()) for _, y in examples]
    assert all(length == len(tok.encode(record['messages'][-1]['content']))+1
               for record,length in zip(train,token_lengths,strict=True))
    rng = random.Random(report['seed'])
    draws = [rng.choices(range(len(examples)), k=24) for _ in range(900)]
    total = sum(token_lengths[index] for batch in draws for index in batch)
    training = branch['training']
    assert training['steps'] == training['planned_steps'] == 900 and training['step_scale'] == 1
    assert training['effective_tokens'] == total and training['records'] == len(train) == 167
    assert training['records_sha256'] == records_sha256(train)
    assert training['parameters'] == training['trainable_parameters'] == sum(p.numel() for p in base.parameters())
    draws_by_mode[mode] = draws
    result['training_accounting'][mode] = {'steps': 900, 'batch_size': 24, 'sampled_examples': 21600,
        'supervised_tokens_recounted': total, 'eos_included': True, 'role_and_prompt_labels_excluded': True,
        'training_records_sha256': records_sha256(train), 'answer_plus_eos_range': [min(token_lengths), max(token_lengths)]}
    budget = branch['budgets'][0]
    assert budget['candidate_count'] == 1
    samples = budget['samples']
    assert len(samples) == len(splits['test']) == 24
    final_correct = fully_verified = tokens = forward_tokens = 0
    for i,(sample,row) in enumerate(zip(samples,splits['test'],strict=True)):
        assert all(sample[k] == row[k] for k in ['family','a','b','c','question','truth'])
        generation = sample['generation']
        assert generation['messages'] == [{'role':'user','content':row['question']}]
        assert generation['input_ids'] == _prompt_ids(generation['messages'], tok)
        assert generation['input_tokens'] == len(generation['input_ids'])
        assert generation['temperature'] == 0.7 and generation['candidate_count'] == 1
        assert generation['max_new_tokens'] == (8 if mode == 'direct' else 48)
        assert len(sample['candidates']) == 1
        candidate = sample['candidates'][0]
        ids = candidate['generated_ids']
        eos = bool(ids and ids[-1] == tok.eos_id)
        body_ids = ids[:-1] if eos else ids
        assert candidate['eos'] == eos and candidate['generated'] == tok.decode(body_ids)
        assert candidate['invalid_special_tokens'] == [x for x in body_ids if x < 8]
        assert candidate['generated_tokens'] == len(ids) == generation['generated_tokens']
        assert len(ids) <= generation['max_new_tokens']
        own = independent_score(candidate,row,mode)
        original = _verify_reasoning(candidate,row,mode)
        for key in ['final_correct','fully_verified']:
            assert candidate[key] == original[key] == own[key]
        final_correct += own['final_correct']; fully_verified += own['fully_verified']
        tokens += len(ids); forward_tokens += generation['forward_input_tokens']
        assert generation['forward_input_tokens'] == generation['input_tokens'] + len(ids)-1
    measure = {'numerator': final_correct, 'denominator':len(samples), 'rate':final_correct/len(samples)}
    assert budget['candidate_final_accuracy'] == measure
    assert budget['generated_tokens'] == tokens and budget['forward_input_tokens'] == forward_tokens
    result['existing_single_candidate_audit'][mode] = {'final_correct':measure, 'fully_verified':fully_verified,
        'generated_tokens':tokens,'forward_input_tokens':forward_tokens,'temperature':0.7,
        'max_new_tokens':8 if mode=='direct' else 48,'raw_samples_checked':len(samples)}
assert draws_by_mode['direct'] == draws_by_mode['steps']
result['training_token_ratio_steps_over_direct'] = result['training_accounting']['steps']['supervised_tokens_recounted'] / result['training_accounting']['direct']['supervised_tokens_recounted']

row={'a':1,'b':2,'c':3,'truth':6}
variants = [
    ('valid','1+2=3;3+3=6;answer=6',[],True,True),
    ('wrong_subtotal_correct_final','1+2=4;4+3=7;answer=6',[],True,False),
    ('wrong_original_operands','0+3=3;3+3=6;answer=6',[],True,False),
    ('missing_step','answer=6',[],False,False),
    ('unparsed_but_final','bad;answer=6',[],True,False),
    ('invalid_special','1+2=3;3+3=6;answer=6',[0],False,False),
]
checked=[]
for name,text,special,final,full in variants:
    sample={'generated':text,'invalid_special_tokens':special}
    own=independent_score(sample,row,'steps'); original=_verify_reasoning(sample,row,'steps')
    assert own == {'final_correct':final,'fully_verified':full}
    assert all(original[k]==v for k,v in own.items())
    checked.append({'case':name,**own})
result['bounded_criterion_variants']=checked
result['manual_arithmetic']={'count_from_two_three_times':[3,4,5], 'final':2+3,
                             'two_addition_example':{'1+2':1+2, '3+3':3+3}}

# The documented shell recipe is inspected, never executed; only asset dispatch is called.
from scripts.course_experiments.run import experiment_spec, list_assets
spec=experiment_spec('reasoning')
assert spec['module']=='applications' and spec['function']=='run_reasoning' and spec['assets']==['gsm8k']
assets=list_assets('reasoning')
assert assets==['assets/training/gsm8k-v1.tar.gz']
result['recipe_contract']={'entrypoint':'scripts/course_experiments/run.py --experiment reasoning --device cpu',
    'dispatch':{'module':spec['module'],'function':spec['function']},'listed_asset_paths':assets,
    'full_recipe_executed':False,'bounded_alternative':'constructor/hash, data reconstruction, label counts, 900 sampler draws, saved sample scoring; no optimizer updates'}

(BASE/'checks/audit-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
