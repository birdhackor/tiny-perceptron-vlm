"""Bounded C.1 checks; no training, model weights, data download, or GPU use."""
import hashlib
import json
import math
import pathlib
import random
import re
import sys

import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from tiny_perceptron.model import loss_sum, masked_loss
from scripts.course_experiments.applications import _reasoning_records, _reasoning_sft
from scripts.course_experiments.common import records_sha256, split_records

ROOT = pathlib.Path.cwd()
torch.set_num_threads(1)
record_path = ROOT / 'docs/course-experiments/results/reasoning.json'
record = json.loads(record_path.read_text())
result = record['results']
out = {'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                       'torch_git_version': torch.version.git_version, 'device': 'cpu',
                       'cuda_available': str(torch.cuda.is_available())},
       'scope': 'Deterministic CPU arithmetic, tokenization, masks and recorded-run audit; no model training or GPU replication.',
       'record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest()}

# Execute the exact lesson's Python block, not a transcription of its output.
section = (ROOT / 'docs/technical-reviews/artifacts/natural-v4-factual/C.1/assigned-section.md').read_text()
code = re.search(r'```python\n(.*?)```', section, re.S)[1]
print('EXACT LESSON BLOCK STDOUT:')
exec(compile(code, 'course/chapters/0C.md#C.1', 'exec'), {})
tok = ByteTokenizer()
answers = ['5', '從2開始數三次：3、4、5，所以是5。', '從2開始數三次：3、4、5，所以是6。']
out['tokenization'] = [{'answer': a, 'characters': len(a), 'utf8_bytes': len(a.encode('utf-8')),
                         'ids': tok.encode(a), 'token_count': len(tok.encode(a))} for a in answers]
assert [len(tok.encode(a)) for a in answers] == [1, 47, 47]
assert all(tok.decode(tok.encode(a)) == a for a in answers)
assert all(i >= 8 for a in answers for i in tok.encode(a))
out['counting_derivation'] = {'start': 2, 'successive_increments': [3, 4, 5], 'sum': 2 + 3,
                              'wrong_final_exercise': 6, 'same_length_but_inconsistent': True}

# Independently construct the complete universe and split without using the project helper.
universe = [{'family': ':'.join(map(str, sorted((a,b,c)))), 'a': a, 'b': b, 'c': c,
             'question': f'({a}+{b})+{c}=?', 'truth': a+b+c}
            for a in range(6) for b in range(6) for c in range(6)]
assert universe == _reasoning_records()
families = sorted({r['family'] for r in universe})
random.Random(42).shuffle(families)
own_splits = {name: [r for f in selected for r in universe if r['family'] == f]
              for name, selected in [('train', families[:44]), ('validation', families[44:50]), ('test', families[50:])]}
assert own_splits == split_records(universe, seed=42)
out['dataset'] = {'ordered_triples': len(universe), 'families': len(families),
                  'independent_family_formula': math.comb(8,3), 'splits': {}}
for name, rows in own_splits.items():
    digest = records_sha256(rows)
    summary = {'records': len(rows), 'families': len({r['family'] for r in rows}), 'sha256': digest}
    assert summary == result['split'][name]
    out['dataset']['splits'][name] = summary
for a,b in [('train','validation'), ('train','test'), ('validation','test')]:
    assert not ({r['family'] for r in own_splits[a]} & {r['family'] for r in own_splits[b]})
out['dataset']['family_intersections'] = 0
out['model_config_record'] = result['model_config']
assert {k:result['model_config'][k] for k in ['width','layers','heads','max_length']} == dict(width=64,layers=2,heads=2,max_length=128)
assert result['comparison']['direct']['base_state_sha256'] == result['comparison']['steps']['base_state_sha256']
out['same_base_record_hash'] = result['comparison']['direct']['base_state_sha256']

out['branches'] = {}
for mode in ['direct','steps']:
    branch = result['comparison'][mode]
    counts = {}
    sft = _reasoning_sft(own_splits['train'], mode)
    assert records_sha256(sft) == branch['training']['records_sha256']
    for name, rows in own_splits.items():
        own_answers = [str(r['truth']) if mode == 'direct' else
                       f"{r['a']}+{r['b']}={r['a']+r['b']};{r['a']+r['b']}+{r['c']}={r['truth']};answer={r['truth']}"
                       for r in rows]
        target_counts = [len(a.encode())+1 for a in own_answers]  # one EOS target
        actual = [int((render_chat(r['messages'])[1] != IGNORE).sum()) for r in _reasoning_sft(rows,mode)]
        assert target_counts == actual
        counts[name] = {'examples': len(rows), 'effective_targets_including_eos': sum(target_counts),
                        'min_targets': min(target_counts), 'max_targets': max(target_counts)}
        if name != 'train': assert sum(target_counts) == branch[name]['effective_tokens']
        if name == 'train': train_counts = target_counts
    rng = random.Random(42)
    scheduled = [i for _ in range(900) for i in rng.choices(range(167),k=24)]
    effective = sum(train_counts[i] for i in scheduled)
    assert effective == branch['training']['effective_tokens']
    assert branch['training']['steps'] == branch['training']['planned_steps'] == 900
    assert branch['training']['step_scale'] == 1.0
    budget = next(b for b in branch['budgets'] if b['candidate_count']==1)
    samples = budget['samples']
    assert len(samples) == 24 and all(len(s['candidates'])==1 for s in samples)
    assert [s['question'] for s in samples] == [r['question'] for r in own_splits['test']]
    correct = 0
    generated_tokens = 0
    for s in samples:
        c = s['candidates'][0]
        ids = c['generated_ids']
        text = tok.decode(ids[:-1] if ids and ids[-1]==tok.eos_id else ids).strip()
        assert text == c['generated'].strip()
        assert len(ids) == c['generated_tokens']
        assert s['generation']['temperature'] == 0.7
        assert s['generation']['max_new_tokens'] == (8 if mode=='direct' else 48)
        assert s['truth'] == s['a']+s['b']+s['c']
        parsed = int(text) if re.fullmatch(r'-?[0-9]+',text) and mode=='direct' else None
        if mode == 'steps':
            match = re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)',text)
            assert match
            a,b,subtotal,previous,carg,total,parsed = map(int,match.groups())
            assert (a,b,carg)==(s['a'],s['b'],s['c']) and subtotal==previous and total==parsed
        assert bool(parsed == s['truth']) == s['candidates'][0]['final_correct']
        correct += int(parsed == s['truth'])
        generated_tokens += len(ids)
    assert correct == budget['candidate_final_accuracy']['numerator']
    assert budget['candidate_final_accuracy']['denominator'] == 24
    assert generated_tokens == budget['generated_tokens']
    test = branch['test']
    recomputed_nll = test['nll_sum']/counts['test']['effective_targets_including_eos']
    assert abs(recomputed_nll-test['nll']) < 1e-12
    out['branches'][mode] = {'target_counts': counts, 'sampled_training_examples': len(scheduled),
         'training_effective_targets_recomputed': effective, 'optimizer_updates_recorded': 900,
         'recorded_full_training_average_nll': branch['training']['final_loss'],
         'recorded_test_nll_sum': test['nll_sum'], 'test_average_nll_recomputed': recomputed_nll,
         'single_candidate_correct_recomputed': correct, 'single_candidate_questions': 24,
         'single_candidate_generated_tokens': generated_tokens, 'max_new_tokens': 8 if mode=='direct' else 48}
assert [out['branches'][m]['single_candidate_correct_recomputed'] for m in ['direct','steps']] == [1,11]
out['training_target_ratio'] = 466322/48803
out['steps_wrong_questions'] = 24-11

# A deterministic cross-entropy probe checks the formula, ignored targets, dtype and gradients.
logits = torch.tensor([[[1.,2.,3.],[4.,-1.,2.],[0.,0.,0.]]], dtype=torch.float64, requires_grad=True)
labels = torch.tensor([[2,IGNORE,0]],dtype=torch.long)
manual = -torch.log_softmax(logits,-1)[0,0,2]-torch.log_softmax(logits,-1)[0,2,0]
summed,n = loss_sum(logits,labels)
mean = masked_loss(logits,labels)
assert int(n)==2 and torch.allclose(manual,summed,rtol=0,atol=1e-12)
assert torch.allclose(mean,manual/2,rtol=0,atol=1e-12)
mean.backward()
assert torch.equal(logits.grad[0,1],torch.zeros(3,dtype=torch.float64))
out['nll_formula_probe']={'logits':logits.detach().tolist(), 'labels':labels.tolist(), 'dtype':'float64',
                         'valid_targets':int(n), 'manual_sum':float(manual.detach()),
                         'project_sum':float(summed.detach()), 'project_mean':float(mean.detach()),
                         'ignored_gradient':logits.grad[0,1].tolist()}
z = torch.tensor([3.,1.,0.],dtype=torch.float64)
out['temperature_probe'] = {str(t):{'p':(z/t).softmax(0).tolist(),
       'entropy_nats':float(-((z/t).softmax(0)*(z/t).log_softmax(0)).sum())} for t in [0.7,1.]}
assert out['temperature_probe']['0.7']['p'][0] > out['temperature_probe']['1.0']['p'][0]
assert out['temperature_probe']['0.7']['entropy_nats'] < out['temperature_probe']['1.0']['entropy_nats']
out['registered_code_hashes']={}
for p in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py',
          'scripts/course_experiments/applications.py','scripts/course_experiments/common.py']:
    h=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    assert h==record['code_sha256'][p]
    out['registered_code_hashes'][p]=h
print('AUDIT RESULT:')
print(json.dumps(out,ensure_ascii=False,indent=2))
print('ALL ASSERTIONS PASSED')
