"""Fresh T.7 audit: source/report arithmetic and one bounded CPU mechanism step.

Does not load, infer with, or retrain any published checkpoint. The historical
GPU measurements remain measurements from the original JSON report.
"""
import ast
import copy
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from scripts.course_experiments.behavior import (
    _dpo_train, _pair_examples, _preference_parts, _state_digest,
)
from scripts.course_experiments.common import Context, split_records
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records, _digest, _utf8_prefix
from scripts.prepare_data import generate_records
from scripts.train import parser, prepare_examples
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import TinyLM, ModelConfig


ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(2)
torch.manual_seed(42)
tok = ByteTokenizer()
report = json.loads((ROOT / 'docs/course-experiments/results/dpo.json').read_text())
dpo = report['results']
style = json.loads((ROOT / 'docs/course-experiments/results/style.json').read_text())['results']
output = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
          'scope': 'Recompute existing report and raw data; one random-model CPU update. No historical checkpoint inference or cloud/GPU training.'}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def close(a, b, tol=1e-6):
    assert abs(a-b) <= tol, (a, b)


def jsonl_digest(rows):
    return sha(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows).encode())


def preference(stats):
    rows = stats['samples']
    assert len(rows) == stats['records']
    for row in rows:
        close(row['policy_margin'], row['policy_chosen_logp']-row['policy_rejected_logp'])
        close(row['relative_margin'], row['policy_margin']-row['reference_margin'])
        for side in ('chosen', 'rejected'):
            assert row[f'{side}_answer_tokens'] == len(tok.encode(row[side]))+1
    counts = {'records': len(rows),
              'chosen_higher_absolute_probability': sum(r['policy_margin'] > 0 for r in rows),
              'relative_preference_improved': sum(r['relative_margin'] > 0 for r in rows),
              'mean_relative_margin': sum(r['relative_margin'] for r in rows)/len(rows)}
    for key, value in counts.items():
        close(value, stats[key])
    return counts


def generation(stats):
    rows = stats['samples']
    assert len(rows) == stats['records']
    exact, ended = 0, 0
    for row in rows:
        question = row['messages'][0]['content']
        a,b = map(int, re.fullmatch(r'(\d+)\+(\d+)=\?', question).groups())
        assert row['expected'] == str(a+b)
        ids = row['generated_ids']
        eos = tok.eos_id in ids
        body = ids[:ids.index(tok.eos_id)] if eos else ids
        assert tok.decode(body) == row['generated']
        matches = body == tok.encode(row['expected'])
        assert matches == row['exact'] and eos == row['eos']
        exact += matches
        ended += eos
    assert stats['matches'] == exact
    close(stats['exact_match'], exact/len(rows))
    close(stats['eos_rate'], ended/len(rows))
    assert stats['effective_tokens'] == sum(len(tok.encode(r['expected']))+1 for r in rows)
    return {'records': len(rows), 'matches': exact, 'eos_count': ended,
            'effective_answer_targets': stats['effective_tokens']}


arithmetic = split_records(arithmetic_records(), seed=42)
pairs = _preference_parts(arithmetic)
assert [len(pairs[s]) for s in ('train','validation','test')] == [49,8,7]
family_sets = [{r['family'] for r in pairs[s]} for s in ('train','validation','test')]
assert all(not (family_sets[i] & family_sets[j]) for i in range(3) for j in range(i))
for split, rows in pairs.items():
    assert jsonl_digest(rows) == dpo['data'][split]['sha256']
output['formal_split'] = {s: {'pairs': len(pairs[s]), 'families': len(family_sets[i]),
                              'jsonl_sha256': jsonl_digest(pairs[s])}
                          for i,s in enumerate(('train','validation','test'))}
formal_config = ModelConfig(width=64,layers=2,max_length=128)
formal_parameters = sum(p.numel() for p in TinyLM(formal_config).parameters())
assert formal_parameters == style['content_training']['parameters'] == 141568
output['formal_model_config'] = {'width':64,'layers':2,'max_length':128,'parameters':formal_parameters,
    'scope':'Constructed stated architecture and compared parameter count with the original style report; no published weights loaded.'}
examples = _pair_examples(pairs['train'], 128)
sampler = random.Random(42)
effective = sum(int((y != -100).sum()) for _ in range(250)
                for pair in sampler.choices(examples, k=8) for x,y in pair)
assert effective == 9448
output['answer_target_sampling'] = {'updates': 250, 'pairs_per_update': 8, 'pair_draws': 2000,
    'seed': 42, 'both_sides_targets_including_eos': effective,
    'scope': 'Replay deterministic sampling and labels only; no optimization or model evaluation.'}
output['before'] = {s: {'preference': preference(dpo['before'][s]),
                       'generation': generation(style['content_evaluation'][s])} for s in ('validation','test')}
output['formal_runs'] = {}
for name,run in dpo['runs'].items():
    assert run['training']['steps'] == 250
    assert run['training']['effective_answer_tokens_both_sides'] == effective
    output['formal_runs'][name] = {'beta': run['beta'], 'steps': run['training']['steps'],
        'preference': {s: preference(run['preference'][s]) for s in ('validation','test')},
        'generation': {s: generation(run['arithmetic'][s]) for s in ('validation','test')}}
specific_pair = next(r for r in dpo['runs']['model']['preference']['test']['samples'] if r['prompt']=='4+2=?')
specific_generation = next(r for r in dpo['runs']['model']['arithmetic']['test']['samples']
                           if r['messages'][0]['content']=='4+2=?')
assert specific_pair['policy_margin'] > 0 and specific_generation['generated'] == '8'
output['four_plus_two'] = {'candidate_scores': specific_pair, 'stored_generation': specific_generation}
formatted = dpo['format_only']
assert formatted['training']['steps'] == 200
for split,rows in pairs.items():
    format_rows = [{**r, 'rejected': r['chosen']+'; answer complete'} for r in rows]
    assert jsonl_digest(format_rows) == formatted['data'][split]['sha256']
output['format_only'] = {'steps': 200,
    'preference': {s: preference(formatted['preference'][s]) for s in ('validation','test')},
    'generation': {s: generation(formatted['arithmetic'][s]) for s in ('validation','test')}}
assert output['format_only']['preference']['test']['relative_preference_improved'] == 7
assert output['format_only']['generation']['test']['matches'] == 0

asset = report['assets'][0]
archive = ROOT/asset['archive']
assert sha(archive.read_bytes()) == asset['archive_sha256']
with tarfile.open(archive) as tar:
    target = next(m for m in tar.getmembers() if m.name.endswith('/train-first-100.jsonl'))
    raw_bytes = tar.extractfile(target).read()
expected_raw_hash = next(f['sha256'] for f in asset['files'] if f['path'].endswith('/train-first-100.jsonl'))
assert sha(raw_bytes) == expected_raw_hash
raw = [json.loads(line) for line in raw_bytes.decode().splitlines()]
natural = []
for i,row in enumerate(raw):
    answers=[]
    for side in ('chosen','rejected'):
        text=row[side]
        if isinstance(text,list):
            text=next(m['content'] for m in reversed(text) if m['role']=='assistant')
        answers.append(_utf8_prefix(text,120))
    natural.append({'family': row.get('prompt_id',_digest(row['prompt'])),
                    'prompt': _utf8_prefix(row['prompt'],120), 'chosen': answers[0], 'rejected': answers[1],
                    'source_row': i,'source_record_sha256': _digest(row)})
natural_parts=split_records(natural,seed=42)
pilot=dpo['ultrafeedback_pilot']
assert len(raw)==pilot['source_records']==100
assert [len(natural_parts[s]) for s in ('train','validation','test')]==[80,10,10]
for split,rows in natural_parts.items():
    assert jsonl_digest(rows)==pilot['data'][split]['sha256']
    for row in rows:
        assert all(len(row[k].encode())<=120 for k in ('prompt','chosen','rejected'))
    if split != 'train':
        for row,sample in zip(rows,pilot['evaluation'][split]['samples'],strict=True):
            assert all(row[k]==sample[k] for k in row)
assert pilot['sft_training']['steps']==pilot['training']['steps']==80
assert 'arithmetic' not in pilot and set(pilot['evaluation'])=={'validation','test'}
output['natural_pilot']={'source_pairs':len(raw),'archive_sha256':sha(archive.read_bytes()),
    'raw_jsonl_sha256':sha(raw_bytes),'split':{s:len(rows) for s,rows in natural_parts.items()},
    'maximum_utf8_bytes':120,'sft_steps':80,'dpo_steps':80,
    'evaluations':{s:preference(pilot['evaluation'][s]) for s in ('validation','test')},
    'scope':'Copied preference labels for byte prefixes; no independent reannotation or free-generation assessment.'}

generated = generate_records('preference')
families=sorted({r['family'] for r in generated}); random.Random(42).shuffle(families)
validation_families=set(families[int(.8*len(families)):int(.9*len(families))])
first=next(r for r in generated if r['family'] in validation_families)
assert first['prompt']=='0+5=?' and first['chosen']=='5' and first['rejected']=='6'
args=parser().parse_args(['--task','dpo','--train','--steps','200','--checkpoint','checkpoints/style.pt'])
assert args.train and args.steps==200 and str(args.checkpoint)=='checkpoints/style.pt'
with tempfile.TemporaryDirectory(prefix='tool-choice-dep-t7-') as tmp:
    path=Path(tmp)/'pair.jsonl'; path.write_text(json.dumps(first)+'\n')
    args.data=path
    pair=prepare_examples(args,tok)[0]
    full_conversation={k:[{'role':'user','content':first['prompt']},{'role':'assistant','content':first[k]}]
                       for k in ('chosen','rejected')}
    path.write_text(json.dumps(full_conversation)+'\n')
    other=prepare_examples(args,tok)[0]
    assert all(torch.equal(a,b) for e,f in zip(pair,other) for a,b in zip(e,f))
    masks=[y.tolist() for x,y in pair]
    assert all(sum(v!=-100 for v in y)==2 for y in masks)
    policy=TinyLM(ModelConfig(width=8,layers=1,max_length=32))
    reference=copy.deepcopy(policy).eval().requires_grad_(False)
    ref_before=_state_digest(reference.state_dict());policy_before=_state_digest(policy.state_dict())
    ctx=Context('cpu',Path(tmp)/'one-step',Path(tmp),Path(tmp),seed=42)
    training=_dpo_train(policy,reference,[first],ctx,'bounded',count=1)
    assert _state_digest(reference.state_dict())==ref_before
    assert _state_digest(policy.state_dict())!=policy_before
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    output['bounded_cpu_update']={'random_model_width':8,'layers':1,'updates':1,'pairs_per_update':8,
        'loss_before_update':training['history'][0]['loss_before_update'],
        'effective_answer_targets_both_sides':training['effective_answer_tokens_both_sides'],
        'reference_unchanged':True,'reference_frozen_and_no_grads':True,'policy_changed':True,
        'scope':'Random model with one repeated addition pair; not the published model or recipe.'}
    cli = [sys.executable,'scripts/infer.py',str(ctx.output/'bounded.pt'),'--chat','--prompt','0+5=?',
           '--tokens','2','--temperature','0','--device','cpu']
    json_result = subprocess.run(cli+['--json'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=20)
    plain_result = subprocess.run(cli,cwd=ROOT,capture_output=True,text=True,check=True,timeout=20)
    transient_report = json.loads(json_result.stdout)
    assert plain_result.stdout.rstrip('\n') == transient_report['answer']
    assert len(transient_report['generated_ids']) <= 2
    output['transient_checkpoint_cli'] = {'json_command':cli+['--json'],'plain_command':cli,
        'json_exit_code':json_result.returncode,'plain_exit_code':plain_result.returncode,
        'generation_report':transient_report,'plain_and_json_answer_equal':True,
        'scope':'Only a temporary width8 one-update random-model checkpoint; bounded tokens2 CPU CLI check. Did not infer style.pt/preferred.pt or published model.pt.'}
output['exploratory_cli']={'first_validation_pair':first,'parsed_updates':200,'labels':masks,
    'full_conversation_and_shared_prompt_formats_equal':True,'full_bash_training_recipe_executed':False}

chosen=torch.tensor([-4.0],requires_grad=True);rejected=torch.tensor([-3.0],requires_grad=True)
ref_chosen=torch.tensor([-4.0],requires_grad=True);ref_rejected=torch.tensor([-3.0],requires_grad=True)
loss=dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta=.1);loss.backward()
close(float(loss.detach()),math.log(2));close(float(chosen.grad),-.05);close(float(rejected.grad),.05)
assert ref_chosen.grad is None and ref_rejected.grad is None
logits=torch.zeros((1,4,264));labels=torch.tensor([[-100,-100,tok.encode('5')[0],tok.eos_id]])
score=sequence_log_probability(logits,labels)
close(float(score),-2*math.log(264),tol=2e-6)
output['dpo_formula']={'loss_at_zero_relative_margin':float(loss.detach()),'chosen_gradient':float(chosen.grad),
    'rejected_gradient':float(rejected.grad),'reference_score_gradients_detached':True,
    'uniform_logits_answer_and_eos_sum':float(score),'formula_expected':-2*math.log(264)}

revision=report['revision']
version_check={}
for name in ('scripts/course_experiments/behavior.py','scripts/course_experiments/run.py'):
    original=subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=ROOT)
    current=(ROOT/name).read_bytes()
    assert sha(original)==report['code_sha256'][name]
    original_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    current_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(current).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    funcs=('_preference_parts','_pair_examples','_dpo_train','_preference_evaluate','_natural_dpo_pilot','run_dpo') if name.endswith('behavior.py') else ()
    assert all(original_functions[n]==current_functions[n] for n in funcs)
    version_check[name]={'original_revision':revision,'original_sha256':sha(original),'current_sha256':sha(current),
        'identical_dpo_function_asts':list(funcs),
        'changed_functions':[n for n in current_functions if current_functions[n]!=original_functions.get(n)]}
for name in ('scripts/course_experiments/common.py','scripts/course_experiments/text.py','tiny_perceptron/alignment.py','tiny_perceptron/data.py','tiny_perceptron/model.py'):
    assert sha((ROOT/name).read_bytes())==report['code_sha256'][name]
output['historical_code_comparison']=version_check
spec=experiment_spec('dpo')
assert spec['function']=='run_dpo' and spec['module']=='behavior'
assert spec['dependencies']==['style'] and spec['assets']==['ultrafeedback-dpo']
output['current_plan_resolution']=spec
output['historical_reference']={'reported_unchanged':dpo['reference_unchanged'],'reported_state_sha256':dpo['reference_sha256'],
    'scope':'Verified reporting flag and code hash check, plus independent random-model freeze probe; did not reload historical reference checkpoints.'}
assert dpo['reference_unchanged'] is True
output['result']='PASS: all report/data/formula/CLI and bounded mechanism assertions passed.'
print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
