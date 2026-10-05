"""Independent bounded CPU checks; never loads, trains, or evaluates a saved model."""
import ast
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat, pad_batch
from tiny_perceptron.alignment import sequence_log_probability, dpo_loss

ART = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def selected_function(path, name, globals_dict):
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), globals_dict)
    return globals_dict[name]

pair_examples = selected_function(ART/'inputs/original-behavior.py', '_pair_examples', {'render_chat': render_chat})
measurements = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()),
                               'cuda_build': str(torch.version.cuda)}, 'checks': {}}
counts = {}
for name, answer, expected in [('short','4',(1,2)),('repeat8','4。'*8,(32,33)),
                                ('repeat2','4。'*2,(8,9)),('suffix','4; answer complete',(18,19))]:
    body = tok.encode(answer)
    x, y = render_chat([{'role':'user','content':'2+2=?'},{'role':'assistant','content':answer}])
    actual = (len(body), int((y != IGNORE).sum()))
    assert actual == expected
    assert tok.decode(body) == answer and int(y[y != IGNORE][-1]) == tok.eos_id
    counts[name] = {'answer':answer,'utf8_hex':answer.encode().hex(),'body_tokens':actual[0],
                    'valid_targets_including_eos':actual[1], 'counted_target_ids':y[y != IGNORE].tolist()}
assert len('4'.encode()) == 1 and len('。'.encode()) == 3 and ord('。') == 0x3002
long_explanation = '兩組各2塊積木合起來，共有4塊'
counts['task_dependent_explanation'] = {'answer':long_explanation,'body_tokens':len(tok.encode(long_explanation)),
                                      'valid_targets_including_eos':len(tok.encode(long_explanation))+1}
measurements['checks']['token_counts'] = counts

# The original sum API uses (batch, positions, candidates), with IGNORE excluded.
logits = torch.tensor([[[0.,2.,0.],[2.,0.,0.],[0.,0.,2.]]], dtype=torch.float64)
labels = torch.tensor([[-100,0,2]])
summed = sequence_log_probability(logits,labels).item()
expected_single = math.log(math.exp(2)/(math.exp(2)+2))
assert abs(summed-2*expected_single) < 1e-12
plus = sequence_log_probability(logits,torch.tensor([[1,0,2]])).item()
assert abs(plus-3*expected_single)<1e-12
# Dividing each response by its own length changes the relative DPO logit.
pc,pr,rc,rr = -4.,-6.,-6.,-8.
sum_margin = pc-pr-(rc-rr)
mean_margin = pc/2-pr/4-(rc/2-rr/4)
assert sum_margin == 0 and mean_margin == .5
sum_loss=dpo_loss(*[torch.tensor([v],dtype=torch.float64) for v in [pc,pr,rc,rr]]).item()
mean_loss=dpo_loss(*[torch.tensor([v],dtype=torch.float64) for v in [pc/2,pr/4,rc/2,rr/4]]).item()
assert abs(sum_loss-math.log(2))<1e-12 and mean_loss<sum_loss
# Padding contributes neither to answer target length nor the sequence score.
examples = pair_examples([{'prompt':'2+2=?','chosen':'4','rejected':'4; answer complete'}],128)[0]
bx,by,mask = pad_batch(list(examples))
assert by.ne(IGNORE).sum(-1).tolist() == [2,19]
zeros=torch.zeros((*by.shape, tok.vocab_size),dtype=torch.float64)
score=sequence_log_probability(zeros,by)
assert torch.allclose(score,torch.tensor([-2*math.log(264),-19*math.log(264)],dtype=torch.float64),atol=1e-12,rtol=0)
try:
    pair_examples([{'prompt':'2+2=?','chosen':'4','rejected':'4; answer complete'}],10)
except ValueError:
    max_length_guard=True
else:
    raise AssertionError('oversized pair must not silently crop')
measurements['checks']['sum_mean_contract']={'single_logp':expected_single,'original_sum':summed,
 'additional_valid_position_sum':plus,'sum_margin':sum_margin,'mean_margin':mean_margin,
 'sum_dpo_loss':sum_loss,'mean_dpo_loss':mean_loss,'uniform_vocab_size':264,
 'uniform_two_side_sequence_scores':score.tolist(),'effective_lengths':[2,19],
 'max_length_guard':max_length_guard,'tolerance':'float64 absolute 1e-12; integer lengths exact'}

# A valid two-completion distribution shows that reference subtraction does not forbid length preference.
# theta controls probability of the long duplicate; reference is an equal fixed distribution.
theta=torch.tensor([0.],dtype=torch.float64,requires_grad=True)
ref=torch.tensor([-math.log(2)],dtype=torch.float64,requires_grad=True)
loss=dpo_loss(torch.nn.functional.logsigmoid(theta),torch.nn.functional.logsigmoid(-theta),ref,ref)
loss.backward()
assert theta.grad.item()<0 and ref.grad is None
measurements['checks']['reference_does_not_remove_label_bias']={
 'initial_loss':loss.item(),'d_loss_d_long_logit':theta.grad.item(),'reference_gradient':None,
 'scope':'2 candidate probability distribution, long and short are both factually 4; gradient only; no parameter updates or trained-language-model claim'}

# Opaque complete result saved for full hash; inspect only explicitly selected raw measurement pointers.
raw_path=ART/'inputs/dpo-result.json'
x=json.loads(raw_path.read_bytes())
f=x['results']['format_only']
pointers=['/revision','/seed','/device','/python_version','/torch_version','/step_scale','/code_sha256',
          '/results/format_only/data/train','/results/format_only/data/validation','/results/format_only/data/test',
          '/results/format_only/training/steps','/results/format_only/training/effective_answer_tokens_both_sides',
          '/results/format_only/training/history/*/step',
          '/results/format_only/preference/validation/{records,chosen_higher_absolute_probability,relative_preference_improved,mean_relative_margin,samples}',
          '/results/format_only/preference/test/{records,chosen_higher_absolute_probability,relative_preference_improved,mean_relative_margin,samples}',
          '/results/format_only/arithmetic/test/{records,matches,exact_match,eos_rate,samples}',
          '/results/before/validation/samples/0/{prompt,chosen,policy_chosen_logp}']
assert f['training']['steps']==200 and x['step_scale']==1
assert f['training']['history'][-1]['step']==200
split_counts={}
all_families=[]
for split in ['train','validation','test']:
    p=ART/'inputs/format-pairs'/f'{split}.jsonl'
    rows=[json.loads(line) for line in p.read_bytes().splitlines()]
    meta=f['data'][split]
    assert sha(p)==meta['sha256']
    assert len(rows)==meta['records']
    families={r['family'] for r in rows}
    assert len(families)==meta['families']
    all_families.append(families)
    for r in rows:
        a,b=map(int,r['prompt'].split('=')[0].split('+'))
        assert r['chosen']==str(a+b) and r['rejected']==r['chosen']+'; answer complete'
    for i,pair in enumerate(pair_examples(rows,128)):
        lengths=[int((y!=IGNORE).sum()) for _,y in pair]
        assert lengths==[len(tok.encode(rows[i]['chosen']))+1,len(tok.encode(rows[i]['rejected']))+1]
    split_counts[split]={'records':len(rows),'families':len(families),'sha256':sha(p)}
assert not(all_families[0]&all_families[1] or all_families[0]&all_families[2] or all_families[1]&all_families[2])
assert sum(v['records'] for v in split_counts.values())==64
train_rows=[json.loads(line) for line in (ART/'inputs/format-pairs/train.jsonl').read_bytes().splitlines()]
train_lengths=[sum(int((y!=IGNORE).sum()) for _,y in pair) for pair in pair_examples(train_rows,128)]
sampler=random.Random(x['seed'])
recomputed_exposure=sum(sum(sampler.choices(train_lengths,k=8)) for _ in range(f['training']['steps']))
assert recomputed_exposure==f['training']['effective_answer_tokens_both_sides']==34554
preference={}
for split in ['validation','test']:
    p=f['preference'][split]; rows=p['samples']
    assert len(rows)==p['records']
    for r in rows:
        m=r['policy_chosen_logp']-r['policy_rejected_logp']
        delta=m-r['reference_margin']
        assert abs(m-r['policy_margin'])<1e-9 and abs(delta-r['relative_margin'])<1e-9
        assert r['chosen_answer_tokens']==len(tok.encode(r['chosen']))+1
        assert r['rejected_answer_tokens']==len(tok.encode(r['rejected']))+1
    chosen_after=sum(r['policy_margin']>0 for r in rows)
    chosen_before=sum(r['reference_margin']>0 for r in rows)
    improved=sum(r['relative_margin']>0 for r in rows)
    assert chosen_after==p['chosen_higher_absolute_probability'] and improved==p['relative_preference_improved']
    assert abs(sum(r['relative_margin'] for r in rows)/len(rows)-p['mean_relative_margin'])<1e-9
    preference[split]={'records':len(rows),'positive_reference_margins_before':chosen_before,
                       'positive_policy_margins_after':chosen_after,'positive_relative_improvements':improved,
                       'reference_margins_before':[r['reference_margin'] for r in rows],
                       'relative_improvements':[r['relative_margin'] for r in rows]}
a=f['arithmetic']['test']; rows=a['samples']
assert len(rows)==a['records']==7
match_count=0; eos_count=0
for row in rows:
    ids=row['generated_ids']; raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact=raw==tok.encode(row['expected'])
    assert exact==row['exact'] and tok.decode(raw)==row['generated']
    assert (tok.eos_id in ids)==row['eos']
    match_count+=int(exact); eos_count+=int(tok.eos_id in ids)
assert match_count==a['matches']==0 and a['exact_match']==0 and eos_count/7==a['eos_rate']==1
sample=next(r for r in rows if r['messages'][0]['content']=='2+2=?')
assert sample['generated']=='3' and sample['expected']=='4'
assert preference['test']['positive_reference_margins_before']==7
assert preference['test']['positive_policy_margins_after']==7
assert preference['test']['positive_relative_improvements']==7
v=next(r for r in f['preference']['validation']['samples'] if r['prompt']=='0+3=?')
b=next(r for r in x['results']['before']['validation']['samples'] if r['prompt']=='0+3=?')
assert b['chosen']==v['chosen']=='3'
before_lp=b['policy_chosen_logp']; after_lp=v['policy_chosen_logp']
assert abs(before_lp-(-13.30943))<5e-6 and abs(after_lp-(-13.37556))<5e-6
assert abs(v['relative_margin']-115.77823)<5e-6 and after_lp<before_lp
rejected_before=before_lp-v['reference_margin']
chosen_change=after_lp-before_lp
rejected_change=v['policy_rejected_logp']-rejected_before
assert abs(chosen_change-rejected_change-v['relative_margin'])<1e-9
measurements['checks']['raw_empirical_recalculation']={
 'result_sha256':sha(raw_path),'original_revision':x['revision'],'original_environment':{k:x[k] for k in ['seed','device','torch_version','python_version','step_scale']},
 'read_pointers':pointers,'data_split_counts':split_counts,'steps':f['training']['steps'],
 'effective_answer_tokens_both_sides':f['training']['effective_answer_tokens_both_sides'],
 'recomputed_training_exposure_tokens':recomputed_exposure,'sampled_preference_pairs':200*8,
 'preference':preference,'generation_test':{'records':7,'exact_matches':match_count,'eos_count':eos_count,'two_plus_two':sample},
 'validation_zero_plus_three':{'before_chosen_logp':before_lp,'after_chosen_logp':after_lp,
 'before_rejected_logp_derived_from_reference_margin':rejected_before,'after_rejected_logp':v['policy_rejected_logp'],
 'chosen_logp_change':chosen_change,'rejected_logp_change':rejected_change,'relative_margin_improvement':v['relative_margin'],
 'chosen_probability_ratio':math.exp(chosen_change)},
 'tolerance':'original sample margin arithmetic 1e-9 absolute; textbook 5 decimal rounding ±5e-6; counts/hashes exact',
 'scope':'recomputed saved evidence and data preparation only; no checkpoint/model load, no generation rerun and no training'}
(ART/'cpu-measurements.json').write_text(json.dumps(measurements,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(measurements,ensure_ascii=False,indent=2,allow_nan=False))
print('ALL BOUNDED CPU AND SAVED-MEASUREMENT CHECKS PASSED')
