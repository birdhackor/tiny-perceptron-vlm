"""Own bounded CPU derivations and independent audit of original/native tiny records.

No GPU, model/data download, global runner, or source mutation. The native tiny
training run was executed separately into the ignored research directory.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import platform
import random
import re
import torch
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel
from scripts.course_experiments.posttraining import build_records, split_records

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/natural-v4/factual-research/13.17/cpu-run'
torch.set_num_threads(2)
out = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                       'torch_git_version': torch.version.git_version, 'device': 'cpu',
                       'cuda_available': torch.cuda.is_available(), 'cpu_threads': torch.get_num_threads()},
       'predictions_before_probes': {'toy_losses_rounded4': [0.6931, 0.3567, 1.2040],
            'test_success_by_mode': {'sft': [6, 0, 0], 'ppo': [6, 0, 6], 'dpo': [6, 0, 6]},
            'parameters': [148, 241, 97], 'families': [44, 5, 6],
            'budgets': {'ppo_policy_updates': 360, 'critic_updates': 360, 'ppo_samples': 7680,
                        'dpo_pair_draws': 23040, 'reward_updates': 300, 'reward_pair_draws': 19200}}}

# Execute the exact current chapter block before adding the exercise case.
section = (ART / 'original-section.md').read_text()
block = re.search(r'```python\n(.*?)\n```', section, re.S).group(1)
print('EXACT CURRENT LESSON BLOCK:')
exec(compile(block, 'course/chapters/13.md#13.17', 'exec'), {})
print('EXERCISE:')
exec(compile(block.replace('[0.7, 0.3]', '[0.3, 0.7]'), '13.17-exercise', 'exec'), {})
out['toy_cases'] = []
for pc in [0.5, 0.7, 0.3]:
    probabilities = torch.tensor([pc, 1-pc])
    logp = probabilities.log()
    ref = torch.tensor([0.5, 0.5], requires_grad=True).log()
    loss = dpo_loss(logp[:1], logp[1:], ref[:1], ref[1:], beta=1.0)
    margin = math.log(pc/(1-pc))
    sigmoid = 1/(1+math.exp(-margin))
    analytic = -math.log(pc)
    assert abs(loss.item()-analytic) < 2e-7
    out['toy_cases'].append({'chosen': pc, 'rejected': 1-pc, 'reference': [0.5, 0.5],
        'beta': 1, 'analytic_margin': margin, 'analytic_sigmoid': sigmoid,
        'analytic_loss': analytic, 'observed_float32_loss': loss.item(), 'rounded4': round(loss.item(),4)})
assert [row['rounded4'] for row in out['toy_cases']] == [0.6931,0.3567,1.204]

# Transparent prerequisite derivative and frozen-reference probes.
c=torch.tensor([-4.0],requires_grad=True); r=torch.tensor([-3.0],requires_grad=True)
rc=torch.tensor([-4.0],requires_grad=True); rr=torch.tensor([-3.0],requires_grad=True)
l=dpo_loss(c,r,rc,rr,beta=0.1);l.backward()
assert c.grad.item() < 0 and r.grad.item() > 0 and rc.grad is None and rr.grad is None
out['prerequisite_gradient']={'loss':l.item(),'chosen_gradient':c.grad.item(),'rejected_gradient':r.grad.item(),
  'reference_gradients':[None,None],'lr':0.1,'chosen_after':-4-0.1*c.grad.item(),
  'rejected_after':-3-0.1*r.grad.item(),'relative_margin_after':(-4-0.1*c.grad.item())-(-3-0.1*r.grad.item())+1}
for beta in [0,-1]:
    try: dpo_loss(c,r,rc,rr,beta=beta)
    except ValueError: pass
    else: raise AssertionError('nonpositive beta accepted')
before=[0.2,0.1,0.7]; after=[0.15,0.05,0.8]
m=math.log(after[0]/after[1])-math.log(before[0]/before[1])
out['absolute_probability_counterexample']={'reference_and_initial':before,'new_policy':after,
  'both_chosen_and_rejected_decrease':True,'relative_margin':m,
  'initial_loss':math.log(2),'new_loss':math.log1p(math.exp(-m)),
  'scope':'A valid normalized three-card distribution; relative preference can improve while both absolute probabilities fall.'}

official_path=ROOT/'docs/course-experiments/results/posttraining.json'
official=json.loads(official_path.read_text()); original=official['results']
rerun=json.loads((RUN/'result.json').read_text()); current=rerun['results']
out['original_record_sha256']=hashlib.sha256(official_path.read_bytes()).hexdigest()
out['original_code_hash_verification']={}
for path, recorded in official['code_sha256'].items():
    actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    assert actual == recorded, (path,actual,recorded)
    out['original_code_hash_verification'][path]={'recorded':recorded,'current':actual,'match':True}
assert official['device']=='cpu' and official['seed']==42 and original['config']==current['config']
out['configuration']=original['config']

# Generate the family split independently of the project's split implementation.
all_families=sorted(f'pair:{a}:{b}' for a in range(1,11) for b in range(a,11))
assert len(all_families)==55
shuffle=all_families.copy();shuffle.remove('pair:1:2');random.Random(42).shuffle(shuffle)
own_split={'train':set(shuffle[:44]),'validation':set(shuffle[44:49]),'test':set(shuffle[49:])|{'pair:1:2'}}
assert all(not own_split[a]&own_split[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
generated=split_records(build_records(),42)
out['split_audit']={}
for name,families in own_split.items():
    assert families==set(original['splits'][name]['families'])=={r['family'] for r in generated[name]}
    rows=original['evaluations'][name]['rows']
    assert len(rows)==len(families)*3
    for row in rows:
        a,b=row['operands'];mode=row['mode']
        assert row['family']==f'pair:{min(a,b)}:{max(a,b)}' and row['family'] in families
        assert row['features']==[a/10,b/10,float(mode=='explain'),float(mode=='missing')]
        assert row['candidates'][0]==str(a+b) and row['candidates'][2]==str(a+b+1)
        assert row['expected_action']=={'number':0,'explain':1,'missing':3}[mode]
        if mode=='missing':
            assert row['preference_pairs']==[[3,0],[3,1],[3,2]]
        else:
            ranking=[0,1,3,2] if mode=='number' else [1,0,3,2]
            assert row['preference_pairs']==[[x,y] for i,x in enumerate(ranking) for y in ranking[i+1:]]
    raw_data=json.dumps(generated[name],sort_keys=True,ensure_ascii=False).encode()
    assert hashlib.sha256(raw_data).hexdigest()==original['splits'][name]['sha256']
    out['split_audit'][name]={'family_count':len(families),'families':sorted(families),
        'contexts':len(rows),'preference_pairs':sum(len(r['preference_pairs']) for r in rows),
        'recreated_records_sha256':hashlib.sha256(raw_data).hexdigest()}
out['family_swap_check']={'1+2':f'pair:{min(1,2)}:{max(1,2)}','2+1':f'pair:{min(2,1)}:{max(2,1)}',
    'scope':'The actual dataset stores only a<=b, one representative per unordered family; modes stay together.'}

def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

out['rerun_state_audit']={};models={}
for name, cls in [('sft',FiniteResponsePolicy),('ppo',FiniteResponsePolicy),('dpo',FiniteResponsePolicy),
                  ('reward',FiniteRewardModel),('value',FiniteValueModel)]:
    model=cls();model.load_state_dict(torch.load(RUN/(name+'.pt'),map_location='cpu',weights_only=True)['model']);model.eval()
    checkpoint=next(c for c in current['checkpoints'] if c['file']==name+'.pt')
    old_checkpoint=next(c for c in original['checkpoints'] if c['file']==name+'.pt')
    assert state_sha(model)==checkpoint['state_sha256']
    out['rerun_state_audit'][name]={'state_sha256':state_sha(model),'matches_own_run_record':True,
                                 'original_record_state_sha256':old_checkpoint['state_sha256'],
                                 'bitwise_equal_to_original':state_sha(model)==old_checkpoint['state_sha256'],
                                 'parameters':sum(p.numel() for p in model.parameters())}
    models[name]=model
assert [out['rerun_state_audit'][n]['parameters'] for n in ['sft','reward','value']]==[148,241,97]
reference_hash=state_sha(models['sft'])
original_hash=original['reference_state_sha256_before']
assert original_hash==original['ppo']['initial_state_sha256']==original['dpo']['initial_state_sha256']
assert original_hash==original['reference_state_sha256_after']
assert reference_hash==current['ppo']['initial_state_sha256']==current['dpo']['initial_state_sha256']
assert reference_hash==current['reference_state_sha256_before']==current['reference_state_sha256_after']
probe=copy.deepcopy(models['sft']).requires_grad_(False)
fork=copy.deepcopy(probe).requires_grad_(True)
with torch.no_grad(): next(fork.parameters()).add_(0.1)
assert state_sha(probe)==reference_hash and state_sha(fork)!=reference_hash
out['same_start_and_fixed_reference']={'own_reference_hash':reference_hash,'original_record_hash':original_hash,
     'original_initials_and_reference_equal':True,
     'rerun_reference_unchanged':True,'deepcopy_mutation_probe_reference_unchanged':True,
     'scope':'Native CPU tiny checkpoints; no claim about a token LM or GPU replay.'}

out['independent_results']={};out['test_predictions']=[]
for split in ['train','validation','test']:
    rows=original['evaluations'][split]['rows'];out['independent_results'][split]={}
    x=torch.tensor([row['features'] for row in rows],dtype=torch.float32)
    for name in ['sft','ppo','dpo']:
        by_mode={mode:[0,0] for mode in ['number','explain','missing']}
        with torch.no_grad(): loaded_probs=models[name](x).softmax(-1).tolist()
        max_error=0.0
        for row, probs in zip(rows,loaded_probs):
            old=row['policies'][name]['probabilities'];max_error=max(max_error,max(abs(a-b) for a,b in zip(probs,old)))
            action=max(range(4),key=old.__getitem__);expected={'number':0,'explain':1,'missing':3}[row['mode']]
            assert action==row['policies'][name]['chosen_action']
            success=action==expected
            assert success==row['policies'][name]['full_request_success']
            by_mode[row['mode']][0]+=int(success);by_mode[row['mode']][1]+=1
            if split=='test':out['test_predictions'].append({'family':row['family'],'mode':row['mode'],
                'policy':name,'probabilities':old,'independent_argmax':action,'independent_expected':expected,'success':success})
        # Cross-run floating point values need not be bit-identical; exact argmax
        # outcomes and within-run checkpoint/reference identities are the claims.
        assert max_error<=5e-6
        metric=original['evaluations'][split]['policies'][name]
        assert sum(v[0] for v in by_mode.values())==metric['greedy_full_request_success']['numerator']
        for mode,counts in by_mode.items():
            assert counts==[metric['by_mode'][mode]['numerator'],metric['by_mode'][mode]['denominator']]
        out['independent_results'][split][name]={'by_mode':by_mode,'total':[sum(v[0] for v in by_mode.values()),len(rows)],
              'loaded_rerun_probability_max_error_vs_original':max_error}
for name in ['sft','ppo','dpo']:
    assert [out['independent_results']['test'][name]['by_mode'][m][0] for m in ['number','explain','missing']]==out['predictions_before_probes']['test_success_by_mode'][name]
cfg=original['config'];out['budgets']={
    'ppo_policy_updates':cfg['ppo_rollout_batches']*cfg['ppo_epochs_per_rollout'],
    'critic_updates':cfg['ppo_rollout_batches']*cfg['ppo_epochs_per_rollout'],
    'ppo_samples':cfg['ppo_rollout_batches']*cfg['ppo_rollout_size'],
    'ppo_reused_action_draws':cfg['ppo_rollout_batches']*cfg['ppo_rollout_size']*cfg['ppo_epochs_per_rollout'],
    'dpo_pair_draws':cfg['dpo_steps']*cfg['dpo_batch_size'],
    'reward_updates':cfg['reward_steps'], 'reward_pair_draws':cfg['reward_steps']*cfg['reward_batch_size'],
    'unique_train_pairs':44*(6+6+3),'sft_demonstrations':44,'sft_demonstration_draws':60*32,'effective_tokens':0}
for k,v in out['predictions_before_probes']['budgets'].items():assert out['budgets'][k]==v
assert out['budgets']['ppo_policy_updates']==original['ppo']['policy_updates']==current['ppo']['policy_updates']
assert out['budgets']['ppo_samples']==original['ppo']['sampled_actions']==current['ppo']['sampled_actions']
assert out['budgets']['dpo_pair_draws']==original['dpo']['processed_pair_draws']==current['dpo']['processed_pair_draws']
out['original_record_timing']={stage:{'seconds':original[stage]['seconds'],'rounded2':round(original[stage]['seconds'],2)}
    for stage in ['ppo','dpo','reward']}
assert [out['original_record_timing'][s]['rounded2'] for s in ['ppo','dpo','reward']]==[0.66,0.34,0.27]
out['timing_scope']='Original one-run per-stage CPU durations only. Own native rerun total is separately recorded and is not required to reproduce the same wall time; no warmup/repeated benchmark, LLM speed or GPU evidence.'
out['cross_run_scope']='The first audit attempt incorrectly demanded exact equality to historical tensor bytes and failed. Its code/stdout/stderr are preserved. The corrected audit verifies exact own checkpoint-to-own-record identity and exact within-run common initialization. Historical state hashes differ; original and own argmax counts are identical and all probability differences are bounded by 5e-6. No bitwise reproduction claim is made.'
out['rerun_elapsed_seconds']=rerun['elapsed_seconds']
# Execute the capstone dependency dispatcher only, intercepting its training
# callable. This inspects the actual selected predecessor without training or
# loading any token model/checkpoint and is not GPU-training replication.
from unittest.mock import patch
from scripts.course_experiments.common import Context
from scripts.course_experiments import capstone as capstone_driver
ctx=Context('cpu',RUN/'dispatch-only',RUN/'dependencies',ROOT/'assets/training',seed=42)
with patch.object(capstone_driver,'train_stage',return_value='intercepted_without_training') as intercepted:
    assert capstone_driver._run_context(ctx,'dpo')=='intercepted_without_training'
    args,kwargs=intercepted.call_args
assert args[0]=='dpo' and kwargs['input_checkpoint']==RUN/'dependencies/capstone_joint/model.pt'
out['capstone_dispatch_probe']={'stage':args[0],'input_checkpoint':str(kwargs['input_checkpoint'].relative_to(ROOT)),
    'steps':kwargs['steps'],'seed':kwargs['seed'],'device':kwargs['device'],
    'finite_posttraining_checkpoint_selected':False,'training_callable_intercepted':True,
    'scope':'Executed only real _run_context dispatch; train_stage mocked to capture its arguments. No capstone model load, token training or GPU execution.'}
out['source_fingerprints']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/'tiny_perceptron/alignment.py',ROOT/'tiny_perceptron/posttraining.py',
              ROOT/'scripts/course_experiments/posttraining.py',ROOT/'scripts/course_experiments/common.py',
              ROOT/'scripts/course_experiments/capstone.py',ROOT/'tiny_perceptron/capstone.py',ROOT/'tiny_perceptron/training.py']}
out['assertions']='All numerical, split, draw, state, checkpoint reload and independent original argmax assertions passed.'
(ART/'audit-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'environment':out['environment'],'toy_cases':out['toy_cases'],
    'test_results':out['independent_results']['test'],'budgets':out['budgets'],
    'same_start_and_fixed_reference':out['same_start_and_fixed_reference'],
    'original_timing':out['original_record_timing'],'assertions':out['assertions']},ensure_ascii=False,indent=2))
