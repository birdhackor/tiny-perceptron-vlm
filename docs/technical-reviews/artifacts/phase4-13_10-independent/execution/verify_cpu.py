"""Fresh bounded CPU checks for 13.10; no existing weights or training runner."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
from tiny_perceptron.posttraining import preference_loss, FiniteRewardModel

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1',
           HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
command = [str(ROOT / '.venv/bin/python'), str(A / 'inputs/fence-1.py')]
run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                     timeout=20, check=False)
(A / 'execution/fence.stdout.txt').write_text(run.stdout)
(A / 'execution/fence.stderr.txt').write_text(run.stderr)
assert run.returncode == 0, run.stderr
assert run.stdout == ('差距 0.0 勝出機率 0.5 代價 0.6931\n'
                      '差距 2.0 勝出機率 0.8808 代價 0.1269\n')

variants = []
for rejected in [0., 1.]:
    for chosen in [0., 2.]:
        c = torch.tensor([chosen], dtype=torch.float64)
        r = torch.tensor([rejected], dtype=torch.float64)
        d = chosen - rejected
        expected_probability = 1 / (1 + math.exp(-d))
        expected_loss = math.log1p(math.exp(-d))
        p = float(torch.sigmoid(c-r).item())
        l = float(preference_loss(c,r).item())
        shifted = float(preference_loss(c+10,r+10).item())
        assert math.isclose(p, expected_probability, abs_tol=1e-14)
        assert math.isclose(l, expected_loss, abs_tol=1e-14)
        assert shifted == l
        variants.append(dict(chosen=chosen, rejected=rejected, gap=d,
                             probability=p, loss=l, shifted_loss=shifted))
grid = torch.tensor([[0.,2.],[-1.,1.]],dtype=torch.float64)
mean_loss = float(preference_loss(grid, torch.zeros_like(grid)))
expected_mean = sum(math.log1p(math.exp(-d)) for d in [0.,2.,-1.,1.])/4
assert math.isclose(mean_loss, expected_mean, abs_tol=1e-14)
errors = []
for c,r in [(torch.empty(0),torch.empty(0)),(torch.zeros(2),torch.zeros(1))]:
    try: preference_loss(c,r)
    except ValueError: errors.append('ValueError')
    else: raise AssertionError('invalid input accepted')
c = torch.tensor([0.,2.],dtype=torch.float64,requires_grad=True)
r = torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True)
preference_loss(c,r).backward()
assert (c.grad<0).all() and (r.grad>0).all()

# Compile only these original AST definitions; avoid unrelated result strings.
source = ROOT / 'scripts/course_experiments/posttraining.py'
tree = ast.parse(source.read_text())
names = ['rule_best_action','build_records','split_records','_features','_pairs']
definitions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
constants = [n for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id in ['MODES','RESERVED_TEST_FAMILY']
                     for t in n.targets)]
ns = dict(random=random, torch=torch)
exec(compile(ast.Module(body=constants+definitions,type_ignores=[]),str(source),'exec'),ns)
records = ns['build_records']()
splits = ns['split_records'](records,42)
assert len(records)==165 and len({r['family'] for r in records})==55
owners = {k:{r['family'] for r in rows} for k,rows in splits.items()}
assert all(not owners[a]&owners[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
assert 'pair:1:2' in owners['test'] and 'pair:1:2' not in owners['train']
assert all(r['operands']==sorted(r['operands']) for r in records)
samples = [r for r in records if r['family']=='pair:1:2']
assert samples[0]['candidates']==['3','1 加 2 是 3。','4','請再提供更多資訊。']
assert [r['expected_action'] for r in samples]==[0,1,3]
assert samples[2]['candidates'][3]=='請補充購買數量。'
assert all(r['features']==[r['operands'][0]/10,r['operands'][1]/10,
                          float(r['mode']=='explain'),float(r['mode']=='missing')]
           for r in records)
x=ns['_features'](samples,'cpu')
torch.manual_seed(0)
model=FiniteRewardModel()
model_input=[]
handle=model.network[0].register_forward_pre_hook(lambda mod,args:model_input.append(args[0].detach().clone()))
scores=model(x)
handle.remove()
assert tuple(x.shape)==(3,4) and tuple(scores.shape)==(3,4)
assert tuple(model_input[0].shape)==(3,4,8)
assert torch.equal(model_input[0][:,:,:4],x[:,None,:].expand(-1,4,-1))
assert torch.equal(model_input[0][:,:,4:],torch.eye(4)[None,:,:].expand(3,-1,-1))
model.requires_grad_(False).eval()
assert not any(p.requires_grad for p in model.parameters())
assert not model(x).requires_grad

# Only raw measurements, methods, and provenance pointers are read below.
raw_path=A/'inputs/posttraining-results.raw.json'
raw=json.loads(raw_path.read_text())
result=raw['results']
provenance={k:raw[k] for k in ['device','seed','torch_version','python_version','code_sha256']}
for path,digest in raw['code_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
assert raw['device']=='cpu' and raw['seed']==42
assert result['config']['reward_steps']==300 and result['config']['reward_batch_size']==64
assert result['reward']['unique_training_pairs']==660
assert result['reward']['processed_pair_draws']==300*64
history=result['reward']['history']
assert [r['step'] for r in history]==[1,50,100,150,200,250,300]
measurements={}
row_fields=['family','operands','mode','prompt','candidates','features','preference_pairs',
            'expected_action','candidate_content_or_clarification_correct','candidate_meets_full_request']
for name,rows in splits.items():
    split=result['splits'][name]
    evaluation=result['evaluations'][name]
    saved_rows=evaluation['rows']
    assert split['families']==sorted(owners[name])
    assert len(rows)==len(saved_rows)==split['contexts']==evaluation['contexts']
    pair_count=len(ns['_pairs'](rows))
    assert pair_count==split['preference_pairs']==evaluation['preference_pairs']
    for current,saved in zip(rows,saved_rows):
        assert all(current[field]==saved[field] for field in row_fields)
        s=saved['reward_model_raw_scores']
        assert len(s)==4 and all(math.isfinite(v) for v in s)
        assert saved['reward_model_best_action']==max(range(4),key=s.__getitem__)
    wins=sum(saved['reward_model_raw_scores'][winner]>saved['reward_model_raw_scores'][loser]
             for saved in saved_rows for winner,loser in saved['preference_pairs'])
    metric=evaluation['reward_model_pairwise_accuracy']
    assert wins==metric['numerator'] and pair_count==metric['denominator']
    assert math.isclose(wins/pair_count,metric['rate'],abs_tol=1e-14)
    measurements[name]=dict(families=len(owners[name]),contexts=len(rows),pairs=pair_count,
                            reconstructed_pair_wins=wins,stored_rate=metric['rate'])

pointers=['/device','/seed','/torch_version','/python_version','/code_sha256',
          '/results/config/reward_steps','/results/config/reward_batch_size',
          '/results/reward/unique_training_pairs','/results/reward/processed_pair_draws',
          '/results/reward/history/*/step','/results/reward/history/*/loss_before_update']
for name in splits:
    pointers += ['/results/splits/'+name+'/'+k for k in ['families','contexts','preference_pairs']]
    pointers += ['/results/evaluations/'+name+'/'+k for k in ['contexts','preference_pairs','reward_model_pairwise_accuracy']]
    pointers += ['/results/evaluations/'+name+'/rows/*/'+k for k in row_fields+['reward_model_raw_scores','reward_model_best_action']]
environment=dict(python=platform.python_version(),python_executable=sys.executable,
                 torch=str(torch.__version__),torch_git=str(torch.version.git_version),
                 device='cpu',cuda_build=str(torch.version.cuda),cuda_available=str(torch.cuda.is_available()),
                 cpu_threads=str(torch.get_num_threads()))
output=dict(original_fence_command=command,original_fence_exit_code=run.returncode,
            original_fence_stdout=run.stdout,variants=variants,mean_reduction=dict(shape=[2,2],
            denominator=4,expected=expected_mean,observed=mean_loss),input_contract_errors=errors,
            gradient_check=dict(chosen=c.grad.tolist(),rejected=r.grad.tolist(),parameter_updates=0),
            feature_contract=dict(context_shape=list(x.shape),model_input_shape=[3,4,8],
            score_shape=list(scores.shape),text_input=False,frozen=True),
            samples=samples,measurements=measurements,reward_history=history,
            processed_reward_pair_draws=300*64,provenance=provenance,raw_json_pointers=pointers,
            limits='No run_posttraining call, no existing weights loaded, no new model evaluation or full training.')
(A/'execution/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
(A/'execution/cpu-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(output,ensure_ascii=False,indent=2))
