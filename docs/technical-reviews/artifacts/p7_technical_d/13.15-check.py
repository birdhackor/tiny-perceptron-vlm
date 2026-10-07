import json,torch,hashlib
from scripts.course_experiments.posttraining import build_records,split_records
from scripts.course_experiments.common import records_sha256
from tiny_perceptron.posttraining import FiniteResponsePolicy,FiniteRewardModel,FiniteValueModel
raw=json.load(open('docs/course-experiments/results/posttraining.json'));r=raw['results'];spl=split_records(build_records(),42)
for k,v in spl.items():
 h=records_sha256(v);print('split',k,len(v),len(set(x['family'] for x in v)),h,h==r['splits'][k]['sha256']);assert h==r['splits'][k]['sha256']
for k,cls in [('policy',FiniteResponsePolicy),('reward_model',FiniteRewardModel),('value_model',FiniteValueModel)]:
 n=sum(p.numel() for p in cls().parameters());print('parameters',k,n);assert n==r['parameters'][k]
for name in ['sft','ppo']:
 rows=r['evaluations']['test']['rows']; n=sum(row['policies'][name]['chosen_action']=={'number':0,'explain':1,'missing':3}[row['mode']] for row in rows)
 print('historical',name,n,len(rows));assert n==r['evaluations']['test']['policies'][name]['greedy_full_request_success']['numerator']
c=r['config']; print('counts',c['sft_steps']*c['sft_batch_size'],c['reward_steps']*c['reward_batch_size'],c['ppo_rollout_batches']*c['ppo_rollout_size'],c['ppo_rollout_batches']*c['ppo_epochs_per_rollout'],c['ppo_rollout_batches']*c['ppo_rollout_size']*c['ppo_epochs_per_rollout'],'effective_tokens',r['effective_tokens'])
logits=torch.tensor([[.2,-.1,.8,.3]]);target=torch.tensor([0]);smoothed=torch.tensor([[.85,.05,.05,.05]])
a=torch.nn.functional.cross_entropy(logits,target,label_smoothing=.2);b=-(smoothed*logits.log_softmax(-1)).sum();print('smoothing',a.item(),b.item(),torch.allclose(a,b))
doc=torch.nn.CrossEntropyLoss.__doc__;i=doc.index('label_smoothing');print('official_class_doc',doc[i:i+580]);print('torch',torch.__version__)
