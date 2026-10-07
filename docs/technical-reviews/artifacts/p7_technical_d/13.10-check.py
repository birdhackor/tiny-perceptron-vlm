import json,torch,hashlib
from tiny_perceptron.posttraining import preference_loss,FiniteRewardModel
from scripts.course_experiments.posttraining import build_records,split_records
for c in (0.,2.):
 x=torch.tensor([c]);y=torch.tensor([1.]);print('reject1',c,torch.sigmoid(x-y).item(),preference_loss(x,y).item(),'shift10same',torch.allclose(preference_loss(x,y),preference_loss(x+10,y+10)))
rs=[r for r in build_records() if r['operands']==[1,2]]
for r in rs:print('card_example',r['mode'],r['candidates'],'expected',r['expected_action'],'features',r['features'])
f=torch.tensor([r['features'] for r in rs]);rm=FiniteRewardModel().requires_grad_(False).eval();print('RM_numeric_input',tuple(f.shape),'output',tuple(rm(f).shape),'output_requires_grad',rm(f).requires_grad)
parts=split_records(build_records());assert all(not {r['family'] for r in rows}.intersection({r['family'] for r in others}) for s,rows in parts.items() for t,others in parts.items() if s!=t)
j=json.load(open('docs/course-experiments/results/posttraining.json'));print('recorded_reward_steps',j['results']['reward']['history'][-1]['step'],'schedule_completed',j['results']['schedule_completed'])
for p in ('scripts/course_experiments/posttraining.py','tiny_perceptron/posttraining.py'):
 h=hashlib.sha256(open(p,'rb').read()).hexdigest();print('source_sha',p,h,'recorded',j['code_sha256'].get(p),'same',h==j['code_sha256'].get(p))
print('torch',torch.__version__)
