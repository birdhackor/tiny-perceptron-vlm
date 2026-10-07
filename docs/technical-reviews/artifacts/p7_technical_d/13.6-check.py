import json,torch
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.model import TinyLM,ModelConfig
j=json.load(open('docs/course-experiments/results/dpo.json'))['results'];style=json.load(open('docs/course-experiments/results/style.json'))['results']
for name,run in [('before',{'preference':j['before'],'arithmetic':style['content_evaluation']})]+list(j['runs'].items()):
 if name!='before':print('settings',name,run['beta'],run['training']['steps'],run['training']['parameters'])
 for split in ('validation','test'):
  p=run['preference'][split];rs=p['samples'];absolute=sum(r['policy_margin']>0 for r in rs);relative=sum(r['relative_margin']>0 for r in rs);mean=sum(r['relative_margin'] for r in rs)/len(rs)
  a=run['arithmetic'][split];gs=a['samples'];exact=sum(r['generated']==r['expected'] for r in gs);eos=sum(r['eos'] for r in gs)
  assert absolute==p['chosen_higher_absolute_probability'] and relative==p['relative_preference_improved'] and abs(mean-p['mean_relative_margin'])<1e-10 and exact==a['matches']
  print(name,split,'n',len(rs),'absolute',absolute,'relative',relative,'mean',mean,'generated_correct',exact,'EOS',eos,len(gs))
  if split=='test':print('2+2',next(r['generated'] for r in gs if r['messages'][0]['content']=='2+2=?'))
m=TinyLM(ModelConfig(width=64,layers=2));print('config',m.config,'parameters',sum(p.numel() for p in m.parameters()))
for beta in (.1,1.,5.):
 c=torch.tensor([-6.],requires_grad=True);loss=dpo_loss(c,torch.tensor([-4.]),torch.tensor([-3.]),torch.tensor([-3.]),beta);loss.backward();print('negative_margin',beta,loss.item(),c.grad.item())
print('torch',torch.__version__)
