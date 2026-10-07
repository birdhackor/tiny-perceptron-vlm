import json,hashlib,torch
from tiny_perceptron.model import TinyLM,ModelConfig
p='docs/course-experiments/results/moe.json'
r=json.load(open(p));v=r['results']['variants']['dense_active_top1']
print('raw_sha',hashlib.sha256(open(p,'rb').read()).hexdigest())
print('revision',r['revision'],'torch',r['torch_version'],'gpu',r['gpu'])
print('dataset_records', {k:x['records'] for k,x in r['results']['dataset'].items()})
m=TinyLM(ModelConfig(**v['model']['config']))
print('ffn_each',sum(p.numel() for p in m.blocks[0].ffn.parameters()),'allparams',sum(p.numel() for p in m.parameters()))
print('training_keys',v['training'].keys())
for k in ['validation','test']:
 t=v['heldout'][k];print(k,t)
 print('quotient',t['nll_sum']/t['effective_tokens'])
print('requested_steps',v['training']['requested_steps'],'actual_updates',v['training'].get('optimizer_updates'))
up=torch.tensor([[1.,0.,1.],[0.,1.,1.]]);down=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
x=torch.tensor([[1.,2.],[2.,1.]])
z=x.repeat((3,1));print('six',len(z),up.numel()+down.numel(),(torch.relu(z@up)@down).tolist())
