import copy,json,torch
from pathlib import Path
from dataclasses import asdict
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.adapters import base_state_sha256,load_lora_adapter
torch.manual_seed(0);m=TinyLM(ModelConfig(width=4,layers=1));digest=base_state_sha256(m.state_dict());p=Path('outputs/technical-checkpoints/p7_technical_b/adapter-probe8_13.pt')
payload={'format_version':'lora-v1','config':asdict(m.config),'base_sha256':digest,'scaling':'alpha/rank','adapter':{'output':{'a':torch.ones(2,4)*.01,'b':torch.ones(264,2)*.01,'rank':2,'alpha':2}}};torch.save(payload,p)
valid=load_lora_adapter(copy.deepcopy(m),p);assert valid['base_sha256_verified'] and valid['scaling']=='alpha/rank'
out={'valid_load':valid,'rejections':{}}
for field,value in [('scaling','alpha/sqrt(rank)'),('base_sha256','0'*64)]:
 bad=copy.deepcopy(payload);bad[field]=value;torch.save(bad,p)
 try:load_lora_adapter(copy.deepcopy(m),p)
 except ValueError as e:out['rejections'][field]=str(e)
 else:raise AssertionError(field)
torch.save(payload,p)
l=LoRALinear(torch.nn.Linear(4,3),rank=2,alpha=2)
with torch.no_grad():l.b.fill_(1)
d=l.merged_weight()-l.base.weight;l.alpha=6;triple=l.merged_weight()-l.base.weight
err=(triple-3*d).abs().max().item();assert torch.allclose(triple,3*d,atol=1e-6,rtol=0)
out['alpha6_triple']={'error':err,'matches':True};out['fingerprint_after_one_weight_change']=None
with torch.no_grad():m.output.weight[0,0]+=1
changed=base_state_sha256(m.state_dict());assert changed!=digest;out['fingerprint_after_one_weight_change']=changed
print(json.dumps(out,ensure_ascii=False,indent=2))
