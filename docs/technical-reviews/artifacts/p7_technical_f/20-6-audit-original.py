import json,hashlib,math
from pathlib import Path
p=Path('outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review/training.json');x=json.loads(p.read_text())
idx=json.loads(Path('docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json').read_text());entry=next(e for e in idx['files'] if e['path']=='review/training.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
a,b=x['initial_adapter_tensors'],x['final_adapter_tensors'];assert set(a)==set(b)
count=sum(math.prod(v['shape']) for v in a.values());changed=[n for n in a if a[n]['sha256_values']!=b[n]['sha256_values']]
f0,f1=x['frozen_parameter_samples_initial'],x['frozen_parameter_samples_final'];assert set(f0)==set(f1)
print('original SHA',entry['sha256']);print('rank',x['lora_rank'],'lr',x['learning_rate'],'targets',x['lora_targets']);print('shape counted parameters',count,'formula',28*8*((2048+2048)+(2048+1024)));print('changed adapter tensors',len(changed),'/',len(a));print('frozen sampled tensors',len(f0),'indices',sum(len(v['flat_indices']) for v in f0.values()),'equal',f0==f1);print('scope',x['frozen_check_scope'])
size=next(e for e in idx['files'] if e['path']=='review/adapter/adapter_model.safetensors');print('historical artifact index bytes',size['bytes'],'decimal MB',size['bytes']/1e6,'binary independently present?',(p.parent/'adapter/adapter_model.safetensors').exists())
assert count==1605632 and len(changed)==112 and f0==f1 and x['learning_rate']==0.00003
