import json,math
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM,ModelConfig
t=ByteTokenizer();r=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
for n,d in r.items():
 m=TinyLM(ModelConfig(**d['model']['config']));print(n,'params_actual',sum(p.numel() for p in m.parameters()),'steps/updates/targets',d['training']['steps'],d['training']['optimizer_updates'],d['training']['effective_tokens'])
 for split in ['validation','test']:
  h=d['heldout'][split];matches=0;ended=0;targets=0
  for s in h['samples']:
   ids=s['generated_ids'];body=ids[:ids.index(t.eos_id)] if t.eos_id in ids else ids;ok=body==t.encode(s['expected']);assert ok==s['exact'];matches+=ok;ended+=t.eos_id in ids;targets+=len(t.encode(s['expected']))+1
  assert matches==h['matches'] and targets==h['effective_tokens'];v=h['nll_sum']/targets;print(split,'NLL',round(v,5),'sum/targets',h['nll_sum'],targets,'matches',matches,'/',len(h['samples']),'EOS',ended);assert abs(v-h['nll'])<1e-12
print('toyNLLprob.5',-math.log(.5))
