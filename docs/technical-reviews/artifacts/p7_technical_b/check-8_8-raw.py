import json,hashlib,torch
from pathlib import Path
from torch import nn
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.tokenization import ByteTokenizer
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_b/sources/lora-result.json').read_text());d=r['results']
assert hashlib.sha256(Path('docs/technical-reviews/artifacts/p7_technical_b/sources/behavior-a7cdffec.py').read_bytes()).hexdigest()==r['code_sha256']['scripts/course_experiments/behavior.py']
assert hashlib.sha256(Path('tiny_perceptron/alignment.py').read_bytes()).hexdigest()==r['code_sha256']['tiny_perceptron/alignment.py']
t=ByteTokenizer();out={}
for name,evaluation in [('concise',d['runs']['concise']['evaluation']['test']),('vivid',d['runs']['vivid']['evaluation']['test']),('full',d['full_sft']['evaluation']['test'])]:
 counts={'content':0,'style':0,'eos':0};samples=[]
 for s in evaluation['samples']:
  ids=s['generated_ids'];assert t.decode(ids)==s['generated'] and ids[-1]==t.eos_id and all(i>=8 for i in ids[:-1])
  a,b=s['messages'][0]['content'].split('=')[0].split('+');expected=str(int(a)+int(b));g=s['generated'];vivid=name!='concise'
  content=g.split('，')[0]==expected if vivid else g==expected
  style='像把兩組積木合在一起再數。' in g if vivid else g.isdigit()
  assert content==s['content_correct'] and style==s['style_correct'] and (g==s['expected'])==s['exact']
  counts['content']+=content;counts['style']+=style;counts['eos']+=s['eos']
  samples.append({'q':s['messages'][0]['content'],'expected_number':expected,'generated':g,'content':content,'style':style,'ids_roundtrip':True})
 assert counts=={'content':0,'style':6 if name=='vivid' else 7,'eos':7}
 out[name]={'counts':counts,'records':7,'samples':samples}
for name,run in d['runs'].items():
 assert run['training']['steps']==450 and len(run['layers'])==11 and run['base_parameters_unchanged'] and run['training']['trainable_parameters']==9504
 assert run['training']['first_gradients']['all_a_zero'] and run['training']['first_gradients']['some_b_nonzero']
out['setup']={'rank':d['rank'],'alpha':d['alpha'],'layers':d['runs']['vivid']['layers'],'trainable':9504,'steps_each':450,'effective_targets':{name:run['training']['effective_tokens'] for name,run in d['runs'].items()},'unchanged_base_reported':True}
torch.manual_seed(0);l=LoRALinear(nn.Linear(16,12),rank=4,alpha=4);x=torch.randn(3,16)
assert sum(p.numel() for p in l.parameters() if p.requires_grad)==112 and torch.equal(l(x),l.base(x)) and l.alpha/l.rank==1
out['variation_rank4_alpha4']={'parameters':112,'scaling':1,'initial_equal':True}
print(json.dumps(out,ensure_ascii=False,indent=2))
