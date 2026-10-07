import json
from pathlib import Path
from tiny_perceptron.model import TinyLM,ModelConfig
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['attributes']
print('runs',list(r['runs']))
for n,v in r['runs'].items():
 print(n,'keys',list(v));s=v['storage'];print('storage',s['parameter_count'],s['parameter_tensor_bytes']);print('training',v.get('training'))
for w in [16,32]:
 m=TinyLM(ModelConfig(width=w,layers=1,heads=2));n=sum(p.numel() for p in m.parameters());print('actual',w,n,4*n)
