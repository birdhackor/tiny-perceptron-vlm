import json,hashlib
from pathlib import Path
for name in ['quantization','qat']:
 p=Path('docs/course-experiments/results')/(name+'.json');d=json.loads(p.read_text())
 print(name,'raw_sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'revision',d['revision'])
 if name=='qat':
  assert d['results']['activation_quantization'] is False
  print('activation_quantization',d['results']['activation_quantization'])
 else:
  print('activation_quantization field present','activation_quantization' in d['results'])
  print('limitations',d['results']['limitations'][:2])
print('QAT flag is explicit; PTQ activation scope is established by raw limitations and QuantizedLinear float-input forward, not a nonexistent flag.')
