import ast,hashlib,json,math,sys,wave
from pathlib import Path
import numpy as np
import torch
p=Path('docs/technical-reviews/artifacts/p7_technical_c/originals/0_jackson_5.wav')
with wave.open(str(p)) as w:
 rate=w.getframerate();n=w.getnframes();values=np.frombuffer(w.readframes(n),dtype='<i2').astype(np.float32)/32768
s=Path('scripts/course_experiments/modalities.py').read_text();f=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='_resample_8_to_16');exec(compile(ast.Module(body=[f],type_ignores=[]),str('actual-_resample_8_to_16'), 'exec'))
torch.set_num_threads(1);y=_resample_8_to_16(values)
r=json.loads(Path('docs/course-experiments/results/real_modal.json').read_text())['results']['fsdd']['resampling'][0]
print('versions',sys.version.split()[0],torch.__version__,np.__version__,'CPU')
print('source',hashlib.sha256(p.read_bytes()).hexdigest(),n,rate,n/rate)
print('actual original resampler',len(y),16000,len(y)/16000)
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']
assert n==r['samples_before'] and len(y)==r['samples_after']==9182
print('record counts match True; 4591/8000 == 9182/16000',n/rate==len(y)/16000)
