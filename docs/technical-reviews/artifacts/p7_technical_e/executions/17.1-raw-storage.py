import json,hashlib
from pathlib import Path
p=Path('docs/course-experiments/results/quantization.json');d=json.loads(p.read_text())
print('source_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
for name,v in d['results']['runs'].items():
 s=v['storage'];b=sum(s['buffers'].values());t=s['parameter_tensor_bytes']+b
 assert b==s['buffer_tensor_bytes'] and t==s['tensor_bytes']
 assert s['file_bytes']-t==s['file_overhead_bytes']
 a=next(a for a in d['artifacts'] if Path(s['checkpoint']).name==a['path'])
 assert a['bytes']==s['file_bytes']
 if name=='fp32': assert s['parameter_count']*4==t
 print(name,{k:s[k] for k in ['parameter_count','parameter_tensor_bytes','buffer_tensor_bytes','tensor_bytes','file_bytes','file_overhead_bytes','optimizer_in_deployment_file']})
 print('recorded_file_sha256',a['sha256'])
print('Historical JSON arithmetic only; checkpoint files not remeasured and training not rerun.')
