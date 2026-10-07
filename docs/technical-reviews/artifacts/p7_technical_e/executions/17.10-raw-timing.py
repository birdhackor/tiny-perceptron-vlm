import json,math,hashlib
from pathlib import Path
p=Path('docs/course-experiments/results/quantization.json');d=json.loads(p.read_text())
print('raw_sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'gpu',d['gpu'],'torch',d['torch_version'])
expected={'fp32':(2.405,2.371,335872),'packed4':(4.958,4.882,376320),'packed8':(2.768,2.814,359936)}
for name,v in d['results']['runs'].items():
 t=v['timing'];pre=t['prefill_seconds']*1000;per=t['decode_seconds_per_token']*1000
 assert t['repetitions']==3 and t['prompt_tokens']==46 and t['decode_tokens']==8
 assert math.isclose(t['decode_seconds']/8,t['decode_seconds_per_token'],abs_tol=1e-15)
 assert t['cuda_peak_allocated']-t['cuda_allocated_before']==t['cuda_additional_peak_bytes']
 a,b,c=expected[name];assert abs(pre-a)<.0005 and abs(per-b)<.0005 and t['cuda_additional_peak_bytes']==c
 print(name,'prefill_ms',pre,'decode_ms_per_fixed_step',per,'additional_peak',c,'start_allocated',t['cuda_allocated_before'],'peak',t['cuda_peak_allocated'])
print('Historical JSON conversion only; no current GPU timing, no isolation or driver/reserved-memory measurement.')
