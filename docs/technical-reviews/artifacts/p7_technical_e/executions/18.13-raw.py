import json,hashlib
from pathlib import Path
p=Path('docs/technical-reviews/artifacts/p7_technical_e/sources/multimodal_distillation.json');d=json.loads(p.read_text());r=d['results']['tasks']['joint']['runs']['ce_kl'];a=r['test']['samples'];b=r['blank_image_test']['samples'];assert len(a)==len(b)==12
assert all(x['row']==y['row'] and x['family']==y['family'] for x,y in zip(a,b))
print('raw_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
for x,y in zip(a,b):print(x['row'],x['target'],x['generated'],x['generated_ids']==y['generated_ids'])
print('same_ids',sum(x['generated_ids']==y['generated_ids'] for x,y in zip(a,b)),'denominator',len(a))
print('all_circle',all(x['generated'].split(',')[0]=='circle' for x in a))
print('pitch',r['test']['pitch_correct'],r['test']['examples'],'shape',r['test']['shape_correct'])
print('blank_audio_pitch',r['blank_audio_test']['pitch_correct'],r['blank_audio_test']['examples'])
for k,t in d['results']['tasks'].items():
 print(k,'frozen',t['teacher_frozen_and_unchanged'],'alignment',t['runs']['ce_kl']['training']['alignment'])
