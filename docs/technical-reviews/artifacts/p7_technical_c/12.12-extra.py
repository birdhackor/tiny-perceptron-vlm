import json,hashlib
from pathlib import Path
print('current-shape-valid-ignore',tuple(out['logits'].shape),int((out['labels']!=-100).sum()),int((out['labels']==-100).sum()));print('gradnorms',float(model.image_projector.weight.grad.norm()),float(model.audio_projector.weight.grad.norm()))
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/joint-raw.json').read_text())['results'];n=r['evaluations']['none']['samples']
for key,a in [('validation',r['validation'])]+list(r['evaluations'].items()):
 rows=a['samples'];full=sum(x['target']==x['generated'] for x in rows);shape=sum(x['target'].split(',')[0]==x['generated'].split(',')[0] for x in rows);pitch=sum(x['target'].split(',')[1]==x['generated'].split(',')[1] for x in rows)
 assert (full,shape,pitch)==(a['correct'],a['shape_correct'],a['pitch_correct']);print('recount',key,len(rows),shape,pitch,full)
 if key.startswith('swap'):
  idx=0 if key=='swap_image' else 1
  changed=sum(x['generated'].split(',')[idx]!=n[i]['generated'].split(',')[idx] and x['generated'].split(',')[1-idx]==n[i]['generated'].split(',')[1-idx] for i,x in enumerate(rows));print('pairs-only-field-changed',key,changed)
e=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/encoders-raw.json').read_text())['results']['audio']['data']['splits']['train']['records'];trained={x['frequency'] for x in e};test={x['frequency'] for x in r['data']['splits']['test']['records']};print('joint-test-vs-audio-pretrain-frequencies',sorted(test&trained));print('history',len(r['training']['history']),sum(x['effective_targets'] for x in r['training']['history']),sum(x['grad_norm']>0 for x in r['training']['history']))
