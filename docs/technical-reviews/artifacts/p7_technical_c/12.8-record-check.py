import json,sys
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/encoders-raw.json').read_text())['results']['audio']
print('python',sys.version.split()[0]);print('config/classes',r['config'],r['classes'])
groups={}
for split,x in r['data']['splits'].items():
 rows=x['records'];groups[split]={y['family'] for y in rows}
 assert len(rows)==x['count'];assert all(y['answer']==('high' if y['frequency']>300 else 'low') for y in rows)
 print('split',split,'records',len(rows),'families',sorted(groups[split]))
assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
for split in ['before','validation','test']:
 x=r[split];c=sum(y['target']==y['predicted'] for y in x['samples']);assert c==x['correct'] and len(x['samples'])==x['count'];print('recount',split,c,x['count'])
h=r['training']['history'];assert len(h)==r['training']['steps'];print('training-history',len(h),'effective-targets',sum(x['effective_targets'] for x in h),'nonzero-grad-steps',sum(x['grad_norm']>0 for x in h));print('training-loss',r['training']['initial_loss'],r['training']['final_loss'],'weights-changed',r['training']['weights_changed'])
