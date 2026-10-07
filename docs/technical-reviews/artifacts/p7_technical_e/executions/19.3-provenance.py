import json,hashlib,itertools,tempfile
from pathlib import Path
from tiny_perceptron.selftrained.dataset import read_records
b=Path('outputs/selftrained-v2/data');m=json.loads(Path('docs/selftrained/v2-manifest.json').read_text());orig={}
for s in ['train','validation','test']:
 rs=[json.loads(l) for l in (b/('voice-'+s+'.jsonl')).read_text().splitlines()];native=[r for r in rs if 'augmentation' not in r];orig[s]={r['audio'] for r in native};print('original_voice',s,'recordings',len(orig[s]),'records',len(native));assert all(r['split']=='train' for r in rs if 'augmentation' in r)
print('assets_count',len(m['assets']))
for e in m['assets']:
 p=b/e['path'];assert p.stat().st_size==e['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
print('all_asset_size_and_sha_verified',len(m['assets']))
r=json.loads((b/'vision-train.jsonl').open().readline());t=dict(r);t['id']='own-leak-probe';t['split']='test'
p=Path('docs/technical-reviews/artifacts/p7_technical_e/executions/19.3-leak-fixture.jsonl');assert not p.exists();p.write_text(json.dumps(r)+'\n'+json.dumps(t)+'\n')
try:read_records([p]);raise AssertionError('guard did not reject')
except ValueError as e:print('actual_rejection',str(e))
