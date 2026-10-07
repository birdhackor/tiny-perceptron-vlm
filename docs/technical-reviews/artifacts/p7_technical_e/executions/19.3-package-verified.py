import json,hashlib,tarfile
from pathlib import Path
from tiny_perceptron.selftrained.dataset import read_records
m=json.loads(Path('docs/selftrained/v2-manifest.json').read_text());arc=Path('outputs/selftrained-v2/data-publish/selftrained-v2.tar.gz');assert hashlib.sha256(arc.read_bytes()).hexdigest()==m['package']['sha256'];dest=Path('outputs/p7-technical-e-cache/data');assert not dest.exists();dest.mkdir(parents=True)
with tarfile.open(arc) as t:t.extractall(dest,filter='data')
for e in m['records']+m['assets']:
 p=dest/e['path'];assert p.stat().st_size==e['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
print('archive',m['package']['sha256'],'record_files_verified',len(m['records']),'all_assets_verified',len(m['assets']))
base=Path('outputs/selftrained/data');heldout=0
for e in m['records']:
 if e['path'].endswith(('-test.jsonl','-validation.jsonl')):
  p=base/e['path'];assert p.read_bytes()==(dest/e['path']).read_bytes();heldout+=1
print('heldout_jsonl_byte_identical_to_base',heldout)
r=json.loads((dest/'vision-train.jsonl').open().readline());t=dict(r);t['id']='own-leak-probe';t['split']='test';p=Path('docs/technical-reviews/artifacts/p7_technical_e/executions/19.3-leak-fixture.jsonl');assert not p.exists();p.write_text(json.dumps(r)+'\n'+json.dumps(t)+'\n')
try:read_records([p]);raise AssertionError('guard did not reject')
except ValueError as e:print('actual_rejection',str(e))
