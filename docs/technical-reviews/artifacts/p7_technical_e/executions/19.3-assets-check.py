import json,hashlib
from pathlib import Path
b=Path('outputs/selftrained-v2/data');m=json.loads(Path('docs/selftrained/v2-manifest.json').read_text());bad=[];ok=0
for e in m['assets']:
 p=b/e['path'];s=p.stat().st_size if p.exists() else None;h=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 if (s,h)==(e['bytes'],e['sha256']):ok+=1
 else:bad.append({'path':e['path'],'expected_bytes':e['bytes'],'actual_bytes':s,'expected_sha':e['sha256'],'actual_sha':h})
print('verified',ok,'total',len(m['assets']),'mismatches',len(bad));print(json.dumps(bad[:20],ensure_ascii=False,indent=2));p=Path('docs/technical-reviews/artifacts/p7_technical_e/executions/19.3-assets-mismatches.json');assert not p.exists();p.write_text(json.dumps(bad,indent=2)+'\n')
