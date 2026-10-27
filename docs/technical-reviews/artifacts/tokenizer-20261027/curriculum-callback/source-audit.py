"""Read-only source receipts and excerpt audit; no upstream training execution."""
from pathlib import Path
import hashlib,json,sys
A=Path(__file__).parent
x=json.loads((A/'upstream-inspection.json').read_text())
rows=[]
for r in x['inspections']:
 p=Path('/workspace/work/tokenizer-dynamic-20261027/curriculum-callback/sources')/r['file'];data=p.read_bytes();lines=data.decode().splitlines()
 assert hashlib.sha256(data).hexdigest()==r['sha256']
 for e in r['short_excerpts']:assert lines[e['line']-1]==e['text']
 rows.append({'file':r['file'],'sha256_matches':True,'locator':r['inspected_locator'],'supports':r['supports']})
y={'command':'uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/source-audit.py','environment':{'python':sys.version.split()[0],'device':'cpu'},'result':'29 fixed-version original files and saved minimal excerpts match inspected source bytes. This records source reading, not upstream training or performance.','inspections':rows}
(A/'source-audit.json').write_text(json.dumps(y,ensure_ascii=False,indent=2)+'\n')
print(y['result'])
