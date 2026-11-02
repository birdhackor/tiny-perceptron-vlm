"""Execute the guide's linked artificial 7.19 token-budget example only."""
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
path=ROOT/'course/chapters/07.md'
text=path.read_text()
match=re.search(r'^## 7\.19 .+$',text,re.M)
following=re.search(r'^## ',text[match.end():],re.M)
section=text[match.start():match.end()+following.start()]
blocks=re.findall(r'```python\n(.*?)\n```',section,re.S)
assert len(blocks)==1
code=blocks[0]
assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(ast.parse(code)))
scope={};output=io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code,'course/chapters/07.md#7.19','exec'),scope)
assert scope['T']==16 and scope['P']==12 and scope['T']-scope['P']==4
assert 'G=4：新增3個token' in output.getvalue()
assert 'G=2：新增2個token' in output.getvalue()
assert 'G=5：P+G=17 超過 T=16' in output.getvalue()
value={'created_at':datetime.now(timezone.utc).isoformat(),
    'environment':{'python':platform.python_version(),'device':'CPU pure Python; no imports in executed example'},
    'source':'course/chapters/07.md#7.19','section_sha256':hashlib.sha256(section.encode()).hexdigest(),
    'exact_code':code,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
    'stdout':output.getvalue(),'observed':{'T':scope['T'],'P':scope['P'],'remaining':scope['T']-scope['P']},
    'scope':'Artificial hand-written tokens; no neural model, formal final family, kernel bundle or calibration used.'}
out=Path(__file__).with_name('route-budget-current-v1.json')
if out.exists():raise RuntimeError('Refusing to overwrite saved current evidence')
out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(value,ensure_ascii=False))
