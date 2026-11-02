"""Clarify the JSONL whitespace difference observed during the independent input audit."""
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import platform

HERE=Path(__file__).resolve().parent
path=Path('/workspace/work/reference/omni-technical-t-5-v1/pku-first100-response.json')
rows=[x['row'] for x in json.loads(path.read_text())['rows']]
results={}
for name,options in [('default_spacing',{}),('compact_spacing',{'separators':(',',':')})]:
    body=''.join(json.dumps(row,ensure_ascii=False,**options)+'\n' for row in rows).encode()
    results[name]={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
expected='c4a88d08ef7456669766f1a3b908f82d918be5625a24794d9843d7f6df1d552b'
result={'created_at':datetime.now(UTC).isoformat(),'environment':{'python':platform.python_version(),'device':'cpu'},
    'input_response_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),
    'serialization':results,'manifest_jsonl_sha256':expected,
    'compact_matches_manifest':results['compact_spacing']['sha256']==expected,
    'clarification':'The earlier raw_jsonl_matches_manifest=false used default JSON spaces. '
       'Compact UTF-8 JSONL reproduces the exact manifest bytes, independently of the already matching 48/6/6 split hashes.'}
with (HERE/'pku-serialization-execution.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result,indent=2))
