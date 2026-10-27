import json,hashlib
from pathlib import Path
from datetime import datetime,UTC
from scripts.export_course import checked_notebook
r=Path.cwd(); idx=json.load(open('course/lesson-index.json')); results=[]
for x in idx:
 original=json.loads((r/x['notebook']).read_bytes()); target=Path('/workspace/work/tokenizer-dynamic-20261027/notebooks')/Path(x['notebook']).relative_to('notebooks'); executed=checked_notebook(original,target)
 results.append({'id':x['id'],'source_sha256':hashlib.sha256((r/x['notebook']).read_bytes()).hexdigest(),'executed_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'code_cells':sum(c['cell_type']=='code' for c in executed['cells'])})
receipt={'created_at':datetime.now(UTC).isoformat(),'checked':len(results),'summary':json.load(open('/workspace/work/tokenizer-dynamic-20261027/notebooks/validation-kernel.json')),'checks':'All current original notebook cell types/sources match existing executed CPU copies; every code cell has execution_count and no error output. Existing outputs inspected, no notebook rerun.','notebooks':results}
Path('docs/technical-reviews/artifacts/tokenizer-20261027/readme-final/notebooks-original-01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print({'checked':len(results),'summary':receipt['summary']})
