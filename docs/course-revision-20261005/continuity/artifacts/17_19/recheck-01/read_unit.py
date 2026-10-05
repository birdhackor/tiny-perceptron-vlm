import json,re,sys
from pathlib import Path
root=Path('/workspace/tiny-perceptron-vlm');base=root/'docs/course-revision-20261005/continuity'
pages={p['page_id']:p for p in json.loads((base/'revised-01/inventory.json').read_text())['pages']}
body=(root/pages[sys.argv[1]]['snapshot']).read_text()
if len(sys.argv)>2:
 units=[s for s in re.split(r'(?=^## )',body,flags=re.M) if s.strip()];n=int(sys.argv[2]);body=units[n];print('UNIT',n,'OF',len(units))
print(body)
