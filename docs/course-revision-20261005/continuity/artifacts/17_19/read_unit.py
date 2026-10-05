import json, re, sys
from pathlib import Path
root=Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/course-revision-20261005/continuity'
pages={p['page_id']:p for p in json.loads((base/'inventory.json').read_text())['pages']}
p=pages[sys.argv[1]]
body=(root/p['snapshot']).read_text()
if len(sys.argv)>2:
    units=re.split(r'(?=^## )',body, flags=re.M)
    units=[s for s in units if s.strip()]
    n=int(sys.argv[2]);body=units[n]
    print('UNIT',n,'OF',len(units))
print(body)
