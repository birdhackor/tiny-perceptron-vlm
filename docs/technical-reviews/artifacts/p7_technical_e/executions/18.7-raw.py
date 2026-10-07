import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']
for n,v in r.items():print(n,'frozen',v['teacher_frozen_and_unchanged'],'cache_seconds',v['teacher_cache']['seconds']);assert v['teacher_frozen_and_unchanged'] is True
