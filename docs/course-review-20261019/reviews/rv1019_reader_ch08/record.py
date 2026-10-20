import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('docs/course-review-20261019/reviews/rv1019_reader_ch08')
TASK = '/root/rv1019_reader_ch08'
data = json.load(sys.stdin)
page = data.pop('page')
session = data.pop('session')
checks = []
for row in data.pop('checks'):
    concept, identity, mechanism, need, usage, scope, quote, verdict = row
    checks.append(dict(concept=concept, identity=identity, mechanism=mechanism,
                       need=need, usage=usage, example_scope=scope,
                       source_quote=quote, verdict=verdict))
data['concept_checks'] = checks
data['reviewer_task'] = TASK
path = ROOT / 'drafts' / f"{page}-{data['unit_index']:03d}.json"
with path.open('x') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write('\n')
subprocess.run([sys.executable, 'docs/course-revision-20261016/tools/reader.py',
                'next', session, '--checkpoint', str(path)],
               check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
