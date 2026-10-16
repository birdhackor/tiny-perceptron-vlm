"""Validate and seal a separate freeze16 owner callback; retain original evidence hashes."""
import hashlib
import json
import platform
from pathlib import Path

from scripts.check_technical_reviews import _validate
from scripts.reading_time import lesson_slices

root=Path.cwd()
base=Path('docs/technical-reviews/artifacts/revision-20261016/backfill')
report_path=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-callback-01.json')
report=json.loads(report_path.read_bytes())
body=lesson_slices((root/'course/chapters/19.md').read_text())['19.13']
errors=_validate(root,root/'course/chapters/19.md','19.13',body,report,{'/root/backfill_tech_19_13':['19.13']},set())
assert errors == [], errors
old_seal=json.loads((base/'19.13-self-check.json').read_bytes())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old_seal['sealed_sha256'].items())
paths=[report_path,*sorted(base.glob('19.13-callback-*'))]
record={'reviewer_task':'/root/backfill_tech_19_13','lesson_id':'19.13','current_context':'informed',
 'command':'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-callback-format-check-01.py',
 'environment':{'python':platform.python_version(),'device':'cpu'},
 'validator':'scripts.check_technical_reviews._validate',
 'validator_sha256':hashlib.sha256((root/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),
 'result':'_validate returned []; complete callback report passed format/source/artifact checks. Independent owner judgment resolved the two issues; checker alone is not substantive verification.',
 'validation_errors':errors,'original_seal_still_matches':True,
 'sealed_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if not p.name.endswith('self-check-01.json')}}
out=base/'19.13-callback-self-check-01.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(report_path),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'verdict':report['verdict'],'validation_errors':errors,'original_unchanged':True,'self_check':str(out)},ensure_ascii=False,indent=2))
