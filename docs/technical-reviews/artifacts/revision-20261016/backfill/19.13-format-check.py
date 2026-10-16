"""Self-check the frozen-source report; a revise verdict intentionally cannot pass release gating."""
import hashlib
import json
import platform
from pathlib import Path

from scripts.check_technical_reviews import _validate
from scripts.reading_time import lesson_slices

root=Path.cwd()
report_path=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-original-01.json')
report=json.loads(report_path.read_bytes())
body=lesson_slices((root/'course/chapters/19.md').read_text())['19.13']
errors=_validate(root,root/'course/chapters/19.md','19.13',body,report,{'/root/backfill_tech_19_13':['19.13']},set())
expected=[
 '審閱要求修訂／尚未核實',
 'claim c10: 未核實／矛盾主張不能通過，應維持revise',
 'claim c11: 未核實／矛盾主張不能通過，應維持revise',
 '尚有未解決問題，不能記為pass',
 'factual_accuracy: 未通過或不適用的NA',
 'source_verification: 未通過或不適用的NA',
 'limitations: 未通過或不適用的NA',
]
assert errors == expected, errors
paths=[report_path,*sorted(Path('docs/technical-reviews/artifacts/revision-20261016/backfill').glob('19.13-*'))]
record={'reviewer_task':'/root/backfill_tech_19_13','lesson_id':'19.13',
 'command':'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-format-check.py',
 'environment':{'python':platform.python_version(),'device':'cpu'},
 'validator':'scripts.check_technical_reviews._validate',
 'validator_sha256':hashlib.sha256((root/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),
 'result':'All required schema/source/artifact/identity fields validate; the seven expected release-gating errors are exclusively the intentionally unresolved c10/c11 and revise verdict/checks. This is not a pass or automatic fact verification.',
 'validation_errors':errors,'unexpected_errors':[],
 'sealed_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if not p.name.endswith('self-check.json')}}
out=Path('docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-self-check.json')
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(report_path),'verdict':report['verdict'],'expected_release_gate_errors':len(errors),'unexpected_format_errors':0,'self_check':str(out)},ensure_ascii=False,indent=2))
