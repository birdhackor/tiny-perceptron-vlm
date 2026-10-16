"""Validate only this fresh whole-file README report; never load old reports or full gates."""
import hashlib
import importlib.util
import json
import platform
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
REPORT=ROOT/'docs/course-revision-20261016/reviews/technical-readme-engineering-clean-original-01.json'
CHECKER=ROOT/'scripts/check_technical_reviews.py'
spec=importlib.util.spec_from_file_location('readme_technical_schema',CHECKER)
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
data=json.loads(REPORT.read_bytes())
body=(ROOT/'README.md').read_bytes().decode('utf-8')
errors=checker._validate(ROOT,ROOT/'README.md','readme',body,data,
                         {data['reviewer_task']:['readme']},set())
hashes=json.loads((Path(__file__).parent/'checks-result.json').read_bytes())['source_sha256']
drift=[name for name,digest in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
result={
 'checked_at':datetime.now(UTC).isoformat(),
 'command':'PYTHONPATH=. .venv/bin/python '+str(Path(__file__).relative_to(ROOT)),
 'python':platform.python_version(),'device':'CPU','reviewer_task':data['reviewer_task'],
 'report':str(REPORT.relative_to(ROOT)),
 'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),
 'checker_sha256':hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
 'errors':errors,'execution_source_drift':drift,
 'whole_course_gate_run':False,
 'report_inputs':'Only this new JSON; reviewer_owners is the sole actual task mapped to readme; no prior reports or reader conclusions opened.'
}
(Path(__file__).parent/'validation-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
assert errors==[] and drift==[]
