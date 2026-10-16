"""Validate and fingerprint this reviewer's completed original report only."""

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from scripts.check_technical_reviews import _validate

report_path = ROOT / 'docs/course-revision-20261016/reviews/technical-curriculum-backfill-original-01.json'
report = json.loads(report_path.read_text())
freeze = json.loads((ROOT/'docs/course-revision-20261016/freeze-14.json').read_text())
page = next(x for x in freeze['pages'] if x['page_id']=='curriculum')
body = (ROOT/page['snapshot']).read_text()
source = ROOT/'docs/curriculum.md'
errors = _validate(ROOT,source,'curriculum',body,report,{'/root/backfill_tech_curriculum':['curriculum']},set())
assert not errors,errors
assert source.read_text()==body
assert report['source_sha256']==page['source_sha256']
assert report['figure_sha256']==page['figures_sha256']
out = {
    'reviewer_task':'/root/backfill_tech_curriculum',
    'completed_at':datetime.now(UTC).isoformat(),
    'command':'.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/curriculum-validate-seal.py',
    'environment':{'python':platform.python_version(),'device':'cpu'},
    'format_validation_errors':errors,
    'frozen_page_and_current_page_identical':True,
    'frozen_figure_map_matches':True,
    'report':{'path':str(report_path.relative_to(ROOT)),'sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()},
    'artifact_sha256':{a['path']:a['sha256'] for a in report['artifacts']},
    'scope':'Actual evidence review completed before this seal. Checker only checks schema, references, identity and SHA; it is not itself proof of factual truth. No other reviews were opened.',
}
Path(__file__).with_name('curriculum-validation-seal.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
