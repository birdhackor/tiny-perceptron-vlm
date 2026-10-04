"""Actual official component functions; explicit wholefile contract, no fake lesson."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[6]
D = Path(__file__).parent
spec = importlib.util.spec_from_file_location('official_technical_checker', ROOT/'scripts/check_technical_reviews.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def components(path):
    r = json.loads(path.read_text())
    errors = []
    a = m._artifacts(ROOT, r.get('artifacts'), errors)
    s = m._sources(ROOT, r.get('sources'), a, errors)
    c = m._claims(r.get('claims'), s, a, errors)
    return r, c, errors

old, oldclaims, olderrors = components(D.parent/'report.json')
r, claims, errors = components(D/'report.json')
whole_errors = []
if r.get('schema_version') != 1 or r.get('review_stage') != 'technical':
    whole_errors.append('technical schema required')
if r.get('reviewer_task') != '/root/v4_review_coordinator/factual_whole_training_course' or not m._task(r.get('reviewer_task')) or r.get('reviewer_context') != 'fresh':
    whole_errors.append('actual original fresh identity required')
if 'lesson_id' in r:
    whole_errors.append('wholefile supplement must not fabricate lesson_id')
if r.get('complete_documents') != ['course/training.md'] or not m._text(r.get('assigned_document_scope')):
    whole_errors.append('complete scope required')
if r.get('source') != 'course/training.md' or r.get('current_document_sha256') != {'course/training.md':sha(ROOT/'course/training.md')}:
    whole_errors.append('current raw fullfile source mismatch')
if sha(ROOT/'course/training.md') != sha(D/'training.md.read-snapshot'):
    whole_errors.append('snapshot/current bytes mismatch')
for f,h in r.get('figure_sha256',{}).items():
    m._hash_file(ROOT,f,h,whole_errors,'figure')
for name in m.CHECKS:
    value = r.get('checks',{}).get(name)
    if not isinstance(value,dict) or value.get('status') != 'pass' or not m._text(value.get('details')):
        whole_errors.append(name+': explicitpass/details required for this substantive49claim/13figure scope')
    else:
        m._references(value.get('claim_ids',[]),claims,whole_errors,name+' claim_ids')
if any(not isinstance(i,dict) or i.get('status') != 'resolved' or not m._text(i.get('resolution')) for i in r.get('issues',[])):
    whole_errors.append('unresolved issue')
if r.get('verdict') != 'pass':
    whole_errors.append('own current verdict is not pass')
result = {
 'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round2/component_validate.py',
 'official_checker_path':'scripts/check_technical_reviews.py','official_checker_sha256':sha(ROOT/'scripts/check_technical_reviews.py'),
 'scope':'Actual official _artifacts/_sources/_claims; separate fullfile SHA/identity/figure/issues/fivechecks using official helpers. No numbered _validate or invented lesson ID. This is schema/local-evidence validation, not automated proof of reading or truth.',
 'preserved_first_final_report':{'path':(D.parent/'report.json').relative_to(ROOT).as_posix(),'sha256':sha(D.parent/'report.json'),'component_errors':olderrors},
 'current_report':{'path':(D/'report.json').relative_to(ROOT).as_posix(),'sha256':sha(D/'report.json'),'source_sha256':sha(ROOT/'course/training.md'),'claims':len(claims),'official_component_errors':errors,'wholefile_contract_errors':whole_errors},
 'result':'pass' if not errors and not whole_errors else 'revise'
}
(D/'component-validator-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors or whole_errors))
