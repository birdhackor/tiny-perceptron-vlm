from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[6]
D=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('official_checker',ROOT/'scripts/check_technical_reviews.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r=json.loads((D/'report.json').read_text())
errors=[]
a=m._artifacts(ROOT,r.get('artifacts'),errors)
s=m._sources(ROOT,r.get('sources'),a,errors)
c=m._claims(r.get('claims'),s,a,errors)
whole=[]
if r.get('complete_documents')!=['course/training.md'] or r.get('source')!='course/training.md' or r.get('current_document_sha256')!={'course/training.md':sha(ROOT/'course/training.md')} or sha(D/'training.md.read-snapshot')!=sha(ROOT/'course/training.md'):
    whole.append('currentwholefile scope/rawSHA mismatch')
if r.get('schema_version')!=1 or r.get('review_stage')!='technical' or r.get('reviewer_task')!='/root/v4_review_coordinator/factual_whole_training_course' or r.get('reviewer_context')!='fresh' or not m._task(r.get('reviewer_task')) or 'lesson_id' in r:
    whole.append('actualsameownertechnical identity orwholefile contract mismatch')
for f,h in r.get('figure_sha256',{}).items():m._hash_file(ROOT,f,h,whole,'figure')
for name in m.CHECKS:
    v=r.get('checks',{}).get(name)
    if not isinstance(v,dict) or v.get('status')!='pass' or not m._text(v.get('details')):whole.append(name+': status/details invalid')
    else:m._references(v.get('claim_ids',[]),c,whole,name+' claim_ids')
if any(not isinstance(i,dict) or i.get('status')!='resolved' or not m._text(i.get('resolution')) for i in r.get('issues',[])):whole.append('unresolvedissue')
if r.get('verdict')!='pass':whole.append('owncurrentverdict notpass')
out={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-course/round3/component_validate.py','official_checker_sha256':sha(ROOT/'scripts/check_technical_reviews.py'),'scope':'Actualofficial _artifacts/_sources/_claims; officialhelpers/semantics forfullfileSHA,identity,figures,issues,fivechecks. No fake numberedlesson. Schema/localhash validation doesnotprovetruth/genuineread.','current_report_path':(D/'report.json').relative_to(ROOT).as_posix(),'current_report_sha256':sha(D/'report.json'),'current_source_sha256':sha(ROOT/'course/training.md'),'round2_preserved_report_sha256':sha(D.parent/'round2/report.json'),'claims':len(c),'official_component_errors':errors,'wholefile_contract_errors':whole,'result':'pass' if not errors and not whole else 'revise'}
(D/'component-validator-output.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors or whole))
