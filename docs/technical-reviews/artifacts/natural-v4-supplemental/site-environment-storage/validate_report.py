import hashlib,importlib.util,json,platform,sys
from pathlib import Path
root=Path.cwd();folder=Path('docs/technical-reviews/artifacts/natural-v4-supplemental/site-environment-storage');p=folder/'report.json';r=json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('unchanged_technical_checker',root/'scripts/check_technical_reviews.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=[];a=m._artifacts(root,r.get('artifacts'),e);s=m._sources(root,r.get('sources'),a,e);c=m._claims(r.get('claims'),s,a,e)
component_errors=list(e);own=[]
if r.get('schema_version')!=1 or r.get('review_stage')!='technical' or r.get('reviewer_context')!='fresh' or not m._task(r.get('reviewer_task')):own.append('Invalid identity/schema')
if 'lesson_id' in r:own.append('A numbered lesson_id was fabricated')
expected=['docs/environment.md','docs/asset-storage.md']
if r.get('complete_documents')!=expected or r.get('assigned_document_scope')!=expected:own.append('Complete document scope mismatch')
for path in expected:
 if r['current_document_sha256'].get(path)!=hashlib.sha256((root/path).read_bytes()).hexdigest():own.append('Assigned source hash mismatch: '+path)
body='\n'.join((root/path).read_text() for path in expected)
if r.get('figure_sha256')!={} or '.svg)' in body:own.append('Unexpected figure mismatch')
if r.get('verdict')!='pass' or r.get('issues')!=[] or any(x.get('status')!='verified' for x in r['claims']):own.append('Verdict/claims/issues not closed')
for name in m.CHECKS:
 check=r.get('checks',{}).get(name,{})
 if not m._text(check.get('details')):own.append('Missing '+name+' details')
 if check.get('status')!=('not_applicable' if name=='figure_consistency' else 'pass'):own.append('Incorrect '+name+' state')
 m._references(check.get('claim_ids'),c,own,name)
output={'command':'.venv/bin/python '+str(folder/'validate_report.py'),'exit_code':int(bool(e+own)),'checker_path':'scripts/check_technical_reviews.py','checker_sha256':hashlib.sha256((root/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'environment':{'python':platform.python_version(),'device':'CPU format/hash validation'},'components_executed_read_only':['_artifacts','_sources','_claims'],'component_errors':component_errors,'nonnumbered_supplement_errors':own,'counts':{'claims':len(c),'sources':len(s),'artifacts':len(a),'documents':2},'limits':'Numbered _validate/check not invoked. Source body/hash, whole-document scope, identity syntax, own five check statuses and figure absence independently checked without inventing lesson IDs. This validates local report structure/evidence/hash consistency, not the truth of all prose or reviewer independence dispatch.'}
(folder/'checker-receipt.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output,ensure_ascii=False,indent=2));sys.exit(output['exit_code'])
