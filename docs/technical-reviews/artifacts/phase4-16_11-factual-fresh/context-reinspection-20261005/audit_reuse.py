import hashlib,json,re,platform
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
PRIOR=BASE.parent
TASK='/root/phase4_factual_coordinator/factual_16_11'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
own_path=ROOT/'docs/technical-reviews/16.11.json'
own=json.loads(own_path.read_text())
assert own['reviewer_task']==TASK
assert digest(own_path)==digest(BASE/'prior-report.opaque.json')
current=(BASE/'current-16.11.md').read_bytes()
old=(PRIOR/'original-execution/section.md').read_bytes()
assert current==old and hashlib.sha256(current).hexdigest()==own['source_sha256']
prior_ctx=(BASE/'prior-16.8.md').read_bytes();current_ctx=(BASE/'current-16.8.md').read_bytes()
before='兩条路'.encode();after='兩條路'.encode()
assert prior_ctx.count(before)==1
assert prior_ctx.replace(before,after)==current_ctx
old_main=prior_ctx[:prior_ctx.index(b'<details>')]
main=(BASE/'current-16.8-main-slice.md').read_bytes()
assert old_main.replace(before,after)==main
print('primary_body_exact_reuse',own['source_sha256'])
print('context_delta_exact',json.dumps({'from':'兩条路','to':'兩條路','replacements':1,'classification':'simplified-to-traditional glyph, unchanged technical meaning','all_other_bytes_identical':True},ensure_ascii=False))
# Current primary Python fence and image references, without execution.
pattern=rb'```python\n(.*?)```'
fences=re.findall(pattern,current,re.S)
assert len(fences)==1 and fences[0]==(PRIOR/'original-execution/fence-1.py').read_bytes()
refs=re.findall(rb'!\[[^\]]*\]\(([^)]+)\)',current)
assert refs==[] and own['figure_sha256']=={}
print('own_fence_exact_reuse',hashlib.sha256(fences[0]).hexdigest(),'figure_map',own['figure_sha256'])
reuse=[]
for artifact in own['artifacts']:
 path=ROOT/artifact['path'];actual=digest(path)
 assert actual==artifact['sha256'],artifact['id']
 reuse.append({'artifact_id':artifact['id'],'path':artifact['path'],'prior_sha256':artifact['sha256'],'current_sha256':actual,'exact_match':True})
source_checks=[]
for source in own['sources']:
 if source['kind']=='repository_code':
  actual=digest(ROOT/source['path']);assert actual==source['sha256']
  source_checks.append({'source_id':source['id'],'path':source['path'],'recorded_sha256':source['sha256'],'current_sha256':actual,'exact_match':True})
current_inputs=[]
for original,frozen in [
 ('scripts/course_experiments/architecture.py','frozen/scripts/course_experiments/architecture.py'),
 ('docs/course-experiments/results/efficiency.json','frozen/docs/course-experiments/results/efficiency.json'),
 ('docs/review-tools/section_facts.py','frozen/docs/review-tools/section_facts.py'),
 ('scripts/check_technical_reviews.py','frozen/scripts/check_technical_reviews.py'),
]:
 a=digest(ROOT/original);b=digest(PRIOR/frozen)
 current_inputs.append({'current_repository_path':original,'current_sha256':a,'prior_snapshot_path':str((PRIOR/frozen).relative_to(ROOT)),'prior_snapshot_sha256':b,'exact_match':a==b})
print('current_repository_input_fingerprints',json.dumps(current_inputs,ensure_ascii=False))
assert all(item['exact_match'] for item in current_inputs)
meta=json.loads((BASE/'input-fingerprints.json').read_text())
record={'reviewer_task':TASK,'operation':'same-owner context-only version reinspection','python':platform.python_version(),'prior_canonical_sha256':digest(BASE/'prior-report.opaque.json'),'source_sha256':own['source_sha256'],'figure_sha256':{},'primary_fence_sha256':hashlib.sha256(fences[0]).hexdigest(),'necessary_context':meta,'delta':{'from':'兩条路','to':'兩條路','replacements':1,'all_other_bytes_identical':True,'technical_meaning_changed':False},'checked_artifacts':reuse,'checked_repository_sources':source_checks,'checked_current_repository_inputs':current_inputs,'reexecution':False,'authority_refetch':False,'gpu_or_training':False,'all_prior_supporting_file_hashes_match':True}
(BASE/'reuse-fingerprints.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('verified_reusable_artifacts',len(reuse),'repository_sources',len(source_checks))
print('PASS same-owner narrow reuse audit; no benchmark or compilation executed')
