"""Same-owner course source12 callback in continuity03: hashes, exact diffs, schema.
No neural, kernel, training, publication, receipt creation or closed-seal edits.
"""
import difflib
import hashlib
import importlib.util
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def ref(path):
    p=ROOT/path
    return {'path':path,'sha256':sha(p),'bytes':p.stat().st_size}


def checked(r):
    actual=ref(r['path']);assert actual['sha256']==r['sha256'],r
    return actual


def write(name,value):
    p=HERE/name
    assert not p.exists(),p
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


prior_ref=checked({'path':'docs/course-revision-20261029-omni/reviews/P5-actual/technical-course/technical-current-v2.json',
    'sha256':'c54d439648daddd2d5a384f2c118a6c09dc2ba6c70d129cc1637d6f59acd8e0e'})
prior=read(ROOT/prior_ref['path'])
old_side_ref=checked(prior['numbered_section_reviews'])
new_side_ref=checked({'path':'docs/technical-reviews/artifacts/omni-P5-numbered-v3/numbered-section-reviews-v1.json',
    'sha256':'5225b24a74b01fe824481d792c61d299f6400c3141c1bfd185ed196b432ec158'})
old_side,new_side=read(ROOT/old_side_ref['path']),read(ROOT/new_side_ref['path'])
new_canonical_ref=checked(new_side['canonical_source_freeze'])
assert new_canonical_ref['sha256']=='1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1'
old_canonical=read(ROOT/old_side['canonical_source_freeze']['path'])
new_canonical=read(ROOT/new_canonical_ref['path'])
new_numbered_ref=checked(new_side['section_freeze'])
assert new_numbered_ref['sha256']=='f7ba7cc4df7edfb693f2024f819e8886a0fabba56bdfb0b6b94f18c364551423'
old_numbered=read(ROOT/old_side['section_freeze']['path']);new_numbered=read(ROOT/new_numbered_ref['path'])
new_issue_ref=checked(new_side['issue_inventory'])
old_issues=read(ROOT/old_side['issue_inventory']['path']);new_issues=read(ROOT/new_issue_ref['path'])
assert old_issues['issues']==new_issues['issues']
repairs_ref=checked({'path':'docs/course-revision-20261029-omni/reviews/P5-actual/coherence-three-author-repairs-source12-v1.json',
    'sha256':'76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7'})
repairs=read(ROOT/repairs_ref['path'])
old_pages={p['page_id']:p for p in old_canonical['pages']};new_pages={p['page_id']:p for p in new_canonical['pages']}
changed_pages=[p for p in new_pages if old_pages[p]!=new_pages[p]]
assert set(changed_pages)=={'9.17','12.17','19.23'}
course=new_pages['course'];body=(ROOT/course['source']).read_bytes()
assert body==(ROOT/course['snapshot']).read_bytes()
assert sha(ROOT/course['source'])==prior['source_sha256']==course['source_sha256']
registered_sources=[{'id':s['id'],**checked(s)} for s in prior['sources'] if s['kind']=='repository_code']
prior_artifacts=[{'id':a['id'],**checked(a)} for a in prior['artifacts']]

diff_rows=[]
for change in repairs['changes']:
    lesson=change['page_id'];checked(change['before']);checked(change['after'])
    before=(ROOT/change['before']['path']).read_text();after=(ROOT/change['after']['path']).read_text()
    assert sha(ROOT/change['after']['path'])==new_pages[lesson]['source_sha256']
    actual=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
        fromfile='source11/'+lesson,tofile='source12/'+lesson,n=2))
    assert actual==change['diff'],lesson
    old_code=re.findall(r'```python\n(.*?)\n```',before,re.S)
    new_code=re.findall(r'```python\n(.*?)\n```',after,re.S)
    code_same=old_code==new_code
    assert code_same==(lesson!='9.17')
    assert old_pages[lesson]['figures_sha256']==new_pages[lesson]['figures_sha256']
    diff_rows.append({'lesson_id':lesson,'before':change['before'],'after':change['after'],
        'exact_diff':actual,'python_code_unchanged':code_same,
        'before_code_sha256':[hashlib.sha256(c.encode()).hexdigest() for c in old_code],
        'after_code_sha256':[hashlib.sha256(c.encode()).hexdigest() for c in new_code],
        'figures_unchanged':True})

old_route=read(HERE/'route-inspection-current-v1.json')['rows']
route_file_drift=sorted({r['target'] for r in old_route if sha(ROOT/r['target'])!=r['target_sha256']})
assert route_file_drift==['course/chapters/09.md','course/chapters/12.md','course/chapters/19.md']


def reconstruct(freeze,path):
    pages=[p for p in freeze['pages'] if p['source']==path]
    intro=next(p for p in pages if p['kind']=='chapter_introduction')
    parts=[(ROOT/intro['snapshot']).read_bytes()]
    for page in sorted([p for p in pages if p['kind']=='lesson'],key=lambda p:int(p['page_id'].split('.')[1])):
        parts.append((ROOT/page['snapshot']).read_bytes())
    return b''.join(parts)


parents=[]
for path in route_file_drift:
    before=reconstruct(old_canonical,path);after=reconstruct(new_canonical,path)
    prior_hash=next(r['target_sha256'] for r in old_route if r['target']==path)
    assert hashlib.sha256(before).hexdigest()==prior_hash
    assert after==(ROOT/path).read_bytes()
    parents.append({'path':path,'previous_sha256':prior_hash,'current_sha256':sha(ROOT/path),
        'entire_parent_reconstructed_from_canonical_slices':True,'only_changes':next(c['lesson_id'] for c in diff_rows if new_pages[c['lesson_id']]['source']==path)})
route_rows=[]
for row in old_route:
    path=ROOT/row['target'];text=path.read_text()
    item={'url':row['url'],'target':row['target'],'target_sha256':sha(path),
        'parent_unchanged':sha(path)==row['target_sha256'],'source_line':row['source_line']}
    if '#' in row['url']:
        lesson=row['url'].partition('#')[2];m=re.search(r'^## '+re.escape(lesson)+r' .+$',text,re.M)
        assert m and m[0]==row['target_heading']
        following=re.search(r'^## ',text[m.end():],re.M);end=m.end()+following.start() if following else len(text)
        section=text[m.start():end]
        item['section_sha256']=hashlib.sha256(section.encode()).hexdigest()
        item['section_changed']=item['section_sha256']!=row['section_sha256']
        if item['section_changed']:assert lesson in changed_pages
    else:
        assert text.splitlines()[0]==row['target_heading']
    route_rows.append(item)
write('route-inspection-current-v3.json',{'created_at':datetime.now(timezone.utc).isoformat(),
    'kind':'bounded route dependency update after source12 changes; not new full semantic review',
    'source_sha256':course['source_sha256'],'local_links':len(route_rows),'rows':route_rows,
    'changed_parent_files':parents,'changed_section_diffs':diff_rows,
    'manual_judgment':{
        '9.17':'Actual primary diff changes tie-breaking from stored probabilities to raw scores then stable ID. It matches the stated score-based choice and the already-inspected formal ChoiceModel.predict. Candidate scoring/reordering navigation remains accurate; no new quality claim.',
        '12.17':'Actual code already uses raw.square().mean() and divides by sqrt(mean-square); added sentence correctly relates mean-square to12.4 sum-square divided by sample count. No code/time/threshold or guide alignment claim changes.',
        '19.23':'Actual wording now permits question plus each level description to identify the rubric. Branch encoding already consumes question plus candidate description, so the formal example and independent state/candidate route remain consistent. No model, data, acceptance threshold or result changes.'},
    'scope':'Read the exact primary diffs and necessary local context; reconstructed whole changed parents by frozen slices to exclude hidden changes. Other routes reused after hash checks. No new lesson-level fresh verdict is claimed.'})

bundle_receipt_ref=checked(repairs['execution']);bundle_receipt=read(ROOT/bundle_receipt_ref['path'])
assert bundle_receipt['source_freeze']==new_side['canonical_source_freeze']
receipt_rows={r['id']:r for r in bundle_receipt['rows']}
prior_notebooks=read(HERE/'notebook-inspection-current-v1.json')['rows']
unchanged_count=0;changed_notebooks=[]
for prior_nb in prior_notebooks:
    lesson=prior_nb['lesson'];row=receipt_rows[lesson]
    checked(row['source'])
    if lesson not in changed_pages:
        assert row['source']['sha256']==prior_nb['source_sha256']
        assert row['joined']['sha256']==prior_nb['executed_sha256']
        unchanged_count+=1
        continue
    checked(row['joined'])
    current,joined=read(ROOT/row['source']['path']),read(ROOT/row['joined']['path'])
    assert len(current['cells'])==len(joined['cells'])
    assert all(a['cell_type']==b['cell_type'] and a['source']==b['source'] for a,b in zip(current['cells'],joined['cells']))
    codes=[c for c in joined['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in codes)
    assert not any(o['output_type']=='error' for c in codes for o in c.get('outputs',[]))
    changed_notebooks.append({'lesson_id':lesson,'source':row['source'],'joined':row['joined'],
        'cell_source_and_type_match':True,'code_cells':len(codes),'all_counts_present':True,'error_outputs':0,
        'author_new_kernel':row['new_source12_CPU_kernel'],'author_markdown_overlay':row['markdown_only_overlay_existing_identical_code_outputs']})
assert unchanged_count==328 and len(changed_notebooks)==3
write('notebook-inspection-current-v3.json',{'created_at':datetime.now(timezone.utc).isoformat(),
    'kind':'bounded static continuity of saved author outputs; reviewer ran no kernel',
    'author_v10_receipt':bundle_receipt_ref,'reused_prior_independent_inventory':ref('docs/technical-reviews/artifacts/omni-P5-course/notebook-inspection-current-v1.json'),
    'unchanged_source_and_prior_joined_digest_rows':unchanged_count,'changed_rows_static_checked':changed_notebooks,
    'total_current_sources_bound':331,'reviewer_new_kernels':0,'reviewer_neural_calls':0,
    'scope':'328 exact unchanged source/output digests reused; only three changed Notebook joins read. The author reran9.17;12.17/19.23 are prose-only overlays with unchanged code. No claim of331fresh executions.'})

old_entries={r['lesson_id']:r for r in old_side['sections']}
entry_changes=[r['lesson_id'] for r in new_side['sections'] if r!=old_entries[r['lesson_id']]]
assert entry_changes==['W.7']
all_refs=[];sections=[]
spec=importlib.util.spec_from_file_location('course_source12_validator',ROOT/'scripts/check_technical_reviews.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
reader_tasks={'/root/omni_p5_reader_entries','/root/omni_p5_reader_operations','/root/omni_p5_reader_decision',
    '/root/omni_p5_reader_multimodal','/root/omni_p5_reader_memory_research','/root/omni_p5_reader_auxiliary'}
for entry in new_side['sections']:
    lesson=entry['lesson_id'];old_page=next(p for p in old_numbered['pages'] if p['page_id']==lesson)
    page=next(p for p in new_numbered['pages'] if p['page_id']==lesson)
    assert {k:v for k,v in old_page.items() if k!='snapshot'}=={k:v for k,v in page.items() if k!='snapshot'}
    assert (ROOT/old_page['snapshot']).read_bytes()==(ROOT/page['snapshot']).read_bytes()
    all_refs.append({'lesson_id':lesson,'original':checked(entry['original_report']),'current':checked(entry['report']),
        'original_freeze':checked(entry['original_section_freeze']),'numbered_source_sha256':page['source_sha256']})
    if not lesson.startswith('R.'):continue
    assert entry==old_entries[lesson]
    raw=(ROOT/page['snapshot']).read_bytes();report=read(ROOT/entry['report']['path'])
    errors=validator._validate(ROOT,ROOT/'course/README.md',lesson,raw.decode(),report,{report['reviewer_task']:[lesson]},reader_tasks)
    assert not errors,(lesson,errors)
    sections.append({'lesson_id':lesson,'source_sha256':page['source_sha256'],'report':entry['report'],
        'original_report':entry['original_report'],'same_entry_as_sidecar_v2':True,'current_validate_errors':errors})
w7=next(r for r in new_side['sections'] if r['lesson_id']=='W.7')
old_w7=read(ROOT/old_entries['W.7']['report']['path']);new_w7=read(ROOT/w7['report']['path'])
assert old_w7['reviewer_task']==new_w7['reviewer_task']=='/root/omni_p5_technical_w_7'
assert old_w7['source_sha256']==new_w7['source_sha256']
w7_changes=[s for s in new_w7['sources'] if s['kind']=='repository_code' and any(t.get('path')==s['path'] and t.get('sha256')!=s['sha256'] for t in old_w7['sources'])]
assert {s['path'] for s in w7_changes}==set(route_file_drift)
for source in w7_changes:checked(source)
continuity_ref=ref('docs/course-revision-20261029-omni/reviews/P5-actual/coherence-current-collection-intake-v3.json')
continuity=read(ROOT/continuity_ref['path'])
matches=[r for r in continuity['collection_index'] if r['path']==w7['report']['path'] and r['sha256']==w7['report']['sha256'] and r['received_at']==w7['current_collection']['received_at']]
assert len(matches)==1 and matches[0]['collection_role']=='continuity'
batch1_ref=ref('docs/course-revision-20261029-omni/reviews/P5-actual/source12-callback-dispatch-batch1-v1.json')
assert read(ROOT/batch1_ref['path'])['dispatch_window']==w7['current_dispatch_window']
phase_ref=ref('docs/course-revision-20261029-omni/reviews/P5-actual/coherence-stage-open-v1.json')
phase=read(ROOT/phase_ref['path']);technical_ref=checked(phase['preceding_technical_seal'])
technical=read(ROOT/technical_ref['path'])
assert technical['status']=='completed' and technical['role']=='technical'
assert 'open' in phase['status'] and phase['role']=='continuity'
assert technical['completed_at']<new_w7['created_at']<=matches[0]['received_at']
write('source12-bounded-current-v3.json',{'created_at':datetime.now(timezone.utc).isoformat(),
    'kind':'same-owner bounded course callback in continuity03; prior semantic work reused',
    'environment':{'python':platform.python_version(),'device':'CPU hashes/schema/static saved-output inspection only'},
    'prior_current_report':prior_ref,'canonical_source_freeze':new_canonical_ref,'numbered_section_freeze':new_numbered_ref,
    'shared_sidecar':new_side_ref,'issue_inventory':new_issue_ref,'author_diff_receipt':repairs_ref,
    'course_source_sha256':course['source_sha256'],'course_bytes':len(body),'course_body_unchanged':True,
    'registered_repository_sources_unchanged':registered_sources,'prior_artifacts_unchanged':prior_artifacts,
    'route_dependency_changes':parents,'exact_primary_diffs_checked':diff_rows,'route_proof':ref('docs/technical-reviews/artifacts/omni-P5-course/route-inspection-current-v3.json'),
    'notebook_proof':ref('docs/technical-reviews/artifacts/omni-P5-course/notebook-inspection-current-v3.json'),
    'numbered_all26_source_bytes_unchanged':True,'numbered_entry_changes':entry_changes,'all26_refs_checked':all_refs,
    'course_R_sections':sections,'issues_array_unchanged':True,
    'W7_metadata_only_check':{'old_report':old_entries['W.7']['report'],'new_report':w7['report'],'same_owner':True,
        'source_bytes_unchanged':True,'changed_registered_repo_sources':[{'path':s['path'],'sha256':s['sha256']} for s in w7_changes],
        'actual_current_collection':w7['current_collection'],'matching_existing_receipt':matches[0],
        'receipt_source':continuity_ref,'dispatch_source':batch1_ref,'current_dispatch_window':w7['current_dispatch_window'],
        'scope':'No W7 semantic rereview or new task verdict by this course collector.'},
    'technical_phase_immutable_completed':{'reference':technical_ref,'completed_at':technical['completed_at']},
    'current_receipt_phase':{'reference':phase_ref,'role':'continuity','status':phase['status']},
    'limits':['No fresh/full blind reread, third-party re-research, new kernel/NN/train/T/tau/Modal/workflow/publish.',
        'This callback is not a new receipt for course. Root must actually collect it in continuity03; technical03 seal was not modified.',
        'No new figures or visual layout validation. Prior limitations and all original/current1/current2 reports retained.']})
print(json.dumps({'proof':ref('docs/technical-reviews/artifacts/omni-P5-course/source12-bounded-current-v3.json'),
    'course_unchanged':True,'changed_route_parents':route_file_drift,'four_R_entries_unchanged':True,
    'section_validate_errors':{r['lesson_id']:r['current_validate_errors'] for r in sections},
    'all26_source_bytes_unchanged':True,'only_sidecar_entry_change':'W.7 same-owner current2 in continuity',
    'saved_notebook_sources_reused':328,'changed_notebook_joins_statically_checked':3,'new_reviewer_kernels':0,
    'actual_receipt_stage_for_this_callback':'continuity; root receipt pending'},ensure_ascii=False))
