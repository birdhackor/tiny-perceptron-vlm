"""Bounded same-owner metadata callback; no new semantic/fresh review or NN work."""

import hashlib
import importlib.util
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=HERE/'metadata-callback-current-v2.json'
if OUT.exists():raise SystemExit('Refusing to overwrite saved bounded callback')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked(ref):
    p=ROOT/ref['path'];assert p.is_file() and sha(p)==ref['sha256'],ref
    return {'path':ref['path'],'bytes':p.stat().st_size,'sha256':ref['sha256']}


def reference(path):
    p=ROOT/path
    return {'path':path,'sha256':sha(p),'bytes':p.stat().st_size}


old_report_ref=checked({'path':'docs/course-revision-20261029-omni/reviews/P5-actual/technical-course/technical-current-v1.json',
    'sha256':'67973255c5eb410532bf187c4c2d050163e0ae6b0645566ef7d222d3877d9199'})
original_ref=checked({'path':'docs/course-revision-20261029-omni/reviews/P5-actual/technical-course/technical-original-v1.json',
    'sha256':'2241e73ca4efa1dacdc2313efaaa79ee112ea6e15a627484814738cf2110d009'})
old_report=read(ROOT/old_report_ref['path'])
old_side_ref=checked(old_report['numbered_section_reviews'])
new_side_ref=checked({'path':'docs/technical-reviews/artifacts/omni-P5-numbered-v2/numbered-section-reviews-v1.json',
    'sha256':'a6b2189776362c7bc883c0ea26d01c2223b191b5989799dfa411fc48ba0dd87f'})
old_side,new_side=read(ROOT/old_side_ref['path']),read(ROOT/new_side_ref['path'])
old_canonical_ref=checked(old_side['canonical_source_freeze'])
new_canonical_ref=checked(new_side['canonical_source_freeze'])
assert new_canonical_ref['path']=='docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v11.json'
assert new_canonical_ref['sha256']=='d46a11ab7e27275f7e6908b8df8e40ed451a21ecc4fd6fedac02c4e01d0df7df'
old_section_ref=checked(old_side['section_freeze']);new_section_ref=checked(new_side['section_freeze'])
assert new_section_ref['sha256']=='1ce6261143dc8321cf9ec7bc9e298c678aab07eed5daaaf66cb8ca04dd16a68a'
old_issue_ref=checked(old_side['issue_inventory']);new_issue_ref=checked(new_side['issue_inventory'])
old_issues,new_issues=read(ROOT/old_issue_ref['path']),read(ROOT/new_issue_ref['path'])
assert old_side['sections']==new_side['sections']
assert old_issues['reports']==new_issues['reports'] and old_issues['issues']==new_issues['issues']
side_changed=[k for k in set(old_side)|set(new_side) if old_side.get(k)!=new_side.get(k)]
assert set(side_changed)=={'canonical_source_freeze','section_freeze','issue_inventory','created_at'}
old_canonical,new_canonical=read(ROOT/old_canonical_ref['path']),read(ROOT/new_canonical_ref['path'])
old_numbered,new_numbered=read(ROOT/old_section_ref['path']),read(ROOT/new_section_ref['path'])
old_numbered_pages={p['page_id']:p for p in old_numbered['pages']}
numbered_snapshot_rebinds=[]
for page in new_numbered['pages']:
    previous=old_numbered_pages[page['page_id']]
    changes={k for k in set(page)|set(previous) if page.get(k)!=previous.get(k)}
    assert changes<= {'snapshot'},(page['page_id'],changes)
    old_raw=(ROOT/previous['snapshot']).read_bytes()
    new_raw=(ROOT/page['snapshot']).read_bytes()
    assert old_raw==new_raw
    assert hashlib.sha256(old_raw).hexdigest()==previous['source_sha256']==page['source_sha256']
    if changes:
        numbered_snapshot_rebinds.append({'lesson_id':page['page_id'],'previous_snapshot':previous['snapshot'],
            'current_snapshot':page['snapshot'],'unchanged_source_sha256':page['source_sha256']})
old_pages={p['page_id']:p for p in old_canonical['pages']}
changed_pages=[]
for page in new_canonical['pages']:
    previous=old_pages[page['page_id']]
    if page!=previous:
        changed_pages.append({'page_id':page['page_id'],
            'changed_fields':[k for k in set(page)|set(previous) if page.get(k)!=previous.get(k)],
            'previous_source_sha256':previous['source_sha256'],'current_source_sha256':page['source_sha256']})
assert len(changed_pages)==1 and changed_pages[0]['page_id']=='curriculum'
course=next(p for p in new_canonical['pages'] if p['page_id']=='course')
body=(ROOT/course['source']).read_bytes()
assert body==(ROOT/course['snapshot']).read_bytes()
assert hashlib.sha256(body).hexdigest()==course['source_sha256']==old_report['source_sha256']

repo_sources=[]
for source in old_report['sources']:
    if source['kind']=='repository_code':
        item=checked(source);repo_sources.append({'source_id':source['id'],**item})
prior_artifacts=[{'id':a['id'],**checked(a)} for a in old_report['artifacts']]
route_proof=read(HERE/'route-inspection-current-v1.json')
route_dependencies=[]
for row in route_proof['rows']:
    item=checked({'path':row['target'],'sha256':row['target_sha256']})
    route_dependencies.append({'guide_url':row['url'],**item})
notebook_proof=read(HERE/'notebook-inspection-current-v1.json')
notebooks_by_id={p.stem:p for p in (ROOT/'notebooks').rglob('*.ipynb')}
notebook_sources=[]
for row in notebook_proof['rows']:
    path=notebooks_by_id[row['lesson']]
    assert sha(path)==row['source_sha256']
    notebook_sources.append({'lesson':row['lesson'],'path':path.relative_to(ROOT).as_posix(),'sha256':row['source_sha256']})

spec=importlib.util.spec_from_file_location('course_bounded_validator',ROOT/'scripts/check_technical_reviews.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
readers={'/root/omni_p5_reader_entries','/root/omni_p5_reader_operations','/root/omni_p5_reader_decision',
    '/root/omni_p5_reader_multimodal','/root/omni_p5_reader_memory_research','/root/omni_p5_reader_auxiliary'}
sections=[];all_refs=[]
for entry in new_side['sections']:
    all_refs.append({'lesson_id':entry['lesson_id'],
        'references':{k:checked(entry[k]) for k in ['report','original_report','original_section_freeze']}})
    if not entry['lesson_id'].startswith('R.'):
        continue
    lesson=entry['lesson_id']
    page=next(p for p in new_numbered['pages'] if p['page_id']==lesson)
    raw=(ROOT/page['snapshot']).read_bytes()
    text=body.decode();match=re.search(r'^## '+re.escape(lesson)+r' .+$',text,re.M)
    following=re.search(r'^## ',text[match.end():],re.M)
    end=match.end()+following.start() if following else len(text)
    assert raw==text[match.start():end].encode()
    report=read(ROOT/entry['report']['path'])
    assert hashlib.sha256(raw).hexdigest()==page['source_sha256']==report['source_sha256']
    assert report['figure_sha256']==page['figures_sha256']
    errors=validator._validate(ROOT,ROOT/'course/README.md',lesson,raw.decode(),report,
        {report['reviewer_task']:[lesson]},readers)
    assert not errors,(lesson,errors)
    sections.append({'lesson_id':lesson,'source_sha256':page['source_sha256'],
        'current_report':entry['report'],'original_report':entry['original_report'],
        'original_collection':entry['original_collection'],'current_collection':entry['current_collection'],
        'current_validate_errors':errors,'reuse':'Same report/artifact/source bytes already semantically read by this owner in current-v1; no new fresh review.'})
canonical_errors=validator._validate(ROOT,ROOT/'course/README.md','course',body.decode(),old_report,
    {old_report['reviewer_task']:['course']},readers)
assert not canonical_errors
open_path=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/technical-stage-open-v1.json'
open_state=read(open_path)
assert 'open' in open_state['status'] and open_state.get('completed_at') is None
value={'created_at':datetime.now(timezone.utc).isoformat(),
    'kind':'bounded same-owner metadata callback; prior semantic work reused',
    'environment':{'python':platform.python_version(),'device':'CPU file hashes and existing schema checks only'},
    'original_report':original_ref,'prior_current_report':old_report_ref,
    'previous_canonical_source_freeze':old_canonical_ref,'canonical_source_freeze':new_canonical_ref,
    'previous_numbered_section_freeze':old_section_ref,'numbered_section_freeze':new_section_ref,
    'previous_shared_sidecar':old_side_ref,'shared_sidecar':new_side_ref,
    'previous_issue_inventory':old_issue_ref,'issue_inventory':new_issue_ref,
    'changed_sidecar_top_fields':sorted(side_changed),'all26_section_entries_exactly_equal':True,
    'all26_numbered_source_slices_exactly_equal':True,'numbered_snapshot_path_rebinds':numbered_snapshot_rebinds,
    'issue_report_refs_and_issues_exactly_equal':True,
    'canonical_page_changes':changed_pages,'course_body_bytes':len(body),'course_source_sha256':sha(ROOT/course['source']),
    'registered_repository_sources_unchanged':repo_sources,'all_prior_current_artifacts_unchanged':prior_artifacts,
    'route_dependencies_unchanged':route_dependencies,'notebook_sources_unchanged':notebook_sources,
    'all26_reference_digests':all_refs,'course_sections':sections,'prior_current_canonical_validate_errors':canonical_errors,
    'technical_open_observation':{'source':reference(open_path.relative_to(ROOT).as_posix()),'status':open_state['status'],'completed_at':None},
    'semantic_reuse':'All75 claims, original official-source reads, saved-row computations, prior R.1–R.4 coverage/issue judgments and actual collection chronology reused from this same owner current-v1 after exact-byte checks.',
    'limits':['No curriculum semantic review; canonical inventory identifies it as sole changed page, outside course prerequisites.',
        'No new full/fresh reading, papers, kernels, numeric experiments, neural inference, training, T/tau, Modal, workflows or publication.',
        'No collection_index/received_at/completed seal fabricated or written; root must collect this new current report.']}
OUT.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proof':reference(OUT.relative_to(ROOT).as_posix()),'canonical_source_freeze':new_canonical_ref,
    'sidecar':new_side_ref,'issue_inventory':new_issue_ref,'sole_canonical_page_change':changed_pages,
    'course_unchanged':True,'route_dependencies':len(route_dependencies),'notebook_sources':len(notebook_sources),
    'course_section_current_validate_errors':{s['lesson_id']:s['current_validate_errors'] for s in sections},
    'all26_entries_unchanged':True,'other_course_dependency_drift':[]},ensure_ascii=False))
