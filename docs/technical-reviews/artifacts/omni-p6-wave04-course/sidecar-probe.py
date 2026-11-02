"""Validate the supplied exact26 only after the collector original was received.

Reads reports and frozen bytes; never executes report artifact commands or model code.
"""
from pathlib import Path
from datetime import UTC, datetime
from collections import defaultdict
import hashlib
import importlib.util
import json
import platform

ROOT=Path('/workspace/work/omni-book-integrated-v1')
ART=ROOT/'docs/technical-reviews/artifacts/omni-p6-wave04-course'
BUNDLE={'path':'docs/technical-reviews/artifacts/omni-P6-numbered-v4/numbered-section-reviews-v1.json',
        'sha256':'ee54a1d1fe683e2a703661c1de797b226a3555eb591ff25c72cfa8d728209703'}
ORIGINAL={'path':'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-course/original-v1.json',
          'sha256':'0fbea4f58bb7d57a60ea56a84345c80940e79f8cadf4bd9f766632730fd17919'}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read_ref(ref):
    p=ROOT/ref['path'];raw=p.read_bytes();assert sha(raw)==ref['sha256'],ref['path'];return json.loads(raw)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

setup=module('course_sidecar_review_setup','scripts/omni_revision/review_setup.py')
validator=module('course_sidecar_technical','scripts/check_technical_reviews.py')
api=module('course_sidecar_incremental','docs/review-tools/incremental_revision.py')
reading=module('course_sidecar_reading','scripts/reading_time.py')
bundle=read_ref(BUNDLE);own=read_ref(ORIGINAL)
canonical=read_ref(bundle['canonical_source_freeze']);sf=read_ref(bundle['section_freeze'])
assert bundle['canonical_source_freeze']['path'].endswith('/source-freeze-v18.json')
assert bundle['section_freeze']['path'].endswith('/technical-numbered-source-freeze-v8.json')
assert sf['canonical_source_freeze']==bundle['canonical_source_freeze']
parents={p['page_id']:p for p in canonical['pages']}
sections={p['page_id']:p for p in sf['pages']}
expected={**{f'R.{i}':'course' for i in range(1,5)},**{f'W.{i}':'first-steps' for i in range(1,8)},
          **{f'T.{i}':'training' for i in range(1,12)},**{f'G.{i}':'glossary' for i in range(1,5)}}
assert set(expected)==set(sections)=={x['lesson_id'] for x in bundle['sections']}
assert len(bundle['sections'])==26
reports={e['lesson_id']:read_ref(e['report']) for e in bundle['sections']}
neutral=json.loads((ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-neutral-inputs-v1.json').read_bytes())
reader_tasks=set(neutral['reader_tasks_for_role_exclusion'])
owners=defaultdict(list)
for sid,r in reports.items(): owners[r['reviewer_task']].append(sid)
assert all(len(v)==1 for v in owners.values())
assert not set(owners).intersection(reader_tasks|set(neutral['authors']))
rows=[]
for e in bundle['sections']:
    sid=e['lesson_id'];pid=expected[sid];page=sections[sid];current=reports[sid];original=read_ref(e['original_report'])
    parent_body=setup.frozen_body(parents[pid]);body=setup.frozen_body(page)
    assert reading.lesson_slices(parent_body)[sid]==body
    assert page['canonical_page_id']==pid and page['source']==parents[pid]['source']
    assert original['reviewer_task']==current['reviewer_task']
    assert api.at(original['created_at'])<=api.at(current['created_at'])
    old_freeze=read_ref(e['original_section_freeze'])
    old_canonical=read_ref(old_freeze['canonical_source_freeze'])
    # Crucial: original goes through the formal historical contract, not current-tree _validate.
    setup.check_original(original,e['original_section_freeze'],sid,pid,current['reviewer_task'],validator,api,
                         original_source_snapshots=e.get('original_source_snapshots',[]))
    current_errors=validator._validate(ROOT,ROOT/page['source'],sid,body,current,owners,reader_tasks)
    assert current['figure_sha256']==page['figures_sha256']
    if current_errors: raise ValueError((sid,current_errors))
    original_issues=set(api.reported_issues(original));closed=current.get('original_issue_resolutions',[])
    assert original_issues=={x['json_pointer'] for x in closed} and len(original_issues)==len(closed)
    assert all(isinstance(x.get('resolution'),str) and x['resolution'].strip() for x in closed)
    earlier=setup.original_document_versions(old_canonical,original)
    historical_refs=[]
    for s in original['sources']:
        if s.get('kind')!='repository_code': continue
        p=ROOT/s['path'];live=sha(p.read_bytes()) if p.is_file() else None
        if live!=s['sha256']:
            historical_refs.append({'source':s['path'],'recorded_sha256':s['sha256'],'current_sha256':live,
                'reconstructed_from_original_canonical':s['sha256'] in earlier.get(s['path'],()),
                'explicit_snapshots':[x for x in e.get('original_source_snapshots',[]) if x['source']==s['path']]})
    rows.append({'lesson_id':sid,'canonical_page_id':pid,'reviewer_task':current['reviewer_task'],
        'original_report':e['original_report'],'current_report':e['report'],'original_section_freeze':e['original_section_freeze'],
        'original_canonical_freeze':old_freeze['canonical_source_freeze'],
        'original_verdict':original['verdict'],'current_verdict':current['verdict'],
        'original_check_original':'passed','current_validate_errors':current_errors,
        'current_source_sha256':current['source_sha256'],'current_body_is_canonical18_exact_slice':True,
        'original_created_at':original['created_at'],'current_created_at':current['created_at'],
        'original_issue_pointers':sorted(original_issues),'issue_closures':closed,
        'historical_repository_references':historical_refs,
        'current_claim_count':len(current['claims']),'source_count':len(current['sources']),
        'artifact_count':len(current['artifacts']),'current_unresolved_claims':[x['id'] for x in current['claims'] if x['status']!='verified']})

own_body=setup.frozen_body(parents['course'])
assert sha(own_body.encode())==own['source_sha256']
assert sha((ROOT/'course/README.md').read_bytes())==own['source_sha256']
assert all(x['canonical_page_id']=='course' for x in rows if x['lesson_id'].startswith('R.'))
result={'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_p6_technical_course',
    'environment':{'python':platform.python_version(),'device':'CPU, metadata/hash-only'},
    'scope':'Original check_original and current _validate for exact26 referenced reports; no report commands or models executed. Semantic compatibility assessed separately; this does not replace whole-course original or independent re-review all other guides.',
    'collector_original':ORIGINAL,
    'root_actual_receipt':{'received_at':'2026-10-30T17:49:47.575856+00:00','intake_version':'v20','sha256':'ab073bddac5d979d400007d123fcbaff05a7475a1e45de4c08d1ff1923da3f2b','provenance':'root followup message; not a claim of this reviewer performing root intake'},
    'bundle':BUNDLE,'canonical_freeze':bundle['canonical_source_freeze'],'numbered_freeze':bundle['section_freeze'],
    'canonical_page_counts':{'course':4,'first-steps':7,'training':11,'glossary':4},
    'reviewer_identity_check':'26 unique technical tasks; none match author/reader exclusions in the supplied neutral inputs. Actual dispatch attestations remain root responsibility.',
    'all_original_checks_passed':True,'all_current_checks_passed':True,'all_issue_pointers_closed':True,
    'collector_and_original_artifacts_untouched':all(sha((ROOT/x['path']).read_bytes())==x['sha256'] for x in own['artifacts']),
    'rows':rows}
out=ART/'sidecar-probe-result.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(out),'rows':len(rows),'all_original_checks_passed':True,'all_current_checks_passed':True,
                 'historical_reference_rows':sum(bool(x['historical_repository_references']) for x in rows),
                 'original_artifacts_untouched':result['collector_and_original_artifacts_untouched']},ensure_ascii=False))
