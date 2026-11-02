"""Whole-glossary collector's read-only sidecar/source/receipt verification.

Uses existing _validate; does not invoke a final check-numbered, create a seal,
copy the shared bundle, run a model, or mutate any source/review input.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re

ROOT=Path(__file__).resolve().parents[4]
PROOF=Path(__file__).resolve().parent
TASK='/root/omni_p5_technical_glossary_whole'
SIDE='docs/technical-reviews/artifacts/omni-P5-numbered-v1/numbered-section-reviews-v1.json'
SIDE_SHA='d57aebb02fadc2b8b3a0f07e6dc59b1d831193dc92b4d06504b52dbda2acabd6'
PROVENANCE='docs/course-revision-20261029-omni/reviews/P5-actual/numbered-neutral-receipt-provenance-v1.json'
PROVENANCE_SHA='211e46f27219c436700e5743d0805a072786e2853e38e59b570656040c62c02d'
ORIGINAL='docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/technical-original-v1.json'
ORIGINAL_SHA='1e4367497e8dec241dfc0cf142cfe886a2ba2d1dac376ed623a7e6dcc7aece1a'


def ref(path):
    raw=(ROOT/path).read_bytes()
    return {'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read_ref(value):
    result=ref(value['path'])
    assert result['sha256']==value['sha256'],value
    return json.loads((ROOT/value['path']).read_bytes())


def timestamp(value):return datetime.fromisoformat(value)


def main():
    started=datetime.now(timezone.utc).isoformat()
    side=read_ref({'path':SIDE,'sha256':SIDE_SHA})
    neutral=read_ref({'path':PROVENANCE,'sha256':PROVENANCE_SHA})
    own=read_ref({'path':ORIGINAL,'sha256':ORIGINAL_SHA})
    canonical=read_ref(side['canonical_source_freeze'])
    current_sections=read_ref(side['section_freeze'])
    inventory=read_ref(side['issue_inventory'])
    assert side['issue_inventory']['sha256']=='53a170ec26339686cb110f593da0787d360f53af33d16ea35fb2421bba0cdb82'
    assert len(side['sections'])==26
    previous=read_ref(side['previous_bundle'])
    oldmap={r['lesson_id']:r for r in previous['sections']}
    unchanged=[r['lesson_id'] for r in side['sections'] if r==oldmap.get(r['lesson_id'])]
    changed=[r['lesson_id'] for r in side['sections'] if r!=oldmap.get(r['lesson_id'])]
    assert len(unchanged)==20 and len(changed)==6,(unchanged,changed)
    page=next(p for p in canonical['pages'] if p['page_id']=='glossary')
    body=(ROOT/page['snapshot']).read_bytes()
    assert hashlib.sha256(body).hexdigest()==page['source_sha256']==own['source_sha256']
    assert body==(ROOT/'course/glossary.md').read_bytes()
    assert page['figures_sha256']==own['figure_sha256']=={}
    own_freeze=read_ref(own['canonical_source_freeze'])
    old_whole=next(p for p in own_freeze['pages'] if p['page_id']=='glossary')
    assert (ROOT/old_whole['snapshot']).read_bytes()==body
    drift=[]
    for source in own['sources']:
        if source['kind']=='repository_code' and ref(source['path'])['sha256']!=source['sha256']:
            drift.append(source['path'])
    assert not drift,drift
    # The shared bundle is authenticated as one exact artifact; other sections'
    # verdict contents are not being independently collected as this reviewer's work.
    owners=defaultdict(list)
    all_references=[]
    for entry in side['sections']:
        for field in ['report','original_report','original_section_freeze']:
            read_ref(entry[field])
            all_references.append({'lesson_id':entry['lesson_id'],'field':field,**ref(entry[field]['path'])})
        header=json.loads((ROOT/entry['report']['path']).read_bytes())
        owners[header['reviewer_task']].append(entry['lesson_id'])
    assert all(len(v)==1 for v in owners.values())
    assert TASK not in owners
    assert ORIGINAL not in json.dumps(side)
    assert 'technical-glossary/technical-current-v1.json' not in json.dumps(side)
    assert neutral['current_wave']['completed_technical_seal'] is None
    opened=read_ref(neutral['current_wave']['technical_open'])
    assert opened['role']=='technical' and opened['status'].startswith('open')
    # Old G evidence is located in its actual completed old technical collection.
    oldwave=next(w for w in neutral['prior_completed_waves'] if w['id']=='omni-mainline-01')
    oldseal=read_ref(oldwave['seals']['technical'])
    assert oldseal['role']=='technical' and oldseal['status']=='completed'
    assignments=read_ref(neutral['prior_assignment_source'])
    reader_tasks={r['task'] for r in assignments.get('tasks',[]) if r.get('role')=='reader'}
    spec=importlib.util.spec_from_file_location('existing_technical_validator',ROOT/'scripts/check_technical_reviews.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    currentmap={p['page_id']:p for p in current_sections['pages']}
    coverage=json.loads((PROOF/'claim-coverage-v1.json').read_text())
    results=[]
    for entry in side['sections']:
        lesson=entry['lesson_id']
        if not lesson.startswith('G.'):continue
        assert entry==oldmap[lesson]
        assert entry['report']==entry['original_report']
        report=read_ref(entry['report'])
        original_section_freeze=read_ref(entry['original_section_freeze'])
        before=next(p for p in original_section_freeze['pages'] if p['page_id']==lesson)
        now=currentmap[lesson]
        before_raw=(ROOT/before['snapshot']).read_bytes()
        now_raw=(ROOT/now['snapshot']).read_bytes()
        assert before_raw==now_raw
        assert hashlib.sha256(now_raw).hexdigest()==now['source_sha256']==before['source_sha256']==report['source_sha256']
        match=re.search(rb'^## '+re.escape(lesson.encode())+rb' .*$',body,re.M)
        next_head=re.search(rb'^## ',body[match.end():],re.M)
        end=match.end()+next_head.start() if next_head else len(body)
        assert body[match.start():end]==now_raw
        assert now['figures_sha256']==before['figures_sha256']==report['figure_sha256']=={}
        actual_assignment=next(r for r in neutral['numbered_task_assignments'] if r.get('lesson_id')==lesson)
        assert actual_assignment['task']==report['reviewer_task']
        assert actual_assignment['context']=='fresh' and actual_assignment['wave']=='omni-mainline-01'
        window=actual_assignment['dispatch_window']
        assert timestamp(oldseal['started_at'])<=timestamp(window['not_before'])<=timestamp(window['not_after'])
        assert timestamp(window['not_after'])<=timestamp(report['created_at'])<=timestamp(oldseal['completed_at'])
        hits=[]
        for label in ['original_collection','current_collection']:
            collection=entry[label]
            assert collection['wave']=='omni-mainline-01' and collection['collection_role']=='technical'
            matching=[r for r in oldseal['collection_index'] if r.get('path')==entry['report']['path'] and r.get('sha256')==entry['report']['sha256'] and r.get('received_at')==collection['received_at']]
            assert len(matching)==1
            assert timestamp(report['created_at'])<=timestamp(collection['received_at'])<=timestamp(oldseal['completed_at'])
            hits.append({'collection_field':label,**matching[0]})
        assert entry['report']['sha256'] in oldseal['report_sha256s']
        assert entry['report'] in inventory['reports']
        issues=[i for i in inventory['issues'] if i.get('lesson_id')==lesson]
        assert not issues and not report['issues']
        errors=validator._validate(ROOT,ROOT/now['source'],lesson,now_raw.decode(),report,dict(owners),reader_tasks)
        assert not errors,(lesson,errors)
        original_errors=validator._validate(ROOT,ROOT/before['source'],lesson,before_raw.decode(),report,dict(owners),reader_tasks)
        assert not original_errors,(lesson,original_errors)
        rows=[r for r in coverage['table_rows'] if r['section']==lesson]
        own_ids=[c['id'] for c in own['claims'] if c['id'].startswith(lesson.lower().replace('.','')+'-')]
        assert all(r['claims'] for r in rows) and own_ids
        results.append({'lesson_id':lesson,'reviewer_task':report['reviewer_task'],'section_bytes':len(now_raw),
                        'source_sha256':now['source_sha256'],'current_original_same_report':True,
                        'report':ref(entry['report']['path']),'created_at':report['created_at'],
                        'original_section_freeze':entry['original_section_freeze'],'current_section_freeze':side['section_freeze'],
                        'dispatch':actual_assignment,'receipt_matches':hits,'retained_exact_entry':True,
                        'existing_validate_current_errors':errors,'existing_validate_original_errors':original_errors,
                        'section_claim_count':len(report['claims']),'source_count':len(report['sources']),'artifact_count':len(report['artifacts']),
                        'section_all_claims_verified':all(c['status']=='verified' for c in report['claims']),
                        'whole_original_claim_ids':own_ids,'whole_original_table_rows':len(rows),
                        'whole_original_all_rows_have_claims':True,'issues':issues,
                        'semantic_crosscheck':'Collector personally read all claim statements/scopes and primary-source inspection locators after saving its independent whole-page original; compared definitions, examples and qualifications to the same frozen page. No new substantive contradiction or omission found.'})
    assert [r['whole_original_table_rows'] for r in results]==[16,35,25,60]
    # Cross-match every index-term destination in the G.4 proof against exact body.
    g4=read_ref(next(e for e in side['sections'] if e['lesson_id']=='G.4')['report'])
    indexbody=body.decode().split('### 英文與縮寫索引',1)[1]
    expected={}
    for line in indexbody.splitlines():
        if not line.startswith('**'):continue
        pairs=re.findall(r'\[([^\]]+)\]\(([^)]+)\)',line)
        if line.startswith('**D**'):
            expected['decode']=[pairs[0][1],pairs[1][1]];pairs=pairs[2:]
        expected.update({term:[target] for term,target in pairs})
    observed={}
    for c in g4['claims']:
        m=re.fullmatch(r'英文索引「([^」]+)」連到本課定義位置：(.*)',c['statement'])
        if m:observed[m[1]]=m[2].split(', ')
    assert observed==expected and len(expected)==194
    output={'created_at':datetime.now(timezone.utc).isoformat(),'started_at':started,'reviewer_task':TASK,
            'command':'uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-glossary/crosscheck-numbered-v1.py',
            'environment':{'python':platform.python_version(),'device':'CPU; JSON/source/hash validation only'},
            'status':'passed','kind':'collector-local-partial-sidecar-crosscheck-not-final-check-numbered-or-seal',
            'shared_sidecar':ref(SIDE),'shared_issue_inventory':ref(side['issue_inventory']['path']),
            'neutral_provenance':ref(PROVENANCE),'canonical_freeze':side['canonical_source_freeze'],
            'section_freeze':side['section_freeze'],'immutable_whole_original':ref(ORIGINAL),
            'whole_glossary_bytes':len(body),'whole_glossary_sha256':page['source_sha256'],
            'whole_body_and_all_original_repository_sources_unchanged':True,'dependency_drift':drift,
            'bundle_reference_integrity':{'sections':26,'unchanged_prior_entries':unchanged,'six_replaced_entries':changed,
                                         'all_report_original_freeze_references_hashed':True,'reference_count':len(all_references),
                                         'whole_collector_back_reference_or_cycle':False,
                                         'scope':'Reference integrity and identity only for other22 sections; no semantic review of them is claimed.'},
            'old_technical_seal_reference':oldwave['seals']['technical'],'sections':results,
            'all_g1_g4_table_rows':sum(r['whole_original_table_rows'] for r in results),
            'old_g_claims_read':sum(r['section_claim_count'] for r in results),'index_terms_compared':194,
            'current_phase':{'wave':neutral['current_wave']['id'],'technical_status':opened['status'],'completed_technical_seal':None,
                             'final_check_numbered_run':False,'collector_collection_or_seal_created':False},
            'temporal_scope_note':'G.4 original c018/c080 describe the design and what still required evidence at its original2026-10-29 reading. Retention is not a new claim that2026-10-30 training remains uncompleted; the whole-page independent original separately inspected original runtime training records and still does not infer ability from CPU examples.',
            'limits':['No model inference, training, calibration/policy selection, Modal, workflow dispatch or publication.',
                      'No other section is independently reviewed under this task. Historical group6 was not delivered or assessed.',
                      'No shared bundle was copied or rewritten; exact common artifact path/SHA is bound by the current report.']}
    path=PROOF/'crosscheck-numbered-v1.json';assert not path.exists();path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'G_sections':len(results),'G_claims':output['old_g_claims_read']},ensure_ascii=False))


if __name__=='__main__':main()
