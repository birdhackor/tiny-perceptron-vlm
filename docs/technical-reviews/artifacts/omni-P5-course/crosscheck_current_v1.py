"""Post-original course collector: source/sidecar/receipt checks, no neural work."""

import hashlib
import importlib.util
import json
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SIDE = 'docs/technical-reviews/artifacts/omni-P5-numbered-v1/numbered-section-reviews-v1.json'
SIDE_SHA = 'd57aebb02fadc2b8b3a0f07e6dc59b1d831193dc92b4d06504b52dbda2acabd6'
PROVENANCE = 'docs/course-revision-20261029-omni/reviews/P5-actual/numbered-neutral-receipt-provenance-v1.json'
PROVENANCE_SHA = '211e46f27219c436700e5743d0805a072786e2853e38e59b570656040c62c02d'
ORIGINAL = 'docs/course-revision-20261029-omni/reviews/P5-actual/technical-course/technical-original-v1.json'
ORIGINAL_SHA = '2241e73ca4efa1dacdc2313efaaa79ee112ea6e15a627484814738cf2110d009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked_ref(ref):
    p = ROOT / ref['path']
    assert p.is_file() and sha(p) == ref['sha256'], ref
    return {'path':ref['path'],'sha256':ref['sha256'],'bytes':p.stat().st_size}


def write(name, value):
    p = HERE/name
    if p.exists():
        raise RuntimeError('Refusing to overwrite current evidence: '+name)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def slice_section(text, lesson):
    match = re.search(r'^## '+re.escape(lesson)+r' .+$',text,re.M)
    assert match is not None
    following = re.search(r'^## ',text[match.end():],re.M)
    end = match.end()+following.start() if following else len(text)
    return text[match.start():end]


def main():
    now = datetime.now(timezone.utc).isoformat()
    original_ref = checked_ref({'path':ORIGINAL,'sha256':ORIGINAL_SHA})
    side_ref = checked_ref({'path':SIDE,'sha256':SIDE_SHA})
    provenance_ref = checked_ref({'path':PROVENANCE,'sha256':PROVENANCE_SHA})
    original, side, neutral = read(ROOT/ORIGINAL), read(ROOT/SIDE), read(ROOT/PROVENANCE)
    canonical_ref = checked_ref(side['canonical_source_freeze'])
    numbered_ref = checked_ref(side['section_freeze'])
    issue_ref = checked_ref(side['issue_inventory'])
    assert issue_ref['sha256']=='53a170ec26339686cb110f593da0787d360f53af33d16ea35fb2421bba0cdb82'
    canonical, numbered = read(ROOT/canonical_ref['path']), read(ROOT/numbered_ref['path'])
    page = next(r for r in canonical['pages'] if r['page_id']=='course')
    body_path, snapshot_path = ROOT/page['source'], ROOT/page['snapshot']
    body = body_path.read_text()
    assert body_path.read_bytes()==snapshot_path.read_bytes()
    assert sha(body_path)==page['source_sha256']==original['source_sha256']
    closure=[]
    for source in original['sources']:
        if source['kind']=='repository_code':
            path=ROOT/source['path']; observed=sha(path)
            closure.append({'source_id':source['id'],'path':source['path'],'expected':source['sha256'],'observed':observed,'unchanged':observed==source['sha256']})
    assert all(r['unchanged'] for r in closure)
    old_artifacts=[{'id':a['id'],**checked_ref(a)} for a in original['artifacts']]

    all_refs=[]
    for entry in side['sections']:
        all_refs.append({'lesson_id':entry['lesson_id'],
            'references':{k:checked_ref(entry[k]) for k in ['report','original_report','original_section_freeze']}})
    previous_ref=checked_ref(side['previous_bundle'])
    previous=read(ROOT/previous_ref['path'])
    old_entries={r['lesson_id']:r for r in previous['sections']}
    retained=[r['lesson_id'] for r in side['sections'] if r==old_entries.get(r['lesson_id'])]
    replaced=[r['lesson_id'] for r in side['sections'] if r!=old_entries.get(r['lesson_id'])]

    prior = next(w for w in neutral['prior_completed_waves'] if w['id']=='omni-mainline-01')
    old_receipt_ref=checked_ref(prior['seals']['technical'])
    current_receipt_ref=checked_ref(neutral['current_wave']['actual_collection_so_far'])
    open_ref=checked_ref(neutral['current_wave']['technical_open'])
    old_receipts=read(ROOT/old_receipt_ref['path'])['collection_index']
    current_receipts=read(ROOT/current_receipt_ref['path'])['collection_index']
    assert neutral['current_wave']['completed_technical_seal'] is None
    assert 'open' in read(ROOT/open_ref['path'])['status']
    owners=defaultdict(list)
    for assignment in neutral['numbered_task_assignments']:
        owners[assignment['task']].append(assignment['lesson_id'])
    spec=importlib.util.spec_from_file_location('course_sidecar_validator',ROOT/'scripts/check_technical_reviews.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    readers={'/root/omni_p5_reader_entries','/root/omni_p5_reader_operations','/root/omni_p5_reader_decision',
        '/root/omni_p5_reader_multimodal','/root/omni_p5_reader_memory_research','/root/omni_p5_reader_auxiliary'}
    sections=[]
    for entry in side['sections']:
        lesson=entry['lesson_id']
        if not lesson.startswith('R.'):
            continue
        current_page=next(r for r in numbered['pages'] if r['page_id']==lesson)
        live_slice=slice_section(body,lesson)
        assert live_slice.encode()==(ROOT/current_page['snapshot']).read_bytes()
        assert hashlib.sha256(live_slice.encode()).hexdigest()==current_page['source_sha256']
        reports={}; receipts={}; checks={}
        for field,collection_field in [('original_report','original_collection'),('report','current_collection')]:
            report=read(ROOT/entry[field]['path']);reports[field]=report
            freeze=read(ROOT/entry['original_section_freeze']['path']) if field=='original_report' else numbered
            target=next(r for r in freeze['pages'] if r['page_id']==lesson)
            raw=(ROOT/target['snapshot']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==target['source_sha256']==report['source_sha256']
            assert report['figure_sha256']==target['figures_sha256']
            errors=validator._validate(ROOT,ROOT/'course/README.md',lesson,raw.decode(),report,owners,readers)
            checks[field]={'verdict':report['verdict'],'errors':errors,'claim_count':len(report['claims']),
                'source_sha256':report['source_sha256'],'created_at':report['created_at'],'reviewer_task':report['reviewer_task']}
            if field=='report' or report['verdict']=='pass':
                assert not errors, (lesson,field,errors)
            else:
                assert lesson=='R.3' and report['verdict']=='revise'
            collection=entry[collection_field]
            pool=old_receipts if collection['wave']=='omni-mainline-01' else current_receipts
            matches=[r for r in pool if r['path']==entry[field]['path'] and r['sha256']==entry[field]['sha256'] and r['received_at']==collection['received_at']]
            assert len(matches)==1,(lesson,field,matches)
            receipt=matches[0]
            assert receipt['reviewer_task']==report['reviewer_task']
            assert report['created_at']<=receipt['received_at']
            receipts[field]={'reference':entry[field],'collection':collection,'unique_matching_receipt':receipt}
        assert reports['original_report']['reviewer_task']==reports['report']['reviewer_task']
        assert reports['original_report']['created_at']<=reports['report']['created_at']
        assignment=next(a for a in neutral['numbered_task_assignments'] if a['lesson_id']==lesson)
        assert assignment['task']==reports['report']['reviewer_task']
        assert assignment['dispatch_window']['not_before']<=reports['original_report']['created_at']
        if entry.get('current_dispatch_window'):
            assert entry['current_dispatch_window']['not_before']<=reports['report']['created_at']
        sections.append({'lesson_id':lesson,'current_exact_slice_sha256':current_page['source_sha256'],
            'original_section_freeze':entry['original_section_freeze'],'validation':checks,'receipts':receipts,
            'assignment':assignment,'current_dispatch_window':entry.get('current_dispatch_window'),
            'original_issues':reports['original_report']['issues'],
            'current_issues':reports['report']['issues'],
            'original_issue_resolutions':reports['report'].get('original_issue_resolutions',[])})

    route_rows=[]
    previous_routes=read(HERE/'route-inspection-original.json')['links']
    for old in previous_routes:
        path=ROOT/old['target'];text=path.read_text()
        row={'label':old['label'],'url':old['url'],'source_line':old['source_line'],'target':old['target'],
            'exists':path.is_file(),'target_sha256':sha(path),'previous_target_sha256':old['target_sha256']}
        if '#' in old['url']:
            section=slice_section(text,old['url'].partition('#')[2])
            row['target_heading']=section.splitlines()[0]
            row['section_sha256']=hashlib.sha256(section.encode()).hexdigest()
            row['opening_excerpt']=section[:1100]
        else:
            row['target_heading']=text.splitlines()[0];row['opening_excerpt']=text[:700]
        row['heading_same_as_original']=row['target_heading']==old['target_heading']
        row['opening_same_as_original']=row['opening_excerpt']==old['opening_excerpt']
        assert row['exists'] and row['heading_same_as_original']
        route_rows.append(row)
    write('route-inspection-current-v1.json',{'created_at':now,'source_sha256':sha(body_path),'local_links':len(route_rows),
        'missing_files_or_anchors':[],'rows':route_rows,'scope':'Guide target topic/anchor check only; no target-lesson review.'})

    bundle=ROOT/neutral['current_exact_CPU_bundle'];notebook_rows=[]
    previous_notebooks={r['lesson']:r for r in read(HERE/'notebook-inspection-original.json')['rows']}
    for executed in sorted(bundle.rglob('*.ipynb')):
        source=ROOT/'notebooks'/executed.relative_to(bundle)
        a,b=read(source),read(executed)
        same=len(a['cells'])==len(b['cells']) and all(x['cell_type']==y['cell_type'] and x['source']==y['source'] for x,y in zip(a['cells'],b['cells']))
        code=[c for c in b['cells'] if c['cell_type']=='code']
        complete=all(c.get('execution_count') is not None for c in code)
        errors=sum(o['output_type']=='error' for c in code for o in c.get('outputs',[]))
        assert same and complete and not errors
        notebook_rows.append({'lesson':executed.stem,'source_sha256':sha(source),'executed_sha256':sha(executed),
            'same_cell_sources':same,'code_cells':len(code),'all_execution_counts_present':complete,'error_outputs':errors,
            'source_changed_since_original':sha(source)!=previous_notebooks[executed.stem]['source_sha256']})
    assert len(notebook_rows)==331
    notebook_receipt=checked_ref(neutral['current_CPU_receipt'])
    write('notebook-inspection-current-v1.json',{'created_at':now,'source_bundle_location':str(bundle),
        'source_bundle_receipt':notebook_receipt,'total_notebooks':len(notebook_rows),
        'total_code_cells':sum(r['code_cells'] for r in notebook_rows),'all_same_sources':True,
        'all_executed_without_error':True,'rows':notebook_rows,
        'scope':'New read-only static inspection of saved current author outputs; no fresh kernel or Colab execution.'})

    chapter_sections=[]
    for path in sorted((ROOT/'course/chapters').glob('*.md')):
        chapter_sections.extend(re.findall(r'^## ([A-Z\d]+\.\d+) .+$',path.read_text(),re.M))
    index=(ROOT/'course/lessons.md').read_text()
    indexed=re.findall(r'\[閱讀\]\(\.\./course/chapters/[^)#]+#([A-Z\d]+\.\d+)\)',index)
    notebooks=re.findall(r'\[開啟\]\(([^)]+\.ipynb)\)',index)
    assert len(chapter_sections)==len(indexed)==len(notebooks)==331
    assert set(chapter_sections)==set(indexed)
    assert all((ROOT/'course'/n).resolve().is_file() for n in notebooks)
    r3=next(r for r in sections if r['lesson_id']=='R.3')
    assert len(r3['original_issues'])==1 and r3['original_issues'][0]['necessity']=='optional'
    assert r3['original_issue_resolutions'][0]['json_pointer']=='/issues/0'
    assert r3['original_issue_resolutions'][0]['source_quote'] in body
    original_schema=validator._validate(ROOT,ROOT/'course/README.md','course',body,original,
        {original['reviewer_task']:['course']},readers)
    assert not original_schema
    write('sidecar-crosscheck-current-v1.json',{'created_at':now,'environment':{'python':platform.python_version(),'device':'CPU read-only files and arithmetic'},
        'original_report':original_ref,'canonical_source_freeze':canonical_ref,'numbered_section_freeze':numbered_ref,
        'shared_sidecar':side_ref,'issue_inventory':issue_ref,'neutral_receipt_provenance':provenance_ref,
        'canonical_course_unchanged_from_original':True,'original_artifacts_unchanged':old_artifacts,
        'original_repository_sources':closure,'original_canonical_validate_errors':original_schema,
        'all_26_reference_digests_checked':all_refs,'previous_bundle':previous_ref,
        'exact_retained_entry_ids':retained,'replacement_entry_ids':replaced,
        'course_sections_read_and_validated':sections,
        'historical_R3_optional_issue':{'original_issue_pointer':'/issues/0','current_resolution_pointer':'/original_issue_resolutions/0',
            'current_source_quote':'各章小節與 Notebook 都列在[目錄](lessons.md)。',
            'chapter_section_count':len(chapter_sections),'indexed_chapter_sections':len(indexed),'indexed_notebooks':len(notebooks),
            'all_chapter_sections_present':True,'all_indexed_notebook_files_exist':True,
            'conclusion':'Resolved within exact current canonical course; auxiliary R/W/T/G are not promised by the narrowed phrase.'},
        'receipt_sources':{'prior_completed_technical':old_receipt_ref,'current_open_intake':current_receipt_ref,'current_open_phase':open_ref},
        'current_technical_phase_completed':False,
        'scope':'Own full-page original precedes sidecar access. Semantic/report crosscheck is only R.1–R.4. Other22 entries receive metadata/digest retention checks only. No new collection record, seal, all-phase check-numbered, neural execution, training or publication.'})
    print(json.dumps({'own_original_unchanged':True,'repository_source_drift':[],'course_sections':[r['lesson_id'] for r in sections],
        'section_current_validate_errors':{r['lesson_id']:r['validation']['report']['errors'] for r in sections},
        'section_original_validate_errors':{r['lesson_id']:r['validation']['original_report']['errors'] for r in sections},
        'retained_entries':retained,'replaced_entries':replaced,'route_opening_changes':[r['url'] for r in route_rows if not r['opening_same_as_original']],
        'notebook_source_changes':[r['lesson'] for r in notebook_rows if r['source_changed_since_original']],
        'all_current_notebooks_match':True,'current_phase_completed':False},ensure_ascii=False))


if __name__=='__main__':
    main()
