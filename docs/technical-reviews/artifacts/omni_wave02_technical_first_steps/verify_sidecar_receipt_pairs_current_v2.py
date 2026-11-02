"""Bounded, post-original audit of the 26 sidecars and four collector pairs.

Calls the repository's existing validators directly. Does not run the global
gate, change reports/seals/checkers, execute models, or infer factual approval
from a metadata pass.
"""
import copy
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = 'docs/course-revision-20261029-omni/reviews/'
HERE = Path(__file__).resolve().parent
BUNDLE_REF = {'path':'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v5.json',
              'sha256':'0077db6122efdd4371d02334fa8aab5ab7002cd3312431b48fc57660bcb66d4f'}
ISSUES_REF = {'path':'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-issues-v5.json',
              'sha256':'0c434bc585e15b69404eab334864cfca143d8744a1078a09ef4ac30340579c82'}
SEAL_REF = {'path':BASE+'assembly-technical-intake-v4/technical-collection-seal.json',
            'sha256':'17d4f6885ecc54800e47df27f16e9d87c1a616004ca43ebec63f4175ecba7b1c'}
MAP_REF = {'path':BASE+'technical-stage/original-freeze-map-v16.json',
           'sha256':'ae3171412b89d0716a037f58d0dd9f45d9998debfbb5d4bbee8090b949b3ecad'}
WHOLE_REF = {'path':BASE+'technical-stage/whole-collectors-source15-index-v1.json',
             'sha256':'5504962eb971fa6218ba1cd4d0123ce50acff7937b7635548cd9d9b6e68c9979'}
DISPOSITION_REF = {'path':BASE+'wave-02/author-first-steps-index-optional-disposition-v1.json',
                   'sha256':'97dd306519f4c79fc6b7a7d79a70e5c8d0c1b2c72944a58a4bae3c13bf2a7130'}
OWN_ORIGINAL = {'path':BASE+'wave-02/omni_wave02_technical_first_steps/technical-original-v1.json',
                'sha256':'75b8eb0ad2005de671e5e1cd5c09e648db0901669d79683c940ff0c635c6b043'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


setup = load_module('own_bounded_setup','scripts/omni_revision/review_setup.py')
api = load_module('own_bounded_incremental','docs/review-tools/incremental_revision.py')
validator = load_module('own_bounded_technical','scripts/check_technical_reviews.py')
reading = load_module('own_bounded_reading','scripts/reading_time.py')
opened_refs = {}


def read(ref):
    raw = api.artifact(ROOT, ref)
    opened_refs[(ref['path'],ref['sha256'])] = copy.deepcopy(ref)
    return json.loads(raw)


started_at = datetime.now(timezone.utc).isoformat()
bundle, issue_index, seal = read(BUNDLE_REF), read(ISSUES_REF), read(SEAL_REF)
original_map, whole_index = read(MAP_REF), read(WHOLE_REF)
disposition, own_original = read(DISPOSITION_REF), read(OWN_ORIGINAL)
canonical = read(bundle['canonical_source_freeze'])
numbered = read(bundle['section_freeze'])
assignments = read(whole_index['assignments'])
numbered_snapshot = read(bundle['actual_collection_snapshot'])
whole_snapshot = read(whole_index['actual_collection_snapshot'])
sealed_snapshot = read(seal['actual_collection_snapshot'])
raw_ledger = read(seal['actual_raw_ledger'])
tasks = {row['task']:row for row in assignments['tasks']}
assert len(tasks) == len(assignments['tasks'])
reader_tasks = {task for task,row in tasks.items() if row['role']=='reader'}
wave = numbered_snapshot['wave']
sealed = {wave:{'technical':seal}}
assert seal['role']=='technical' and seal['status']=='completed'
assert api.at(seal['completed_at']) <= datetime.now(timezone.utc)
assert bundle['canonical_source_freeze']==whole_index['source_freeze']==seal['source_freeze']
assert numbered['canonical_source_freeze']==bundle['canonical_source_freeze']
assert whole_index['bundle']==BUNDLE_REF and whole_index['issues']==ISSUES_REF
assert bundle['issue_inventory']==ISSUES_REF
assert whole_index['numbered_freeze']==bundle['section_freeze']
assert numbered['kind']=='auxiliary_numbered_technical_freeze_not_canonical'
sealed_assignments=read(seal['assignments'])
# The completed packet copies the exact assignment bytes to its own path.
# Compare the bound bytes, rather than wrongly requiring storage paths to match.
assert seal['assignments']['sha256']==whole_index['assignments']['sha256']
assert sealed_assignments==assignments
canonical_pages = {p['page_id']:p for p in canonical['pages']}
numbered_pages = {p['page_id']:p for p in numbered['pages']}
entries = {e['lesson_id']:e for e in bundle['sections']}
assert len(entries)==len(bundle['sections'])==26
assert set(entries)==set(numbered_pages)
whole_rows = whole_index['whole_pages']
assert len(whole_rows)==4 and {r['page_id'] for r in whole_rows}==set(numbered['canonical_page_ids'])
original_map_entries = {e['lesson_id']:e for e in original_map['entries']}
reports = {}
owners = {}
for sid,entry in entries.items():
    reports[sid] = (read(entry['original_report']),read(entry['report']))
for row in whole_rows:
    reports[row['page_id']] = (read(row['original_report']),read(row['current_report']))
for identifier,(_,current) in reports.items():
    owners.setdefault(current['reviewer_task'],[]).append(identifier)
assert len(owners)==30 and all(len(ids)==1 for ids in owners.values())
rows = []
failures = []


def record_check(result, name, operation):
    try:
        value = operation()
        result['checks'][name]={'status':'pass','result':value}
    except Exception as error:
        detail=type(error).__name__+': '+str(error)
        result['checks'][name]={'status':'fail','error':detail}
        failures.append({'lesson_id':result['lesson_id'],'check':name,'error':detail})


def demand(condition, message):
    if not condition:
        raise ValueError(message)
    return True


def validate_current(page, report, identifier):
    errors = validator._validate(ROOT, ROOT/page['source'],identifier,setup.frozen_body(page),
                                 report,owners,reader_tasks)
    if errors:
        raise ValueError(repr(errors))
    demand(report['figure_sha256']==page['figures_sha256'],'exact figure list differs')
    return {'errors':errors,'source_sha256':report['source_sha256'],
            'figure_count':len(report['figure_sha256'])}


def map_check(identifier, entry, original, task, parent, original_freeze):
    mapped=original_map_entries[identifier]
    demand(mapped['original_report']==entry['original_report'],'original report map mismatch')
    demand(mapped['reviewer_task']==task,'original map owner mismatch')
    demand(mapped['actual_original_collection']==entry['original_collection'],'original receipt map mismatch')
    demand(mapped['original_source_freeze']==original_freeze,'original freeze map mismatch')
    if 'canonical_parent' in mapped:
        demand(mapped['canonical_parent']==parent,'canonical parent map mismatch')
        parent_method='explicit original map field'
    else:
        original_freeze_data=read(original_freeze)
        original_page=next(p for p in original_freeze_data['pages'] if p['page_id']==identifier)
        demand(original_page.get('canonical_page_id',identifier)==parent,'original freeze canonical parent mismatch')
        parent_method='derived from original freeze page; optional map field absent'
    demand(api.at(mapped['created_at'])==api.at(original['created_at']),'original map created_at mismatch')
    return {'original_freeze':original_freeze,'canonical_parent':parent,'same_owner':task,
            'canonical_parent_mapping_method':parent_method}


def receipt_rows(ref, receipt, report, subset):
    found=[]
    for label,ledger in [('true_completed_seal',seal),('sealed_actual_snapshot',sealed_snapshot),
                          ('raw_actual_ledger',raw_ledger),('scope_snapshot',subset)]:
        matched=[r for r in ledger['collection_index'] if r.get('path')==ref['path'] and r.get('sha256')==ref['sha256']]
        demand(len(matched)==1,label+' lacks unique exact report receipt')
        row=matched[0]
        demand(row['received_at']==receipt['received_at'],label+' receipt timestamp differs')
        demand(row['reviewer_task']==report['reviewer_task'],label+' receipt owner differs')
        demand(row['lesson_id']==report['lesson_id'],label+' receipt lesson differs')
        demand(api.at(row['created_at'])==api.at(report['created_at']),label+' created_at differs')
        demand(ref['sha256'] in ledger['report_sha256s'],label+' missing report hash membership')
        found.append({'ledger':label,'received_at':row['received_at'],'created_at':row['created_at'],
                      'kind':row['kind'],'unique_matches':len(matched)})
    return found


def closure_check(original,current):
    pointers=set(api.reported_issues(original))
    closures=current.get('original_issue_resolutions',[])
    demand(len(closures)==len(pointers),'original issue closure cardinality differs')
    demand({c.get('json_pointer') for c in closures}==pointers,'original issue pointer closure set differs')
    demand(all(isinstance(c.get('resolution'),str) and c['resolution'].strip() for c in closures),
           'original issue resolution lacks explanation')
    return [{'json_pointer':pointer,'original_issue':api.node(original,pointer),
             'current_resolution':next(c for c in closures if c['json_pointer']==pointer)}
            for pointer in sorted(pointers)]


def source_slice_check(page, parent):
    body=setup.frozen_body(page)
    parent_page=canonical_pages[parent]
    expected=reading.lesson_slices(setup.frozen_body(parent_page))[page['page_id']]
    demand(body==expected,'numbered snapshot is not exact canonical slice')
    demand(page['source']==parent_page['source'],'numbered source path differs')
    demand(page['selector']=='section:'+page['page_id'],'numbered selector differs')
    expected_figures=reading.figure_hashes(ROOT,ROOT/page['source'],body)
    demand(expected_figures==page['figures_sha256'],'numbered figure coverage differs')
    return {'canonical_parent':parent,'exact_slice':True,'figure_count':len(expected_figures)}


for sid,entry in entries.items():
    original,current=reports[sid]
    page=numbered_pages[sid]
    parent=page['canonical_page_id']
    task=current['reviewer_task']
    assigned=tasks[task]
    result={'lesson_id':sid,'canonical_parent':parent,'kind':'numbered_section_pair',
            'original_ref':entry['original_report'],'current_ref':entry['report'],
            'original_verdict':original['verdict'],'current_verdict':current['verdict'],
            'reviewer_task':task,'checks':{}}
    record_check(result,'fresh_task_assignment',lambda: demand(
        assigned['role']=='technical' and assigned['context']=='fresh' and assigned['lesson_id']==sid
        and original['reviewer_task']==task and task not in reader_tasks,'assignment/owner mismatch'))
    record_check(result,'exact_canonical_slice',lambda:source_slice_check(page,parent))
    record_check(result,'original_freeze_map',lambda:map_check(sid,entry,original,task,parent,entry['original_section_freeze']))
    record_check(result,'check_original',lambda:setup.check_original(
        original,entry['original_section_freeze'],sid,parent,task,validator,api,
        original_source_snapshots=entry.get('original_source_snapshots',[])))
    record_check(result,'current_native_validate',lambda:validate_current(page,current,sid))
    record_check(result,'original_issue_identities_and_closures',lambda:closure_check(original,current))
    record_check(result,'check_report_pair',lambda:setup.check_report_pair(api,sealed,entry,original,current,assigned))
    record_check(result,'original_true_receipt_rows',lambda:receipt_rows(entry['original_report'],entry['original_collection'],original,numbered_snapshot))
    record_check(result,'current_true_receipt_rows',lambda:receipt_rows(entry['report'],entry['current_collection'],current,numbered_snapshot))
    rows.append(result)

for row in whole_rows:
    pid=row['page_id']
    original,current=reports[pid]
    task=row['reviewer_task']
    assigned=tasks[task]
    callbacks=[r for r in assignments['callback_dispatches'] if r.get('task')==task
               and r.get('section_bundle')==BUNDLE_REF]
    demand(len(callbacks)==1,'whole collector lacks unique actual bundle5 callback')
    entry={**row,'report':row['current_report'],'current_dispatch_window':callbacks[0]['dispatch_window']}
    page=canonical_pages[pid]
    result={'lesson_id':pid,'canonical_parent':pid,'kind':'whole_collector_pair',
            'original_ref':row['original_report'],'current_ref':row['current_report'],
            'original_verdict':original['verdict'],'current_verdict':current['verdict'],
            'reviewer_task':task,'actual_callback':callbacks[0],'checks':{}}
    record_check(result,'fresh_task_assignment',lambda:demand(
        assigned['role']=='technical' and assigned['context']=='fresh' and assigned['lesson_id']==pid
        and original['reviewer_task']==current['reviewer_task']==task and task not in reader_tasks,
        'whole assignment/owner mismatch'))
    record_check(result,'original_freeze_map',lambda:map_check(pid,entry,original,task,pid,row['original_source_freeze']))
    record_check(result,'check_original',lambda:setup.check_original(
        original,row['original_source_freeze'],pid,pid,task,validator,api,
        original_source_snapshots=row.get('original_source_snapshots',[])))
    record_check(result,'current_native_validate',lambda:validate_current(page,current,pid))
    record_check(result,'original_issue_identities_and_closures',lambda:closure_check(original,current))
    record_check(result,'check_report_pair',lambda:setup.check_report_pair(api,sealed,entry,original,current,assigned))
    record_check(result,'bundle5_source_snapshot_artifact',lambda:demand(any(
        artifact.get('path')==BUNDLE_REF['path'] and artifact.get('sha256')==BUNDLE_REF['sha256']
        and artifact.get('kind')=='source_snapshot' for artifact in current.get('artifacts',[])),
        'whole current does not bind bundle5 as source_snapshot'))
    record_check(result,'original_true_receipt_rows',lambda:receipt_rows(row['original_report'],row['original_collection'],original,whole_snapshot))
    record_check(result,'current_true_receipt_rows',lambda:receipt_rows(row['current_report'],row['current_collection'],current,whole_snapshot))
    record_check(result,'current_self_check_reference',lambda:{'path':row['current_self_check']['path'],
        'sha256':hashlib.sha256(api.artifact(ROOT,row['current_self_check'])).hexdigest()})
    rows.append(result)

global_checks={}
scratch={'lesson_id':'__bounded_bundle__','checks':global_checks}
record_check(scratch,'issue_index_recomputed_from_actual_reports',lambda:demand(
    setup.section_issue_index(bundle,issue_index['author_task'],api)==issue_index,
    'bundle issue index differs from original/current real pointers'))
preserved=[]
for key in ['preserved_intermediate_issue_inventories','preserved_current_discovery_reports']:
    for ref in bundle.get(key,[]):
        report=read(ref)
        preserved.append({'category':key,'ref':ref,'hash_checked':True,
                          'issue_pointers':api.reported_issues(report)})
own_claim=next(c for c in own_original['claims'] if c['id']=='FS-C054')
own_issue=next(i for i in own_original['issues'] if i['id']=='FS-T-OPT-001')
decision=disposition['items'][0]
record_check(scratch,'own_original_and_unfixed_issue_preserved',lambda:demand(
    own_original['verdict']=='revise' and own_claim['status']=='contradicted'
    and own_issue['status']=='unresolved' and decision['issue_id']==own_issue['id']
    and decision['action']=='defer' and decision['implemented'] is False
    and decision['report']==OWN_ORIGINAL,'own original issue/disposition was misrepresented'))
record_check(scratch,'own_source15_body_unchanged',lambda:demand(
    hashlib.sha256((ROOT/'course/first-steps.md').read_bytes()).hexdigest()==own_original['source_sha256'],
    'own canonical body changed'))

helper_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
    'scripts/omni_revision/review_setup.py','scripts/check_technical_reviews.py',
    'docs/review-tools/incremental_revision.py','scripts/reading_time.py']}
counts={}
for page in numbered['canonical_page_ids']:
    counts[page]=sum(1 for r in rows if r['kind']=='numbered_section_pair' and r['canonical_parent']==page)
receipt={
    'schema_version':1,'kind':'own_bounded_post_original_sidecar_and_true_receipt_pair_check',
    'reviewer_task':'/root/omni_wave02_technical_first_steps',
    'started_at':started_at,'completed_at':datetime.now(timezone.utc).isoformat(),
    'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni_wave02_technical_first_steps/verify_sidecar_receipt_pairs_current_v2.py',
    'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu','executable':sys.executable},
    'original_was_received_before_this_informed_task_at':'2026-10-29T20:31:07.847518+00:00',
    'bound_inputs':{'bundle':BUNDLE_REF,'issues':ISSUES_REF,'true_completed_technical01_seal':SEAL_REF,
                    'original_freeze_map':MAP_REF,'four_whole_index':WHOLE_REF,
                    'own_original':OWN_ORIGINAL,'own_issue_disposition':DISPOSITION_REF},
    'native_helpers_used':helper_hashes,
    'assignment_storage_identity':{'seal':seal['assignments'],'earlier_index':whole_index['assignments'],
                                   'same_sha256_and_bytes':True},
    'reviewer_probe_history':{'path':(HERE/'sidecar-probe-attempt01-current-v1.json').relative_to(ROOT).as_posix(),
                              'sha256':hashlib.sha256((HERE/'sidecar-probe-attempt01-current-v1.json').read_bytes()).hexdigest(),
                              'explanation':'Initial probe stopped before pair checks because it required the assignment reference paths to match. The completed packet uses a copied path with identical SHA/bytes. The corrected probe verifies bytes and the actual selected assignments, without changing any input record.'},
    'preserved_first_full_probe':{'path':(HERE/'sidecar-receipt-pairs-check-current-v1.json').relative_to(ROOT).as_posix(),
                                  'sha256':hashlib.sha256((HERE/'sidecar-receipt-pairs-check-current-v1.json').read_bytes()).hexdigest(),
                                  'script_path':(HERE/'verify_sidecar_receipt_pairs_current_v1.py').relative_to(ROOT).as_posix(),
                                  'script_sha256':hashlib.sha256((HERE/'verify_sidecar_receipt_pairs_current_v1.py').read_bytes()).hexdigest(),
                                  'interpretation':'All30 native original/current/receipt-pair and issue-pointer checks passed. Twelve extra map lookups raised KeyError only because this reviewer assumed canonical_parent was mandatory in every index row. The format does not require that auxiliary key. This v2 explicitly verifies the actual parent in each bound original freeze when the map omits it; no input or native validator changed.'},
    'bounded_counts':{'numbered_pairs':26,'whole_pairs':4,'by_canonical_page':counts,
                       'declared_tasks_unique_in_selected_scope':len(owners),'issue_index_pointers':len(issue_index['issues'])},
    'rows':rows,'bundle_checks':global_checks,'preserved_intermediate_reference_checks':preserved,
    'result':{'passed_bounded_data_checks':not failures,'failure_count':len(failures),'failures':failures},
    'own_issue_state':{'issue_id':'FS-T-OPT-001','claim_id':'FS-C054','claim_status':'contradicted',
                       'issue_status':'unresolved','necessity':'optional','action':'defer','implemented':False,
                       'whole_current_verdict_remains':'revise'},
    'scope_and_limits':[
        'This is an independently executed metadata/source/receipt check after the fresh whole-page original was actually received. It is not another fresh reading of all 26 lessons.',
        'Existing _validate, check_original, check_report_pair and section_issue_index ran against the actual supplied bytes. Original revise states and stable issue JSON pointers remain visible.',
        'Only the 26 numbered and four supplied whole collector pairs were checked in the true completed technical01 seal and corresponding actual ledgers. Other seal rows and all global gates were not audited.',
        'Declared fresh-task/role/uniqueness checks are not proof of actual agent isolation or human student testing.',
        'Prior whole collector pass metadata does not override this later independent first-steps FS-C054 finding. The source fix is deferred, unimplemented and publication still needs the actual fix/recheck.',
        'No source/checker/other reviewer record/seal changed. No model training, inference, GPU, paid job, global tool approval, overall continuity approval or publication was performed.'
    ]
}
target=HERE/'sidecar-receipt-pairs-check-current-v2.json'
with target.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':target.relative_to(ROOT).as_posix(),
                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'result':receipt['result'],'counts':receipt['bounded_counts']},ensure_ascii=False))
raise SystemExit(1 if failures else 0)
