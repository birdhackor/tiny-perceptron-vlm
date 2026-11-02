"""有界 source21 保留證據核對；source22 hold 期間不建立 current。"""

import hashlib
import json
import math
import platform
import re
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
COORD = ROOT / 'docs/course-revision-20261101-c-v3/technical-coordination-v21'
OLD = 'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-training/'
ZIP = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
PRODUCT_ROOT = Path('/workspace/work/c-product-v3-20261101')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def verify(ref):
    raw = (ROOT / ref['path']).read_bytes()
    assert sha(raw) == ref['sha256'], ref
    return json.loads(raw)


def refs(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for child in value.values():
            yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


result = {'created_at': datetime.now(UTC).isoformat(), 'actor': '/root/c_v3_technical_coord/canonical_training',
          'status': 'bounded source21 evidence only; final current held for genuine source22 final26 callback',
          'environment': {'python': platform.python_version(), 'device': 'CPU JSON/hash/arithmetic only', 'root': str(ROOT)},
          'limits': 'No new research, training, neural inference, final evaluator, publication, network or weight/ZIP copies. Stored labels/logits are inspected, not regenerated. Source21 bundle remains historical after source22 hold.'}
reference_map = read(str((COORD / 'prior-evidence-reference-map-resume-v1.json').relative_to(ROOT)))
training = next(row for row in reference_map['pages'] if row['page_id'] == 'training')
old_current = verify(training['prior_technical_proof']['report'])
old_original = verify(training['prior_technical_proof']['original_report'])
freeze18 = verify(training['prior_technical_proof']['original_source_freeze'])
page18 = next(row for row in freeze18['pages'] if row['page_id'] == 'training')
assert sha((ROOT / page18['snapshot']).read_bytes()) == page18['source_sha256'] == old_current['source_sha256']
result['old_training'] = {'original': training['prior_technical_proof']['original_report'], 'current': training['prior_technical_proof']['report'],
                          'owner': old_current['reviewer_task'], 'original_created_at': old_original['created_at'],
                          'current_created_at': old_current['created_at'], 'source18': page18,
                          'current_source_sha256': sha((ROOT / 'course/training.md').read_bytes()),
                          'unchanged_prefix_before_T11': (ROOT / page18['snapshot']).read_bytes().split(b'## T.11')[0] == (ROOT / 'course/training.md').read_bytes().split(b'## T.11')[0]}
result['retained_dependencies'] = []
for item in old_current['sources'] + old_current['artifacts']:
    if 'path' in item and 'sha256' in item:
        actual = sha((ROOT / item['path']).read_bytes())
        assert actual == item['sha256'], item['path']
        result['retained_dependencies'].append({'id': item['id'], 'kind': item['kind'], 'path': item['path'], 'sha256': actual, 'identical': True})
    elif item.get('outside_git_fulltext'):
        fulltext = Path(item['outside_git_fulltext'])
        actual = sha(fulltext.read_bytes())
        assert actual == item['original_sha256']
        result['retained_dependencies'].append({'id': item['id'], 'kind': item['kind'], 'url': item['url'], 'path_outside_git': str(fulltext),
                                                 'sha256': actual, 'original_owner': old_current['reviewer_task'], 'original_accessed_on': item['accessed_on'],
                                                 'locator': item['inspection_note'], 'identity_only_no_new_research': True})
old_raw = read('docs/technical-reviews/artifacts/omni-p6-wave04-training/original-data-inspection.json')
old_cpu = read('docs/technical-reviews/artifacts/omni-p6-wave04-training/execution.json')
result['historical_timings'] = {}
for name, historical in old_raw['historical_results'].items():
    raw = read('docs/course-experiments/results/' + name + '.json')
    assert raw['elapsed_seconds'] == historical['elapsed_seconds']
    assert round(raw['elapsed_seconds'], 1) == historical['quoted_seconds']
    result['historical_timings'][name] = {'elapsed_seconds': raw['elapsed_seconds'], 'quoted_seconds': historical['quoted_seconds'],
                                          'device': raw['device'], 'gpu': raw.get('gpu'), 'seed': raw['seed'],
                                          'timing_scope': raw.get('timing_scope', historical.get('timing_scope', 'not supplied in this stored result; no broader timing scope inferred')),
                                          'original_inspection_created_at': old_raw['created_at']}
result['retained_cpu_receipt'] = {'owner': old_cpu['reviewer_task'], 'created_at': old_cpu['created_at'], 'commands': len(old_cpu['commands']),
                                  'commands_all_exit0': all(row.get('returncode', row.get('exit_code')) == 0 for row in old_cpu['commands']),
                                  'scope': old_cpu['scope'], 'inline_code': old_cpu['inline_code']}
result['old_products'] = {key: old_raw[key] for key in ('selected_weight_bytes', 'selected_weight_total_bytes', 'selected_weight_total_mib',
                                                       'final_choice', 'final_gates', 'selection', 'provider_core_jobs')}

bundle_ref = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-final-v1.json',
              'sha256': '6b0849e632c43964359de7deb0f2abf30cb6224cea42de0003f6f8efbfd69f04'}
bundle = verify(bundle_ref)
section_freeze = verify(bundle['section_freeze'])
assert len(bundle['sections']) == 26
pages = {row['page_id']: row for row in section_freeze['pages']}
result['source21_numbered_bundle'] = {'reference': bundle_ref, 'status': 'historical source21 only; not current source22 closure',
                                     'canonical_freeze': bundle['canonical_source_freeze'], 'section_freeze': bundle['section_freeze'], 'rows': []}
for row in bundle['sections']:
    report, original = verify(row['report']), verify(row['original_report'])
    assert report['source_sha256'] == pages[row['lesson_id']]['source_sha256']
    assert sha((ROOT / pages[row['lesson_id']]['snapshot']).read_bytes()) == report['source_sha256']
    assert original['reviewer_task'] == report['reviewer_task']
    assert original['created_at'] <= row['original_collection']['received_at']
    assert report['created_at'] <= row['current_collection']['received_at']
    check = {'lesson_id': row['lesson_id'], 'owner': report['reviewer_task'], 'original': row['original_report'], 'current': row['report'],
             'original_created_at': original['created_at'], 'current_created_at': report['created_at'], 'original_received_at': row['original_collection']['received_at'],
             'current_received_at': row['current_collection']['received_at'], 'source_identity': True, 'dependencies': []}
    for item in report['sources'] + report['artifacts']:
        if item.get('path') and item.get('sha256'):
            p = ROOT / item['path']; actual = sha(p.read_bytes()) if p.is_file() else None
            check['dependencies'].append({'id': item['id'], 'path': item['path'], 'expected': item['sha256'], 'actual': actual, 'identical_live': actual == item['sha256']})
    if row['lesson_id'] in ('R.2', 'R.3', 'R.4', 'T.11', 'W.7'):
        label = {'R.2':'numbered-r2', 'R.3':'numbered-r3', 'R.4':'numbered-r4', 'T.11':'numbered-t11', 'W.7':'w7'}[row['lesson_id']]
        version = 'v2' if row['lesson_id'] in ('R.4', 'T.11') else 'v1'
        receipt_path = COORD / 'receipts' / (label + '-current-' + version + '.json')
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt['report'] == row['report'] and receipt['reviewer_task'] == report['reviewer_task']
        assert receipt['received_at'] == row['current_collection']['received_at']
        check['real_current_receipt'] = {'path': str(receipt_path.relative_to(ROOT)), 'sha256': sha(receipt_path.read_bytes()), 'received_at': receipt['received_at']}
    result['source21_numbered_bundle']['rows'].append(check)

public_path = 'docs/technical-reviews/artifacts/c-v3-v21-numbered-t11/public-delta-v2.json'
public = read(public_path)
assert public['temporary_directory_started_empty'] and public['bundle_started_absent'] and public['student_output_absent_immediately_before_download']
assert public['http']['status'] == 200 and public['http']['received_bytes'] == 6168350
assert not public['http']['authorization_header_sent_by_client'] and not public['http']['archive_reused_for_student_cli']
assert all(row['returncode'] == 0 for row in public['cli'])
judge = json.loads(public['cli'][1]['stdout'])
assert [judge['results'][kind]['selected'] for kind in ('choice','info','rubric')] == ['record','yes','2']
descriptor = read('docs/decision-product/v3/public-release.json')
assert {name: value['sha256'] for name,value in public['fresh_installed_files'].items()} == descriptor['bundle']
result['public_delta'] = {'path': public_path, 'sha256': sha((ROOT / public_path).read_bytes()), 'owner': public['reviewer_task'],
                          'started_at': public['started_at'], 'finished_at': public['finished_at'], 'empty_student_output': True,
                          'http_status': 200, 'bytes': public['http']['received_bytes'], 'archive_sha256': public['http']['sha256'],
                          'independent_archive_not_reused_by_cli': True, 'cli_returncodes':[row['returncode'] for row in public['cli']],
                          'selected': public['judge_selected'], 'four_file_hashes': descriptor['bundle'],
                          'historical_404_and_acquisition_false_preserved': True, 'no_new_network_by_collector': True}

assert sha(ZIP.read_bytes()) == descriptor['sha256']
with zipfile.ZipFile(ZIP) as archive:
    def original(name):
        return json.loads(archive.read('tiny-perceptron-c-v3/' + name))
    manifest = original('MANIFEST.json')
    assert all(sha(archive.read('tiny-perceptron-c-v3/' + name)) == entry['sha256'] for name,entry in manifest['files'].items())
    observations, examples, families = (original('evidence/' + name + '.json') for name in ('final-observations','final-examples','final-families'))
    final, opening, selection = (original('evidence/' + name + '.json') for name in ('final-results','final-opening','selection'))
    for name,key in (('final-observations','observations_sha256'),('final-examples','examples_sha256'),('final-families','families_sha256'),('final-opening','final_opening_sha256')):
        assert sha(archive.read('tiny-perceptron-c-v3/evidence/'+name+'.json')) == final[key]
    assert opening['selection_sha256'] == sha(archive.read('tiny-perceptron-c-v3/evidence/selection.json'))
    assert selection['created_at'] < opening['created_at'] < final['created_at']
    assert opening['weights_sha256'] == descriptor['bundle']['weights.safetensors']
    counts = Counter(); correct = Counter(); primary_families = defaultdict(set); accepted = errors = 0
    for observed, example in zip(observations, examples, strict=True):
        assert observed['family'] == example['family'] and observed['kind'] == example['kind']
        ids = [pair[0] for pair in example['candidates']] + ['none']
        assert ids == observed['ids'] and ids[observed['label']] == example['gold'] == observed['gold_id']
        logits = [-math.inf if value is None else value for value in observed['logits']]
        chosen = max(range(len(logits)), key=lambda index: logits[index])
        counts[observed['kind']] += 1; correct[observed['kind']] += chosen == observed['label']
        primary_families[observed['kind']].add(observed['family'])
        if observed['kind'] == 'choice':
            scaled = [value/selection['temperatures']['choice'] for value in logits]
            maximum = max(scaled); confidence = 1/sum(math.exp(value-maximum) for value in scaled)
            take = ids[chosen] != 'none' and observed['eligible'][chosen] and confidence >= selection['selected']['threshold']
            accepted += take; errors += take and chosen != observed['label']
    assert dict(counts) == dict(correct) == {'choice':1500,'info':1500,'rubric':1500}
    assert all(len(value) == 1500 for value in primary_families.values()) and len(families) == 1500
    assert accepted == 1025 and errors == 0
    manual, legacy = original('evidence/product-checks.json'), original('evidence/legacy-regression.json')
    manual_counts=Counter();manual_correct=Counter();local_states=set()
    for item in manual['observations']:
        kind=item['kind']; prediction=item['prediction']; expected=item['expected']
        manual_counts[kind]+=1;manual_correct[kind]+=(prediction['selected'] or 'none')==expected
        if kind in ('choice','info','rubric'):
            local_states.add((item['task'],item['permission'],item['ready']))
    legacy_counts=Counter();legacy_correct=Counter()
    for item in legacy['observations']:
        legacy_counts[item['kind']]+=1;legacy_correct[item['kind']]+=(item['prediction']['selected'] or 'none')==item['gold']
    assert len(local_states)==27 and manual_counts==manual_correct and legacy_counts==legacy_correct
    assert dict(legacy_counts)=={'choice':60,'info':60,'rubric':60}
    index=original('evidence/reviews/index.json');review_metadata=[];review_refs=[]
    for entry in index['reviews']:
        raw=archive.read('tiny-perceptron-c-v3/evidence/reviews/'+entry['path']);assert sha(raw)==entry['sha256'];report=json.loads(raw)
        assert report['reviewer_task']==entry['actor'] and report['implementation']==index['implementation']
        review_metadata.append({'kind':entry['kind'],'owner':entry['actor'],'created_at':report['created_at'],'report_sha256':sha(raw),
                                'scope_preserved':report.get('empirical_claim_limit',report.get('scope',report.get('report_stage')))})
        for item in refs(report):
            p=Path(item['path']);p=p if p.is_absolute() else PRODUCT_ROOT/p
            actual=sha(p.read_bytes()) if p.is_file() else None
            review_refs.append({'path':str(p),'expected':item['sha256'],'actual':actual,'identical':actual==item['sha256']})
    result['c3_original_stored_measurements']={'archive_sha256':descriptor['sha256'],'manifest_members_hashed':len(manifest['files']),
        'stored_primary_counts':dict(counts),'stored_primary_correct':dict(correct),'families_per_kind':{k:len(v) for k,v in primary_families.items()},
        'correlated_questions_not_4500_independent_families':True,'accepted':accepted,'accepted_errors':errors,'coverage':accepted/1500,
        'selection_before_opening_before_result':[selection['created_at'],opening['created_at'],final['created_at']],
        'manual_counts':dict(manual_counts),'manual_correct':dict(manual_correct),'manual_local_fact_states':len(local_states),
        'legacy_counts':dict(legacy_counts),'legacy_correct':dict(legacy_correct),'legacy_scope':legacy['scope'],
        'review_metadata':review_metadata,'review_references':review_refs,'review_reference_occurrences':len(review_refs),
        'review_unique_references':len({(x['path'],x['expected']) for x in review_refs}),
        'all_review_references_identical':all(x['identical'] for x in review_refs),
        'measurement_limit':'Original saved labels/logits only; no predictor run, no fresh final, no pooled-kinds CI; manual/old60 are engineering/regression, not independent fresh3-device testing.'}

result['completed_at']=datetime.now(UTC).isoformat()
out=HERE/'retention-source21-v1.json'
with out.open('x',encoding='utf-8') as target:json.dump(result,target,ensure_ascii=False,indent=2,allow_nan=False);target.write('\n')
print(json.dumps({'output':str(out.relative_to(ROOT)),'sha256':sha(out.read_bytes()),'status':result['status'],
                  'retained_dependencies':len(result['retained_dependencies']),'numbered_sections':26,
                  'historical_timings':len(result['historical_timings']),'public_delta_verified':True,
                  'c3_counts':result['c3_original_stored_measurements']['stored_primary_correct'],
                  'review_refs':[len(review_refs),result['c3_original_stored_measurements']['review_unique_references']]},ensure_ascii=False))
