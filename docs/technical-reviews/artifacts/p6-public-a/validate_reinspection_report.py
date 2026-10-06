"""Validate current public-page review, frozen history and evidence references."""
from pathlib import Path
import hashlib, json, re

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
REPORT = ROOT / 'docs/technical-reviews/public-pages-p6-a.json'
INITIAL = BASE / 'history/initial-report.json'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
def unique(items):
    ids = [x['id'] for x in items]
    assert len(set(ids)) == len(ids)
    return set(ids)
r = json.loads(REPORT.read_bytes())
initial = json.loads(INITIAL.read_bytes())
assert sha(INITIAL) == '1cf5f7fef1587d4e9c20126d60f565ea3c5f1307884f1450a70ad212ac206f1e'
assert r['initial_review']['report_sha256'] == sha(INITIAL)
assert r['initial_review']['verdict'] == initial['verdict'] == 'revise'
assert r['initial_review']['source_files'] == initial['source_files']
assert r['initial_review']['issues'] == initial['issues']
assert sha(ROOT / r['initial_review']['initial_validation_path']) == r['initial_review']['initial_validation_sha256']
assert r['schema_name'] == 'public-pages-factual-review-v1'
assert r['reviewer_task'] == r['reinspection']['reviewer'] == '/root/p6_fact_public_a'
assert r['verdict'] == 'pass' and r['reinspection']['status'] == 'completed'
source_ids = unique(r['sources'])
artifact_ids = unique(r['artifacts'])
claims = [c for p in r['pages'] for c in p['claims']]
unique(claims)
assert len(r['pages']) == 7 and len(claims) == 72
assert all(c['status'] == 'verified' for c in claims)
for group in ['sources', 'artifacts']:
    for item in r[group]:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
for s in r['sources']:
    assert s['verified'] and s['checked_original']
    if s['kind'] in ['official_source', 'paper']:
        assert s['url'].startswith('https://')
changed = {'docs/asset-storage.md', 'assets/training/README.md'}
assert set(r['reinspection']['full_current_pages_read']) == changed
assert set(r['reinspection']['unchanged_sha_carry_forward_pages']) == set(initial['source_files']) - changed
assert r['reinspection']['dependency_count'] == 177
assert r['reinspection']['all_initial_dependency_hashes_match']
assert not r['reinspection']['historical_anonymous_readback_verified']
figures = {}
for p in r['pages']:
    path = ROOT / p['source']
    assert sha(path) == r['source_files'][p['source']] == p['source_sha256']
    assert sha(ROOT / p['frozen_input_path']) == p['frozen_input_sha256'] == p['source_sha256']
    assert p['verdict'] == 'pass'
    if p['source'] in changed:
        assert p['read_event']['kind'] == 'full_current_page_reinspection'
        assert p['source_sha256'] != initial['source_files'][p['source']]
        assert p['initial_source_sha256'] == initial['source_files'][p['source']]
        assert '8,950個其他檔案（包含圖片、錄音與來源說明）' in path.read_text()
    else:
        assert p['read_event']['kind'] == 'unchanged_sha_carry_forward'
        assert p['source_sha256'] == initial['source_files'][p['source']]
    direct = {}
    for rel in re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)', path.read_text()):
        f = (path.parent / rel).resolve()
        direct[str(f.relative_to(ROOT))] = sha(f)
    assert direct == p['figure_sha256']
    figures.update(direct)
    assert all(c['status'] in ['pass', 'not_applicable'] for c in p['checks'].values())
    for c in p['claims']:
        assert c['statement'] and c['location'] and c['scope'] and c['evidence']
        assert set(c['artifact_ids']) <= artifact_ids
        assert all(e['source_id'] in source_ids and e['locator'] and e['supports'] for e in c['evidence'])
assert figures == r['figure_sha256'] == initial['figure_sha256']
current_issues = {x['id']: x for x in r['issues']}
for old in initial['issues']:
    new = current_issues[old['id']]
    assert new['status'] == 'resolved'
    assert new['initial_status'] == old['status']
    assert new['details'] == old['details']
    assert new['initial_judgment_preserved'] and new['resolution']
    assert set(new['resolution_artifact_ids']) <= artifact_ids
assert not current_issues['i2']['historical_action_verified']
original_claims = {c['id']: c for p in initial['pages'] for c in p['claims']}
assert r['initial_review']['affected_claims'] == {k: original_claims[k] for k in ['s4', 'a8', 'a2']}
assert all(x['status'] == 'pass' for x in r['checks'].values())
assert not r['independence']['old_reader_or_technical_report_bodies_read']
assert not r['independence']['author_extra_result_notes_read']
assert all(r['execution_boundary'][k] is False for k in [
    'installed_packages', 'uploads', 'paid_compute', 'full_training', 'model_generation', 'new_holdout_answer_grading'])
result = {'status': 'passed', 'report': str(REPORT.relative_to(ROOT)), 'report_sha256': sha(REPORT),
    'initial_report': str(INITIAL.relative_to(ROOT)), 'initial_report_sha256': sha(INITIAL),
    'pages': len(r['pages']), 'claims': len(claims), 'sources': len(r['sources']),
    'artifacts': len(r['artifacts']), 'full_revised_pages_read': sorted(changed),
    'unchanged_pages_carried_forward': 5, 'all_source_artifact_hashes_match': True,
    'all_claim_references_resolve': True, 'initial_issue_judgments_preserved': True,
    'historical_anonymous_readback_newly_verified': False, 'verdict': r['verdict'],
    'command': '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/validate_reinspection_report.py'}
(BASE / 'reinspection-report-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
