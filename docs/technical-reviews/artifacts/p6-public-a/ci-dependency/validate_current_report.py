"""Audit the final source-only callback and its preserved review history."""
from pathlib import Path
import hashlib, json

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
REPORT = ROOT / 'docs/technical-reviews/public-pages/p6-a.json'
PRE = BASE / 'history/pre-ci-dependency-callback-report.json'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
r, old = json.loads(REPORT.read_bytes()), json.loads(PRE.read_bytes())
assert sha(PRE) == 'a35e8963d474f2779ef5cff77feeed05256b21447b922f0bd60ada2c84d5136f'
assert r['source_files'] == old['source_files']
assert r['initial_review'] == old['initial_review'] and r['issues'] == old['issues']
assert r['reviewed_at'] == old['reviewed_at']
assert r['verdict'] == 'pass' and len(r['pages']) == 7
source_ids = [s['id'] for s in r['sources']]
artifact_ids = [a['id'] for a in r['artifacts']]
assert len(source_ids) == len(set(source_ids)) == 172
assert len(artifact_ids) == len(set(artifact_ids)) == 29
for group in ['sources', 'artifacts']:
    for item in r[group]:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
for s in r['sources']:
    assert s['verified'] and s['checked_original']
    if s['kind'] in ['official_source', 'paper']:
        assert s['url'].startswith('https://')
new_claims = {c['id']: c for p in r['pages'] for c in p['claims']}
old_claims = {c['id']: c for p in old['pages'] for c in p['claims']}
assert len(new_claims) == 73 and len(old_claims) == 72
for cid, c in old_claims.items():
    for key in ['kind', 'statement', 'location', 'status', 'scope']:
        assert new_claims[cid][key] == c[key], (cid, key)
for p in r['pages']:
    assert sha(ROOT / p['source']) == p['source_sha256'] == r['source_files'][p['source']]
    assert sha(ROOT / p['frozen_input_path']) == p['frozen_input_sha256'] == p['source_sha256']
    assert p['verdict'] == 'pass'
    for c in p['claims']:
        assert c['status'] == 'verified' and c['scope']
        assert set(c['artifact_ids']) <= set(artifact_ids)
        assert all(e['source_id'] in source_ids and e['locator'] and e['supports'] for e in c['evidence'])
for path, h in r['figure_sha256'].items():
    assert sha(ROOT / path) == h == old['figure_sha256'][path]
callback = r['source_dependency_callbacks'][-1]
assert callback['prior_report_sha256'] == sha(PRE)
assert callback['changed_source_ids'] == ['ci', 'pages', 'evaluate']
assert callback['other_unchanged_dependency_count'] == 184
assert callback['authored_public_pages_unchanged'] and callback['all_other_evidence_dependency_hashes_match']
assert not callback['native_windows_run_checked'] and not callback['native_windows_pass_claimed']
for sid in callback['changed_source_ids']:
    current = next(s for s in r['sources'] if s['id'] == sid)
    prior = next(s for s in old['sources'] if s['id'] == sid)
    assert callback['previous_source_versions'][sid] == prior
    assert callback['current_source_versions'][sid] == current
    assert current['source_history'][0]['sha256'] == prior['sha256']
    assert sha(ROOT / current['source_history'][0]['frozen_snapshot_path']) == prior['sha256']
    assert sha(ROOT / current['current_frozen_snapshot_path']) == current['sha256']
assert all(c['status'] == 'pass' for c in r['checks'].values())
assert all(r['execution_boundary'][k] is False for k in [
    'installed_packages', 'uploads', 'paid_compute', 'full_training', 'model_generation', 'new_holdout_answer_grading'])
result = {'status': 'passed', 'current_report': str(REPORT.relative_to(ROOT)), 'current_report_sha256': sha(REPORT),
    'pre_callback_report_sha256': sha(PRE), 'verdict': r['verdict'],
    'all_7_authored_pages_unchanged': True, 'original_72_claim_judgments_unchanged': True,
    'initial_issue_history_preserved': True, 'sources': len(source_ids), 'artifacts': len(artifact_ids),
    'all_current_source_artifact_hashes_match': True, 'all_references_resolve': True,
    'changed_source_ids': callback['changed_source_ids'], 'new_native_windows_pass_claimed': False,
    'command': '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/ci-dependency/validate_current_report.py'}
(BASE / 'ci-dependency/report-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
