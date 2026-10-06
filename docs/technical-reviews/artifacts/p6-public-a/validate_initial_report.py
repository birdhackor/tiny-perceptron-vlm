"""Validate this custom public-page schema and frozen evidence references."""
from pathlib import Path
import collections, hashlib, json, re, struct

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
REPORT = ROOT / 'docs/technical-reviews/public-pages-p6-a.json'
EXPECTED = {
    'README.md': '1139d66e6733793e144c060c30e3deaf272ba59b9bd278e8a8ea0ee5f0b31c2c',
    'docs/environment.md': '5737f863ff760c217b00fad240a49b368225bbe3903f4cab32e177f816dc3c02',
    'docs/asset-storage.md': 'ccbdc60e9dd848480a1e9adedb577b1735fe11703c1b9941cc822ed0187d599a',
    'assets/training/README.md': '33079748eaf5a5c1da96dee9bd48317d6b8f9654a84a8b6ebd269e0f829c7d6d',
    'docs/publishing.md': 'e273ffd5c58d4af4a5e105814863f8c53bca8c41ae781a96cdc869f55f5a7024',
    'docs/validation.md': '373cb54bfc4e9d974bdefe89d775e47cb6208af1c0a9931c30ee1a6f68d1388b',
    'docs/curriculum.md': 'ce2519f211b5909592ea2f43b7fe21e1eda0d4ad6c0736d6369648d08478713e',
}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique_ids(items):
    ids = [x['id'] for x in items]
    assert len(ids) == len(set(ids)), ids
    return set(ids)
r = json.loads(REPORT.read_bytes())
assert r['schema_name'] == 'public-pages-factual-review-v1'
assert r['reviewer_task'] == '/root/p6_fact_public_a'
assert r['reviewer_context'] == 'fresh'
assert r['source_files'] == EXPECTED
assert len(r['pages']) == 7 and r['verdict'] == 'revise'
source_ids = unique_ids(r['sources'])
artifact_ids = unique_ids(r['artifacts'])
claims = [c for p in r['pages'] for c in p['claims']]
claim_ids = unique_ids(claims)
assert len(claims) == 72
for item in r['sources'] + r['artifacts']:
    assert sha(ROOT / item['path']) == item['sha256'], item['path']
for s in r['sources']:
    assert s['verified'] and s['checked_original']
    if s['kind'] in ['official_source', 'paper']:
        assert s['url'].startswith('https://'), s['id']
figures = {}
for p in r['pages']:
    path = ROOT / p['source']
    expected = EXPECTED[p['source']]
    assert p['source_sha256'] == p['frozen_input_sha256'] == expected
    assert sha(path) == sha(ROOT / p['frozen_input_path']) == expected
    assert p['read_scope'].startswith('Full current')
    direct = {}
    for rel in re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)', path.read_text()):
        f = (path.parent / rel).resolve()
        direct[str(f.relative_to(ROOT))] = sha(f)
    assert direct == p['figure_sha256']
    figures.update(direct)
    assert p['verdict'] == ('revise' if any(c['status'] != 'verified' for c in p['claims']) else 'pass')
    assert set(p['checks']) == {'factual_accuracy', 'numeric_verification', 'figure_consistency', 'source_verification', 'limitations'}
    for c in p['claims']:
        assert c['kind'] and c['statement'] and c['location'] and c['scope']
        assert c['status'] in ['verified', 'contradicted', 'unresolved']
        assert c['evidence']
        for e in c['evidence']:
            assert e['source_id'] in source_ids and e['locator'] and e['supports']
        assert set(c['artifact_ids']) <= artifact_ids
assert r['figure_sha256'] == figures and len(figures) == 2
render_widths = collections.Counter()
for a in r['artifacts']:
    if a['kind'] == 'figure_render':
        b = (ROOT / a['path']).read_bytes()
        assert b[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', b[16:24])
        assert width in [640, 360] and height > 0
        render_widths[width] += 1
assert render_widths == {640: 2, 360: 2}
issue_claims = {cid for issue in r['issues'] for cid in issue['claim_ids']}
assert issue_claims == {'s4', 'a8', 'a2'}
assert issue_claims == {c['id'] for c in claims if c['status'] != 'verified'}
assert not r['independence']['old_reader_or_technical_report_bodies_read']
assert not r['independence']['author_extra_result_notes_read']
assert all(r['execution_boundary'][k] is False for k in [
    'installed_packages', 'uploads', 'paid_compute', 'full_training', 'model_generation', 'new_holdout_answer_grading'])
result = {'status': 'passed', 'report': str(REPORT.relative_to(ROOT)), 'report_sha256': sha(REPORT),
    'pages': len(r['pages']), 'claims': len(claims), 'sources': len(r['sources']),
    'artifacts': len(r['artifacts']), 'figures': len(figures), 'render_widths': dict(render_widths),
    'initial_page_hashes_preserved': True, 'all_source_artifact_hashes_match': True,
    'all_claim_references_resolve': True, 'verdict': r['verdict'],
    'command': '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/validate_initial_report.py'}
(BASE / 'initial-report-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
