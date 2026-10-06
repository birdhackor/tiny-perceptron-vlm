from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
REPORT = ROOT / 'docs/technical-reviews/public-pages-p6-b.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

errors = []
checks = []

def check(ok, label):
    checks.append({'check': label, 'pass': bool(ok)})
    if not ok:
        errors.append(label)

def file_sha(path, expected, label):
    path = ROOT / path
    check(path.is_file(), label + ': file exists')
    if path.is_file():
        check(sha(path) == expected, label + ': SHA256 agrees')

d = json.loads(REPORT.read_text())
check(d['schema'] == 'public-pages-independent-factual-v1' and d['schema_version'] == 1, 'custom report schema identity')
check(d['review_stage'] == 'technical' and d['reviewer_task'] == '/root/p6_fact_public_b' and d['reviewer_context'] == 'fresh', 'reviewer identity and initial independent context')
check(d['verdict'] == 'pass', 'current aggregate verdict')
expected_scopes = {
    'docs/selftrained/v2-public-cpu-commands.md': 'full file',
    'docs/selftrained/TRAINING.md': 'full file',
    'docs/selftrained/model-cards/repository-README.md': 'full file',
    'docs/natural-assistant/v4/STUDENT.md': 'lines 1-10',
    'docs/natural-assistant/v4/DATA.md': 'lines 1-10',
    'docs/natural-assistant/v4/TRAINING.md': 'lines 1-10',
}
check(len(d['files']) == 6 and {f['source']: f['scope'] for f in d['files']} == expected_scopes, 'exact six assigned scopes')
source_ids = [s['id'] for s in d['sources']]
artifact_ids = [a['id'] for a in d['artifacts']]
check(len(source_ids) == len(set(source_ids)), 'unique source identifiers')
check(len(artifact_ids) == len(set(artifact_ids)), 'unique artifact identifiers')
claim_ids = []
for f in d['files']:
    current = (ROOT / f['source']).read_bytes()
    scoped = current if f['scope'] == 'full file' else b''.join(current.splitlines(keepends=True)[:10])
    check(hashlib.sha256(current).hexdigest() == f['full_file_sha256'], f['source'] + ': current full-file fingerprint')
    check(hashlib.sha256(scoped).hexdigest() == f['source_sha256'], f['source'] + ': current assigned-scope fingerprint')
    frozen = (ROOT / f['frozen_input']).read_bytes()
    frozen_scoped = frozen if f['scope'] == 'full file' else b''.join(frozen.splitlines(keepends=True)[:10])
    check(frozen_scoped == scoped, f['source'] + ': frozen source equals current assigned scope')
    check(f['verdict'] == 'pass', f['source'] + ': current file verdict')
    check(set(f['checks']) == {'factual_accuracy', 'numeric_verification', 'figure_consistency', 'source_verification', 'limitations'}, f['source'] + ': mandatory check fields')
    for name, value in f['checks'].items():
        check(value['status'] == ('not_applicable' if name == 'figure_consistency' else 'pass') and bool(value['details']), f['source'] + ': ' + name)
    if f['source'] != 'docs/selftrained/model-cards/repository-README.md':
        check(f['recheck'] == {'scope_sha256_unchanged': True, 'full_file_sha256_unchanged': True, 'original_personal_checks_retained': True, 'checks_rerun': False, 'evidence_versions_revalidated': True}, f['source'] + ': unchanged-scope recheck attribution')
    for c in f['claims']:
        claim_ids.append(c['id'])
        check(c['status'] == 'verified' and all(bool(c[k]) for k in ['id', 'kind', 'statement', 'location', 'scope']), c['id'] + ': verified claim fields')
        check(bool(c['artifact_ids']) and all(a in artifact_ids for a in c['artifact_ids']), c['id'] + ': valid artifact references')
        for e in c['evidence']:
            check(e['source_id'] in source_ids and bool(e['locator']) and bool(e['supports']), c['id'] + ': source reference ' + e['source_id'])
check(len(claim_ids) == 56 and len(claim_ids) == len(set(claim_ids)), '56 unique individually verified claims')

for a in d['artifacts']:
    file_sha(a['path'], a['sha256'], 'artifact ' + a['id'])

for s in d['sources']:
    check(s['verified'] is True, 'source verified ' + s['id'])
    if s['kind'] == 'repository_code':
        file_sha(s['path'], s['sha256'], 'repository source ' + s['id'])
    elif s['kind'] == 'official_source':
        check(s['checked_original'] is True and s['url'].startswith('https://'), 'official original inspected ' + s['id'])
        expected = re.search(r'SHA256\s*([0-9a-f]{64})', s['version'])
        check(expected is not None, 'official source version contains SHA256 ' + s['id'])
        if expected:
            file_sha(str((BASE / 'official' / s['id']).relative_to(ROOT)), expected[1], 'official immutable evidence snapshot ' + s['id'])
    else:
        check(False, 'supported source kind ' + s['id'])

for row in json.loads((BASE / 'code-source-manifest.json').read_text()):
    file_sha(row['snapshot'], row['sha256'], 'frozen repository-code bytes ' + row['path'])

history = d['review_history']
check(history[0]['verdict'] == 'revise' and history[0]['issues'] == ['card-text-exact', 'card-ocr'], 'initial verdict and issues retained')
h = history[-1]
for key in ['initial_report', 'opaque_formal_backup']:
    file_sha(h[key], h[key + '_sha256'], 'opaque original report ' + key)
check((ROOT / h['initial_report']).read_bytes() == (ROOT / h['opaque_formal_backup']).read_bytes(), 'both original report copies byte-identical')
initial = json.loads((ROOT / h['initial_report']).read_text())
check(initial['verdict'] == 'revise', 'preserved report carries actual initial verdict')
for old in initial['issues']:
    current = next(i for i in d['issues'] if i['claim_id'] == old['claim_id'])
    check(all(current[k] == old[k] for k in ['claim_id', 'file', 'details', 'suggestion']), old['claim_id'] + ': original issue text retained')
    check(current['status'] == 'resolved' and current['resolved_source_sha256'] == h['current_model_card_sha256'] and all(a in artifact_ids for a in current['resolution_artifact_ids']), old['claim_id'] + ': resolution refers to current source and evidence')
check(len(d['issues']) == 2 and all(i['status'] == 'resolved' for i in d['issues']), 'no unresolved technical issues')
check(h['current_model_card_lines_read'] == '1-183,fullfile' and h['new_generations'] == 0 and h['training_runs'] == 0 and h['body_edits_by_reviewer'] is False, 'actual recheck scope and no new generation/training/body edits')

probe = json.loads((OUT / 'probe-result.json').read_text())
file_sha('scripts/selftrained/evaluate.py', probe['normalize']['source_sha256'], 'normalization probe current implementation')
file_sha('docs/selftrained/model-cards/repository-README.md', probe['normalize']['current_card_sha256'], 'normalization probe current card')
file_sha('docs/selftrained/v2-manifest.json', probe['package']['manifest_sha256'], 'package probe current manifest')
file_sha('assets/training/selftrained-v2.tar.gz', probe['package']['archive_sha256'], 'package probe actual LFS archive')
check(probe['normalize']['documented_equals_actual'] and probe['normalize']['documented_strip_codepoints'] == probe['normalize']['strip_character_codepoints'], 'documented normalizer exact nine-codepoint agreement')
check(probe['package']['actual_files'] == 8962 and probe['package']['record_declarations'] == 12 and probe['package']['other_declarations'] == 8950 and probe['package']['other_file_suffix_counts'] == {'.md': 5, '.wav': 603, '.png': 8335, '': 1, '.txt': 3, '.json': 3}, 'actual record/other-file classification agrees')
versions = json.loads((OUT / 'evidence-version-check.json').read_text())
check(versions['evidence_errors'] == [] and versions['original_raw_input_audit_rows_checked'] == {'input-audit.json': 80, 'extra-input-audit.json': 40}, 'original raw receipt/source audit has no version mismatch')

result = {
    'validator': str(Path(__file__).relative_to(ROOT)),
    'report': str(REPORT.relative_to(ROOT)),
    'report_sha256': sha(REPORT),
    'current_model_card_sha256': h['current_model_card_sha256'],
    'verdict': 'pass' if not errors else 'revise',
    'checked_scopes': 6,
    'claims': len(claim_ids),
    'artifacts': len(d['artifacts']),
    'sources': len(d['sources']),
    'checks': checks,
    'errors': errors,
}
(OUT / 'final-report-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False))
raise SystemExit(bool(errors))
