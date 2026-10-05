"""Unmodified official read-only schema components, own supplemental guide only."""
from pathlib import Path
import hashlib, importlib.util, json, sys
ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
EVIDENCE = OUT.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('official_technical_checker', ROOT / 'scripts/check_technical_reviews.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
report = json.loads((EVIDENCE / 'report.json').read_text())
errors = []
artifacts = checker._artifacts(ROOT, report.get('artifacts'), errors)
sources = checker._sources(ROOT, report.get('sources'), artifacts, errors)
claims = checker._claims(report.get('claims'), sources, artifacts, errors)
assert 'lesson_id' not in report
assert report['reviewer_task'] == '/root/v4_review_coordinator/factual_guide_training_v4'
assert checker._task(report['reviewer_task']) and report['reviewer_context'] == 'fresh'
assert report['complete_documents'] == ['docs/natural-assistant/v4/TRAINING.md']
assert report['schema_version'] == 1 and report['review_stage'] == 'technical'
assert (EVIDENCE / 'report.json').read_bytes() == (OUT / 'report.initial.original.json').read_bytes()
for p, h in report['current_document_sha256'].items(): checker._hash_file(ROOT, p, h, errors, 'unchanged whole-document source')
for p, h in report['figure_sha256'].items(): checker._hash_file(ROOT, p, h, errors, 'current registered figure')
for p, h in report['current_registered_dependency_sha256'].items(): checker._hash_file(ROOT, p, h, errors, 'current necessary dependency')
assert set(report['checks']) == set(checker.CHECKS)
for name in checker.CHECKS:
    check = report['checks'][name]
    if check['status'] != 'pass' or not checker._text(check['details']): errors.append('incomplete check: ' + name)
    checker._references(check['claim_ids'], claims, errors, name)
for issue in report['issues']:
    if issue['status'] != 'resolved' or not checker._text(issue.get('resolution')): errors.append('unresolved issue')
receipt = json.loads((OUT / 'recheck-receipt.json').read_text())
for item in receipt['initial_probe_failures_preserved'] + receipt['first_issue_and_first_report_original_bytes_rechecked']:
    checker._hash_file(ROOT, item['path'], item['sha256'], errors, 'preserved original or failure')
assert sha(OUT / 'report.prior.original.json') == 'aafd27515cb0415bb187c001795df9130e7284deabcc256a83f04597f02a3ce1'
print(json.dumps({'command': '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-v4/round3-answer-mask/check_report.py', 'executed_original_components': ['_artifacts', '_sources', '_claims', '_hash_file', '_references', '_task'], 'official_checker_sha256': sha(ROOT / 'scripts/check_technical_reviews.py'), 'scope': 'Own registered-figure corrective report/schema/hash checks; no fabricated lesson identity, peer report reading, product/checker edits or whole-release/inventory rerun.', 'errors': errors, 'report_sha256': sha(EVIDENCE / 'report.json'), 'verdict': report['verdict'], 'current_document_sha256': report['current_document_sha256'], 'figure_sha256': report['figure_sha256'], 'current_registered_dependency_sha256': report['current_registered_dependency_sha256'], 'claims': len(claims), 'sources': len(sources), 'artifacts': len(artifacts), 'remaining_issue_ids': [i['id'] for i in report['issues'] if i['status'] != 'resolved'], 'environment': {'python': sys.version, 'device': 'cpu'}}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
