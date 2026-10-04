"""Run unmodified official read-only components on this supplemental report only."""
from pathlib import Path
import hashlib, importlib.util, json, sys
ROOT = Path(__file__).resolve().parents[6]
EVIDENCE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('original_technical_checker', ROOT / 'scripts/check_technical_reviews.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = json.loads((EVIDENCE / 'report.json').read_text())
errors = []
artifacts = checker._artifacts(ROOT, report.get('artifacts'), errors)
sources = checker._sources(ROOT, report.get('sources'), artifacts, errors)
claims = checker._claims(report.get('claims'), sources, artifacts, errors)
assert 'lesson_id' not in report
assert report['reviewer_task'] == '/root/v4_review_coordinator/factual_guide_training_v4'
assert checker._task(report['reviewer_task']) and report['reviewer_context'] == 'fresh'
assert report['schema_version'] == 1 and report['review_stage'] == 'technical'
assert report['complete_documents'] == ['docs/natural-assistant/v4/TRAINING.md']
assert report['assigned_document_scope']['start_line'] == 1 and report['assigned_document_scope']['end_line'] == 327
assert report['verdict'] == 'pass'
for p, digest in report['current_document_sha256'].items(): checker._hash_file(ROOT, p, digest, errors, 'complete document')
assert report['source_sha256'] == report['current_document_sha256'][report['source']]
for p, digest in report['figure_sha256'].items(): checker._hash_file(ROOT, p, digest, errors, 'necessary figure')
assert set(report['checks']) == set(checker.CHECKS)
for name in checker.CHECKS:
    value = report['checks'][name]
    if value['status'] != 'pass' or not checker._text(value['details']): errors.append(name + ': incomplete check')
    checker._references(value['claim_ids'], claims, errors, name)
for issue in report['issues']:
    if issue['status'] != 'resolved' or not checker._text(issue.get('resolution')): errors.append('unresolved issue')
assert (EVIDENCE / 'report.json').read_bytes() == (EVIDENCE / 'report.round2.initial.original.json').read_bytes()
originals = json.loads((EVIDENCE / 'round2/preserved-original-bytes.json').read_text())['actually_checked']
for original in originals: checker._hash_file(ROOT, original['path'], original['sha256'], errors, 'preserved original')
result = {'command': '.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/training-v4/round2/check_report.py', 'official_checker_path': 'scripts/check_technical_reviews.py', 'official_checker_sha256': sha(ROOT / 'scripts/check_technical_reviews.py'), 'executed_original_components': ['_artifacts', '_sources', '_claims', '_hash_file', '_references', '_task'], 'scope': 'Supplemental whole-file guide has no numbered lesson_id. Original read-only components were actually imported and executed, without modifying checker, fabricating lesson identity, loading other reports, or claiming global publication inventory validation.', 'errors': errors, 'report_sha256': sha(EVIDENCE / 'report.json'), 'current_document_sha256': report['current_document_sha256'], 'figure_sha256': report['figure_sha256'], 'claims': len(claims), 'sources': len(sources), 'artifacts': len(artifacts), 'remaining_issue_ids': [i['id'] for i in report['issues'] if i['status'] != 'resolved'], 'environment': {'python': sys.version, 'device': 'cpu'}}
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
