"""Own-section validation using unchanged official checker functions.

Does not run the global main or open other readers' reports.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
out = Path(__file__).resolve().parent
checker_path = root / 'scripts/check_course_reviews.py'
checker_sha = hashlib.sha256(checker_path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('unchanged_official_reader_checker', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

report_path = root / 'docs/reader-reviews/11.14.json'
report_raw = report_path.read_bytes()
report = json.loads(report_raw)
source_path = root / 'course/chapters/11.md'
body = next(body for lesson, body in checker.sections(source_path) if lesson == '11.14')
checks = {name: checker.checked(value, allow_not_applicable=(name == 'program_explanation' and '```' not in body))
          for name, value in report['checks'].items()}
source_sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
assert report['lesson_id'] == '11.14'
assert report['source'] == 'course/chapters/11.md#11.14'
assert report['reviewer_task'] == '/root/v4_review_coordinator/reader_final_11_14'
assert report['reviewer_context'] == 'fresh'
assert report['verdict'] == 'pass'
assert source_sha == report['source_sha256'] == '8d338be14a26e0b7a4a48b01e85741e687e37b8f6be05e25fa5bd8c2110cd6be'
assert (out / 'section-snapshot.md').read_bytes() == body.encode('utf-8')
assert report['reader_summary'] and isinstance(report['issues'], list)
assert set(['background_and_links', 'terminology', 'examples', 'program_explanation', 'exercise']).issubset(checks)
assert all(checks.values())
required_figures = {(source_path.parent / ref).resolve().relative_to(root).as_posix()
                    for ref in re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)', body)}
required_figures.update(report['figure_sha256'])
for path in required_figures:
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == report['figure_sha256'][path]
assert checker_sha == hashlib.sha256(checker_path.read_bytes()).hexdigest()
round1 = out.parent / 'round1'
assert hashlib.sha256((round1 / 'report.json').read_bytes()).hexdigest() == '888aeaf605d69189483847007d068956fdb0fa2e50eb48bb4f0f96f339a3f88a'
assert hashlib.sha256((round1 / 'section-snapshot.md').read_bytes()).hexdigest() == '8a8d6b6441b5e66b01424cbd3b8eeee7d6191ebba47b41afb3a187e7948ce12e'
assert (out / 'report.json').read_bytes() == report_raw
result = {'kind': 'target-only check with unchanged official sections() and checked()',
          'command': 'env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/reader-reviews/artifacts/natural-v4-cleanup/11.14/round2/check-own-report.py',
          'official_checker': str(checker_path), 'official_checker_sha256_before_and_after': checker_sha,
          'global_main_run': False, 'other_reader_reports_opened': False,
          'global_duplicate_reviewer_check': 'not run; coordinator may run the full official main',
          'checks': checks, 'current_source_sha256': source_sha,
          'current_figure_sha256': report['figure_sha256'],
          'actual_report_sha256': hashlib.sha256(report_raw).hexdigest(),
          'verdict': report['verdict'], 'remaining_issues': len(report['issues']),
          'round1_original_report_preserved': True, 'round1_raw_section_preserved': True,
          'report': str(report_path), 'evidence_directory': str(out), 'status': 'pass'}
(out / 'final-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
