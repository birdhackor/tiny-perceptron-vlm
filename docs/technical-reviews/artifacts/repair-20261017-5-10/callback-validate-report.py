"""Validate the current 5.10 callback report only, preserving original reports."""

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

root = Path(__file__).resolve().parents[4]
tool = root / 'scripts/check_technical_reviews.py'
spec = importlib.util.spec_from_file_location('technical_checker', tool)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
path = root / 'docs/course-repair-20261017/reviews/technical-5-10-current-01.json'
source = root / 'course/chapters/05.md'
report = json.loads(path.read_text())
body = dict(checker.sections(source))['5.10']
owners = {'/root/repair_tech_5_10': ['5.10']}
readers = ['/root/astra_repair_01_readability', '/root/repair_local_reader_05']
errors = checker._validate(root, source, '5.10', body, report, owners, readers)
print(json.dumps({
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-5-10/callback-validate-report.py > docs/technical-reviews/artifacts/repair-20261017-5-10/callback-checker-output.json',
    'reviewer_task': '/root/repair_tech_5_10',
    'current_context': 'informed_original_owner_source_revision_callback',
    'reviewer_owners': owners,
    'reader_tasks': readers,
    'report_path': str(path.relative_to(root)),
    'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'checker_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
    'environment': {'python': platform.python_version(), 'device': 'cpu'},
    'errors': errors,
    'interpretation': 'Checks structural rules, source/artifact hashes and declared reviewer identities only; does not automatically establish content correctness or a new fresh review.',
}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
