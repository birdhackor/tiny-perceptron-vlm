"""Run the existing technical validator against this owner's current 9.8 report."""

import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[4]
checker_path = root / 'scripts/check_technical_reviews.py'
spec = importlib.util.spec_from_file_location('technical_checker', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
source = root / 'course/chapters/09.md'
body = dict(checker.sections(source))['9.8']
report_path = root / 'docs/course-repair-20261017/reviews/technical-9-8-current-01.json'
report = json.loads(report_path.read_text())
owners = {'/root/repair_tech_9_8': ['9.8']}
readers = ['/root/astra_repair_01_readability', '/root/repair_local_reader_05']
errors = checker._validate(root, source, '9.8', body, report, owners, readers)
result = {
    'reviewer_task': '/root/repair_tech_9_8',
    'current_context': 'informed_original_owner_source_revision_callback',
    'checked_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-9-8/callback-checker-current-01.py',
    'checker': 'scripts/check_technical_reviews.py::_validate',
    'checker_sha256': hashlib.sha256(checker_path.read_bytes()).hexdigest(),
    'report': str(report_path.relative_to(root)),
    'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'reviewer_owners': owners,
    'reader_tasks': readers,
    'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu; validator uses standard library only'},
    'errors': errors,
    'interpretation': 'errors=[] only validates structure, references, identity declarations, and hashes; it is not automatic proof of technical correctness or an independent fresh review.',
}
serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
destination = Path(__file__).parent / 'callback-checker-output.json'
with destination.open('x') as handle:
    handle.write(serialized)
print(serialized, end='')
raise SystemExit(bool(errors))
