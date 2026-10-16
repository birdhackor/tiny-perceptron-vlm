"""對current完整頁報告實跑既有_validate；僅格式／指紋檢查。"""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from scripts.check_technical_reviews import _validate


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'course/training.md'
REPORT = ROOT / 'docs/course-repair-20261017/reviews/technical-training-current-01.json'
OUT = Path(__file__).parent / 'callback-checker-output.json'
assert not OUT.exists()
reviewer_owners = {'/root/repair_tech_training': ['training']}
reader_tasks = {'/root/astra_repair_01_readability', '/root/repair_local_reader_05'}
body = SOURCE.read_text()
report = json.loads(REPORT.read_text())
errors = _validate(ROOT, SOURCE, 'training', body, report, reviewer_owners, reader_tasks)
receipt = {
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-training/callback-validate-freeze03.py',
    'environment': {'python': platform.python_version(), 'device': 'CPU metadata only'},
    'reviewer_task': '/root/repair_tech_training',
    'current_context': 'informed_original_owner_source_revision_callback',
    'source': str(SOURCE.relative_to(ROOT)),
    'lesson_id': 'training',
    'body_scope': 'whole-file, 574 lines; no sections() or T.10 fragment',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'report': str(REPORT.relative_to(ROOT)),
    'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    'reviewer_owners': reviewer_owners,
    'reader_tasks': sorted(reader_tasks),
    'errors': errors,
    'format_passed': not errors,
    'completed_at': datetime.now(UTC).isoformat(),
    'interpretation': '僅證明結構、路徑、hash及身分欄位符合既有checker；不能代替本人對正文、實作及實際CPU結果的技術判斷。',
}
OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
