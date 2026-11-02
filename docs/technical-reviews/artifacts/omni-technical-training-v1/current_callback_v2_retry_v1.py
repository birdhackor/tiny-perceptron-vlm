"""保留初次404，僅修本reviewer把flat HTML誤寫成directory的探針路徑。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'current_callback_v2.py'
original = source.read_text()
changes = [
    ("('training/', site / 'training/index.html')", "('training.html', site / 'training.html')"),
    ("if tail == 'training/':", "if tail == 'training.html':"),
]
for old, new in changes:
    assert original.count(old) == 1
    original = original.replace(old, new)
receipt = {
    'recorded_at': datetime.now(UTC).isoformat(),
    'failed_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
    'actual_failed_command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-training-v1/current_callback_v2.py',
    'exit_code': 1,
    'failure': 'urllib.error.HTTPError: HTTP Error 404: File not found',
    'failed_location': 'current_callback_v2.py main line201, HTTP loop; /training/ directory route assumption',
    'diagnosis': 'Actual preview is flat HTML: build-info.json200; training/404; training.html200; notebooks/12/12.2.ipynb200. No site/checker/source changed.',
    'retry_source_replacements': changes,
    'effective_retry_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'scope': 'First attempt failed before intake/pytest/history CLI commands and before creating callback-current-v2.json. Original code and reports preserved.',
}
with (HERE / 'callback-current-v2-initial-http-failure-v1.json').open('x', encoding='utf-8') as f:
    f.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
exec(compile(original, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
