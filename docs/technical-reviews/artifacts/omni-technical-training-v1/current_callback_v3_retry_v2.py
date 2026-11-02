"""第二個真失敗保留：舊8775已停；source15實際預覽為8776。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
original_path = HERE / 'current_callback_v3.py'
source = original_path.read_text()
for line in [
    '    ("\'source14/bundle4", "\'source15/bundle5"),\n',
    '    ("1 new12.2 kernel", "1 new5.7 kernel"),\n',
]:
    assert source.count(line) == 1
    source = source.replace(line, '')
source = source.replace("context_path = HERE / 'callback-current-v3-context.json'; save(context_path, context)",
                        "context_path = HERE / 'callback-current-v3-context.json'; assert context_path.is_file()")
source = source.replace("source = source.replace('First109', 'Original109')", "source = source.replace('First109', 'Original109').replace('8775', '8776')")
source = source.replace("callback-current-v3-transform.json", "callback-current-v3-transform-v2.json")
receipt = {'created_at': datetime.now(UTC).isoformat(), 'failed_command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-training-v1/current_callback_v3_retry_v1.py',
           'failed_source': {'path': str(HERE / 'current_callback_v3_retry_v1.py'), 'sha256': hashlib.sha256((HERE / 'current_callback_v3_retry_v1.py').read_bytes()).hexdigest()},
           'exit_code': 1, 'failure': 'urllib.error.URLError: <urlopen error [Errno 111] Connection refused>',
           'diagnosis': 'Reused audit retained prior source14 port8775, whose server is stopped. Source15 task explicitly supplied8776; verified actual8776 build-info/training.html/5.7download GET200. Correcting this reviewer reference does not change site, source or checker.',
           'preserved_original_transform': {'path': str(HERE / 'callback-current-v3-transform.json'), 'sha256': hashlib.sha256((HERE / 'callback-current-v3-transform.json').read_bytes()).hexdigest()},
           'effective_retry_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'scope': 'Initial semantic context and both failed drivers/receipts remain immutable. Failure occurred before intake or new rawresult was saved.'}
with (HERE / 'callback-current-v3-old-port-failure-v1.json').open('x', encoding='utf-8') as f:
    f.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
exec(compile(source, str(original_path), 'exec'), {'__file__': str(original_path), '__name__': '__main__'})
