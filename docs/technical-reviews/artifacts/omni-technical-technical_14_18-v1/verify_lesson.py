"""Execute the exact frozen 14.18 snippet and its stated exercise on CPU."""
from contextlib import redirect_stdout
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys

import torch

root = Path(__file__).resolve().parents[4]
freeze_path = root / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
freeze = json.loads(freeze_path.read_bytes())
page = next(p for p in freeze['pages'] if p['page_id'] == '14.18')
raw = (root / page['snapshot']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page['source_sha256']
chapter = (root / page['source']).read_text()
start = re.search(r'^## 14\.18\s', chapter, re.M).start()
next_section = re.search(r'^## ', chapter[start + 1:], re.M)
end = start + 1 + next_section.start() if next_section else len(chapter)
assert chapter[start:end].encode() == raw
assert page['figures_sha256'] == {}
blocks = re.findall(r'```python\n(.*?)\n```', raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0]
expected = [
    '共享一個區塊跑一輪 參數 1200000 每題區塊執行 1 次',
    '共享一個區塊跑五輪 參數 1200000 每題區塊執行 5 次',
    '堆疊五個區塊 參數 5200000 每題區塊執行 5 次',
]

def run(source):
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, str(root / page['snapshot']), 'exec'), namespace)
    return namespace, output.getvalue().splitlines()

namespace, observed = run(code)
assert observed == expected, (observed, expected)
assert namespace['other'] == 200000 and namespace['block'] == 1000000
assert namespace['designs'] == [('共享一個區塊跑一輪', 1, 1), ('共享一個區塊跑五輪', 1, 5), ('堆疊五個區塊', 5, 5)]
exercise_code = code.replace('for name, copies, rounds in designs:', 'designs.append(("堆疊兩個區塊", 2, 2))\nfor name, copies, rounds in designs:')
exercise_namespace, exercise_observed = run(exercise_code)
exercise_expected = expected + ['堆疊兩個區塊 參數 2200000 每題區塊執行 2 次']
assert exercise_observed == exercise_expected
pairs = [(namespace['other'] + copies * namespace['block'], rounds) for _, copies, rounds in exercise_namespace['designs']]
assert pairs[0][0] == pairs[1][0] and pairs[1][1] == pairs[2][1]
assert all(pairs[3][0] != pair[0] and pairs[3][1] != pair[1] for pair in pairs[:3])
notebook_raw = (root / page['notebook']).read_bytes()
assert hashlib.sha256(notebook_raw).hexdigest() == page['notebook_sha256']
notebook = json.loads(notebook_raw)
notebook_sources = [''.join(cell['source']).strip() for cell in notebook['cells'] if cell['cell_type'] == 'code']
assert any(code in source for source in notebook_sources)
record = {
    'reviewer_task': '/root/omni_technical_coordinator/technical_14_18',
    'executed_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_14_18-v1/verify_lesson.py',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'CPU; arithmetic snippet uses Python integers; no GPU or training'},
    'freeze': str(freeze_path.relative_to(root)),
    'source_snapshot': page['snapshot'],
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'source_current_equals_snapshot': True,
    'notebook_sha256': hashlib.sha256(notebook_raw).hexdigest(),
    'exact_snippet_in_notebook': True,
    'figure_count': 0,
    'executed_frozen_code': code,
    'expected': expected,
    'observed': observed,
    'exercise_change': 'Append ("堆疊兩個區塊", 2, 2) before the for loop',
    'exercise_expected': exercise_expected,
    'exercise_observed': exercise_observed,
    'parameter_round_pairs': pairs,
    'integer_tolerance': 'exact equality',
    'comparison_checks': {'fixed_parameter_first_second': True, 'fixed_round_second_third': True, 'exercise_matches_no_original_parameter_or_round_count': True},
    'result': 'All frozen snippet, exercise, exact integer output, notebook presence and freeze checks passed. No quality, latency, peak memory or GPU speed was measured.'
}
print(json.dumps(record, ensure_ascii=False, indent=2))
