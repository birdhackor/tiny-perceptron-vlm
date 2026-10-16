"""Execute only lesson 7.25's authored, hypothetical reporting example."""
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from scripts.reading_time import lesson_slices

body = lesson_slices((root / 'course/chapters/07.md').read_bytes().decode('utf-8'))['7.25']
freeze = json.loads((root / 'docs/course-revision-20261016/freeze-08.json').read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == '7.25')
digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
assert digest == page['source_sha256']
code = re.findall(r'```python\n(.*?)```', body, re.S)[0]
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code, 'course/chapters/07.md#7.25', 'exec'), namespace)
expected = 'low 正確率 0.5 平均工作 1.0\nmedium 正確率 1.0 平均工作 3.0\nhigh 正確率 0.75 平均工作 5.0\n'
assert output.getvalue() == expected
logs = namespace['logs']
costs = {k: sum(v['work']) / len(v['work']) for k, v in logs.items()}
allowed = [k for k, v in costs.items() if v <= 3]
assert allowed == ['low', 'medium']
oracle = [max(logs[k]['ok'][i] for k in logs) for i in range(4)]
assert oracle == [1, 1, 1, 1]
assert max([0, 0, 0]) == 0
result = {
    'reviewer_task': '/root/tech_20261016_7_25',
    'command': 'uv run python docs/technical-reviews/artifacts/revision-20261016/7.25-verify.py',
    'source_sha256': digest,
    'freeze_08_match': True,
    'authored_code': code,
    'stdout': output.getvalue(),
    'expected_stdout': expected,
    'sample_denominator': '4 hypothetical questions per preset; no trained model or measured service',
    'allowed_at_average_3_rounds': allowed,
    'oracle_ok_over_given_outputs': oracle,
    'oracle_if_all_wrong': 0,
    'assertions': 'all passed',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; pure Python; no torch, inference, or training'},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
