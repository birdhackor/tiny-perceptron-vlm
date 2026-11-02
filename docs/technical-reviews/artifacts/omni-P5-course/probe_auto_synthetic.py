"""Artificial stubs check existing Auto control flow without neural models."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from tiny_perceptron.lesson_products.inference import auto_solve


class B:
    auto_policy = {'version': 1, 'sum_depth': 1, 'low_max_events': 1, 'medium_max_events': 3}

    def synchronize(self):
        pass

    def solve(self, state, loops):
        return {'answer': 'synthetic', 'loops': loops}


class C:
    def __init__(self, choice, info):
        self.choice, self.info = choice, info

    def synchronize(self):
        pass

    def judge(self, state, kind):
        return {'selected': self.choice if kind == 'choice' else self.info, 'seconds': 0.0}


state = '兩盒追蹤;A=2;B=6;事件=交換;查詢=B'
rows = []
for choice, info, expected in [('none', 'yes', 'unsupported'), ('boxes', 'no', 'clarification_needed'), ('boxes', 'yes', 'answered')]:
    result = auto_solve(B(), C(choice, info), state)
    assert result['status'] == expected
    assert result['b_loops'] == (1 if expected == 'answered' else 0)
    rows.append({'choice': choice, 'info': info, 'result': result})
value = {'created_at': datetime.now(timezone.utc).isoformat(), 'environment': {'python': platform.python_version(), 'device': 'CPU control flow only'},
         'scope': 'Artificial stub inputs exercising existing auto_solve control flow; no neural model or weights', 'rows': rows}
Path(__file__).with_name('auto-synthetic-probe-original.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(value, ensure_ascii=False))
