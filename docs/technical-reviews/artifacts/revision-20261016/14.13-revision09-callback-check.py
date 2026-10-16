"""Informed 14.13 callback: compare frozen sources and rerun scalar gradients."""
import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

old_report_path = ROOT / 'docs/course-revision-20261016/reviews/technical-14.13-original.json'
old_report_bytes = old_report_path.read_bytes()
old_report = json.loads(old_report_bytes)
f8 = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
f9 = json.loads((ROOT / 'docs/course-revision-20261016/freeze-09.json').read_text())
p8 = next(p for p in f8['pages'] if p['page_id'] == '14.13')
p9 = next(p for p in f9['pages'] if p['page_id'] == '14.13')
b8 = (ROOT / p8['snapshot']).read_text()
b9 = (ROOT / p9['snapshot']).read_text()
current = lesson_slices((ROOT / 'course/chapters/14.md').read_text())['14.13']
def sha(b):
    return hashlib.sha256(b).hexdigest()
def digest(text):
    return sha(text.encode())
def get_code(body):
    return re.findall(r'```python\n(.*?)```', body, re.S)[0]
c8, c9 = get_code(b8), get_code(b9)
ast_equal = ast.dump(ast.parse(c8), include_attributes=False) == ast.dump(ast.parse(c9), include_attributes=False)
noncode_equal = b8.replace(c8, '<PYTHON_CODE>') == b9.replace(c9, '<PYTHON_CODE>')
assert digest(b8) == p8['source_sha256'] == old_report['source_sha256']
assert digest(b9) == p9['source_sha256']
assert b9 == current
assert ast_equal and noncode_equal
assert p8['figures_sha256'] == p9['figures_sha256'] == {}
ns = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(c9, p9['snapshot'], 'exec'), ns)
assert stdout.getvalue() == 'state 0.6 loss 0.16 grad -2.4\n'
three = {'state': ns['state'].item(), 'loss': ns['loss'].item(), 'grad': ns['w'].grad.item(), 'w_after_backward': ns['w'].item()}
for key, value in [('state', 0.6), ('loss', 0.16), ('grad', -2.4), ('w_after_backward', 0.2)]:
    assert abs(three[key] - value) < 1e-6
state = torch.tensor(0.0)
for _ in range(2):
    state = state + torch.tensor(0.2)
two = {'state': state.item(), 'loss': (state - 1).square().item()}
assert abs(two['state'] - 0.4) < 1e-6 and abs(two['loss'] - 0.36) < 1e-6
w = torch.tensor(0.2, requires_grad=True)
state = torch.tensor(0.0)
states = []
for _ in range(3):
    state = state + w
    state.retain_grad()
    states.append(state)
(state - 1).square().backward()
full = {'state': state.item(), 'w_grad': w.grad.item(), 'state_grads': [s.grad.item() for s in states]}
ws = [torch.tensor(0.2, requires_grad=True) for _ in range(3)]
s = torch.tensor(0.0)
for wi in ws:
    s = s + wi
(s - 1).square().backward()
contributions = [wi.grad.item() for wi in ws]
assert all(abs(g + 0.8) < 1e-6 for g in contributions)
assert abs(sum(contributions) - full['w_grad']) < 1e-6
w = torch.tensor(0.2, requires_grad=True)
state = torch.tensor(0.0)
with torch.no_grad():
    for _ in range(2):
        state = state + w
state = state + w
(state - 1).square().backward()
truncated = {'state': state.item(), 'w_grad': w.grad.item()}
assert abs(truncated['state'] - full['state']) < 1e-6
assert abs(truncated['w_grad'] + 0.8) < 1e-6
x = Fraction(1, 5)
analytic = {'s3': str(3*x), 'L3': str((3*x-1)**2), 'dL3_dw': str(2*(3*x-1)*3), 's2': str(2*x), 'L2': str((2*x-1)**2)}
assert analytic == {'s3': '3/5', 'L3': '4/25', 'dL3_dw': '-12/5', 's2': '2/5', 'L2': '9/25'}
assert old_report_path.read_bytes() == old_report_bytes
record = {
 'schema_version': 1,
 'reviewer_task': '/root/tech_20261016_14_13',
 'review_context': 'informed callback after original fresh independent review',
 'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.13-revision09-callback-check.py',
 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'dtype': str(torch.get_default_dtype())},
 'comparison': {'old_snapshot': p8['snapshot'], 'new_snapshot': p9['snapshot'], 'old_source_sha256': digest(b8), 'new_source_sha256': digest(b9), 'new_snapshot_equals_current_slice': b9 == current, 'python_ast_equal': ast_equal, 'noncode_equal': noncode_equal, 'figure_sha256': {}, 'unified_diff': ''.join(difflib.unified_diff(b8.splitlines(keepends=True), b9.splitlines(keepends=True), fromfile='freeze08/14.13', tofile='freeze09/14.13')), 'old_code_sha256': digest(c8), 'new_code_sha256': digest(c9)},
 'original_report_sha256_before_and_after': sha(old_report_bytes),
 'exact_snippet_stdout': stdout.getvalue(),
 'exact_snippet_raw_values': three,
 'two_steps': two,
 'full_three_steps': full,
 'per_use_contributions_checked_with_distinct_leaf_copies': contributions,
 'last_step_only_backward': truncated,
 'exact_rational_derivation': analytic,
 'result': 'All assertions passed. Only .2 to 0.2 and 0. to 0.0 changed; Python AST, non-code text and figures are unchanged. Exact new snippet and all gradient checks passed at absolute tolerance 1e-6; original report unchanged.',
 'scope': 'One new reviewer-owned standalone Python CPU process, not a notebook/kernel execution09 rerun. No optimizer update, training, GPU or quality/memory benchmark.'
}
output = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/14.13-revision09-callback-execution.json'
with output.open('x') as f:
    f.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False, indent=2))
