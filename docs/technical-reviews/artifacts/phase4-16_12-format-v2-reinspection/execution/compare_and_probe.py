"""Same-owner bounded reinspection of the current original fence's semantics."""
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
prior_base = base.parent / 'phase4-16_12-independent'
old_code = (prior_base / 'execution/fence-1.py').read_bytes()
new_code = (base / 'execution/fence-1.py').read_bytes()
old_tree = ast.parse(old_code)
new_tree = ast.parse(new_code)
assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
assert [line for line in old_code.splitlines() if line.strip()] == [line for line in new_code.splitlines() if line.strip()]

original_output = io.StringIO()
namespace = {'__name__': '__main__'}
with contextlib.redirect_stdout(original_output):
    exec(compile(new_code, '16.12-current-original-fence', 'exec'), namespace)
assert original_output.getvalue().encode() == (base / 'execution/stdout.txt').read_bytes()
assert original_output.getvalue().encode() == (prior_base / 'execution/stdout.txt').read_bytes()

namespace['window'] = 2
visible = namespace['visible']
direct = sorted(visible(7))
two_layers = sorted(set().union(*(visible(p) for p in visible(7))))
assert direct == [6, 7]
assert two_layers == [5, 6, 7]
assert visible(0) == {0}
assert sum(len(visible(p)) for p in range(8)) == 15

print(json.dumps({
    'old_code_sha256': hashlib.sha256(old_code).hexdigest(),
    'current_code_sha256': hashlib.sha256(new_code).hexdigest(),
    'ast_equal_ignoring_location_attributes': True,
    'nonblank_lines_byte_equal': True,
    'added_blank_lines': len(new_code.splitlines()) - len(old_code.splitlines()),
    'current_ast': ast.dump(new_tree, include_attributes=False),
    'current_ast_nodes': [{'type': type(n).__name__, 'name': getattr(n, 'name', None), 'first_line': n.lineno, 'last_line': n.end_lineno} for n in new_tree.body],
    'original_stdout_byte_equal_to_prior_and_current_helper': True,
    'current_original_stdout': original_output.getvalue(),
    'current_fence_window2_probe': {'direct7': direct, 'two_layers7': two_layers, 'visible0': sorted(visible(0)), 'pair_count_length8': 15},
    'scope': 'Original current code execution, AST and blank-line change verification, and one bounded exercise/boundary probe. Prior large sweeps and model measurements were not rerun.',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu', 'dependencies': 'stdlib only'},
    'actual_probe_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2))
