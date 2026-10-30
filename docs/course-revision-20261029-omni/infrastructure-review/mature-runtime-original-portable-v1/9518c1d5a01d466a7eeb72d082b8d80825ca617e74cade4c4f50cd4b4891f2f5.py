"""真 bootstrap 的私有空 FS／缺 marker／錯 marker／完整 marker 控制；runpy 為記錄用 stub，沒有 NN。"""

import ast
import hashlib
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/workspace/work/omni-models-v1')
sys.path.insert(0, str(ROOT))
from scripts.omni_revision.staged_execution import staged_bootstrap

HERE = Path(__file__).resolve().parent
payload = (Path('/workspace/work/omni-kev-first-preparation-failure-41004743884') / 'input.zip').read_bytes()
expected = hashlib.sha256(payload).hexdigest()
code = staged_bootstrap(payload)
base = HERE / 'bootstrap-private-fixtures'
base.mkdir(exist_ok=False)
results = []
for mode in ('empty', 'payload_without_marker', 'wrong_marker', 'complete_identity'):
    folder = base / mode
    folder.mkdir()
    if mode != 'empty':
        (folder / 'input.zip').write_bytes(payload)
    if mode in ('wrong_marker', 'complete_identity'):
        (folder / 'marker.sha256').write_text('wrong' if mode == 'wrong_marker' else expected)

    class LocalPathsAndShortTimeout(ast.NodeTransformer):
        def visit_Constant(self, node):
            replacements = {'/tmp/omni-input.zip': str(folder / 'input.zip'), '/tmp/omni-input.sha256': str(folder / 'marker.sha256'), '/tmp/omni-job': str(folder / 'job')}
            if isinstance(node.value, str) and node.value in replacements:
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            if node.value == 150:
                return ast.copy_location(ast.Constant(.02), node)
            if node.value == .1:
                return ast.copy_location(ast.Constant(.001), node)
            return node

    parsed = ast.fix_missing_locations(LocalPathsAndShortTimeout().visit(ast.parse(code)))
    calls = []

    def capture_module(name, **kwargs):
        calls.append({'module': name, **kwargs})

    before = Path.cwd()
    sys_path_before = sys.path[:]
    error = None
    try:
        with patch.object(runpy, 'run_module', side_effect=capture_module):
            exec(compile(parsed, '<private-exact-bootstrap-with-local-paths-and-short-timeout>', 'exec'), {})
    except (TimeoutError, ValueError) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        import os
        os.chdir(before)
        sys.path[:] = sys_path_before
    if mode in ('empty', 'payload_without_marker'):
        assert error['type'] == 'TimeoutError' and not calls
    elif mode == 'wrong_marker':
        assert error['type'] == 'ValueError' and not calls
    else:
        assert error is None and calls == [{'module': 'scripts.omni_revision.research_worker', 'run_name': '__main__'}]
    results.append({'mode': mode, 'error': error, 'recorded_worker_module_invocations': calls, 'real_worker_invocations': 0})
out = {'schema': 'omni-mature-runtime-bootstrap-marker-offline-probe-v1', 'reviewer_task': '/root/omni_development_execution_review', 'created_at': datetime.now(UTC).isoformat(), 'all_assertions_passed': True, 'actual_payload_sha256': expected, 'actual_generated_bootstrap_sha256': hashlib.sha256(code.encode()).hexdigest(), 'modifications_in_private_fixture_only': ['three /tmp paths replaced by private unique local paths', '150s wait shortened to .02s and .1s polling shortened to .001s', 'runpy.run_module records instead of executing any worker'], 'cases': results, 'evidentiary_limit': 'These cases verify the source gate with local fixtures. They are not direct observations of the failed provider sandbox filesystem or neural execution history.'}
(HERE / 'bootstrap-marker-probe-results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'all_assertions_passed': True, 'cases': len(results), 'real_worker_invocations': 0, 'nn_predictions': 0}))
