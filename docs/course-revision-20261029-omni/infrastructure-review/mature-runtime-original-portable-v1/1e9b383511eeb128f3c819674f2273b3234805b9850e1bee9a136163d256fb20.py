"""真 SDK raw filesystem 的 deadline／取消與輸入拒絕；只 mock container.exec 邊界。"""

import asyncio
import hashlib
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, '/workspace/work/omni-models-v1')
import modal
from modal._utils.async_utils import synchronizer
from modal._utils.sandbox_fs_utils import validate_absolute_remote_path
from modal.sandbox import _Sandbox
from modal_proto import api_pb2
from scripts.omni_revision.execution import hard_wait

HERE = Path(__file__).resolve().parent
calls, cancelled = [], []


async def blocked_exec(self, *args, **kwargs):
    calls.append('exec')
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        cancelled.append('exec')
        raise


async def main():
    sandbox = _Sandbox._new_hydrated('sb-reviewdeadline', SimpleNamespace(), api_pb2.SandboxHandleMetadata(app_id='ap-reviewdeadline'))
    fs = sandbox.filesystem
    outcomes = []
    for name, coroutine in [('write_deadline', fs.write_bytes(data=b'fixed', remote_path='/tmp/review')), ('read_deadline', fs.read_bytes('/tmp/review'))]:
        before, cancels_before = len(calls), len(cancelled)
        try:
            await hard_wait(coroutine, .02)
        except TimeoutError:
            pass
        else:
            raise AssertionError('blocked operation must time out')
        await asyncio.sleep(0)
        assert len(calls) - before == len(cancelled) - cancels_before == 1
        outcomes.append({'case': name, 'timeout': True, 'exec_calls': 1, 'cancel_received_at_mock_exec': True, 'retry_observed': False})
    for name, coroutine, expected in [('non_bytes_data', fs.write_bytes(data='invalid', remote_path='/tmp/review'), 'TypeError'), ('relative_remote_path', fs.write_bytes(data=b'valid', remote_path='relative'), 'InvalidError')]:
        before = len(calls)
        try:
            await hard_wait(coroutine, 1)
        except Exception as error:
            assert type(error).__name__ == expected
        else:
            raise AssertionError('invalid input must fail')
        assert len(calls) == before
        outcomes.append({'case': name, 'error_type': expected, 'exec_calls': 0})
    return outcomes


with patch.object(_Sandbox, 'exec', new=blocked_exec):
    outcomes = synchronizer.create_blocking(main)()
source_path = Path(inspect.getfile(validate_absolute_remote_path))
raw = source_path.read_bytes()
snapshot = HERE / 'dependency-snapshot/_utils/sandbox_fs_utils.py'
snapshot.parent.mkdir(parents=True, exist_ok=True)
snapshot.write_bytes(raw)
result = {'schema': 'omni-mature-runtime-real-sdk-deadline-offline-v1', 'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_development_execution_review', 'modal_version': modal.__version__, 'all_assertions_passed': True, 'cases': outcomes, 'path_validator_source': {'path': str(source_path), 'snapshot_path': str(snapshot), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, 'real_rpc_calls': 0, 'limitations': ['Cancellation is observed at the private mocked exec await point. This does not prove remote RPC cancellation or termination; known Sandbox/App cleanup remains mandatory.']}
(HERE / 'real-sdk-deadline-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'all_assertions_passed': True, 'cases': 4, 'real_rpc_calls': 0}))
