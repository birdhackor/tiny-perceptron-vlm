"""Modal 1.6.0 真 internal _Sandbox / filesystem；只替換 exec/RPC 邊界，禁止網路。"""

import asyncio
import gc
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import socket
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/workspace/work/omni-models-v1')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

import modal
import modal.sandbox_fs as fs_module
from modal._utils import async_utils
from modal._utils.async_utils import synchronizer
from modal.sandbox import _Sandbox
from modal.sandbox_fs import SandboxFilesystem, _SandboxFilesystem
from modal_proto import api_pb2

from scripts.omni_revision.execution import hard_wait
from scripts.omni_revision import staged_execution as staged

assert modal.__version__ == '1.6.0'
PAYLOAD = (Path('/workspace/work/omni-kev-first-preparation-failure-41004743884') / 'input.zip').read_bytes()
assert hashlib.sha256(PAYLOAD).hexdigest() == '8652c3843caeabf93dbb3ce99a24dd06632b27ab1d7db0c45da0004b83540823'
started = datetime.now(UTC).isoformat()
memory, exec_events, network_events, rpc_events = {}, [], [], []


def no_network(*args, **kwargs):
    network_events.append('blocked')
    raise AssertionError('offline review forbids network')


class Reader:
    def __init__(self, value):
        self.value = value

    async def read(self):
        return self.value


class FakeExecProcess:
    def __init__(self, command, binary):
        self.command = command
        self.binary = binary
        self.stderr = Reader(b'' if binary else '')
        path = next(iter(command.values()))['path']
        self.stdout = Reader(memory.get(path, b''))

    async def _stdin_write_stream(self, source):
        content = source.read()
        memory[self.command['WriteFile']['path']] = content
        return len(content)

    async def wait(self):
        return 0


async def exec_boundary(self, binary_path, command, **kwargs):
    assert binary_path == '/__modal/.bin/modal-sandbox-fs-tools'
    parsed = json.loads(command)
    exec_events.append({'command': parsed, 'binary': kwargs.get('text') is False})
    return FakeExecProcess(parsed, kwargs.get('text') is False)


def make_sandbox():
    # 必須在和 production create_blocking(execute) 相同的 SDK synchronizer loop 建立。
    return _Sandbox._new_hydrated('sb-reviewfixture', SimpleNamespace(), api_pb2.SandboxHandleMetadata(app_id='ap-reviewfixture'))


async def api_probe():
    sandbox = make_sandbox()
    fs = sandbox.filesystem
    assert type(fs) is _SandboxFilesystem
    assert inspect.iscoroutinefunction(fs.write_bytes) and inspect.iscoroutinefunction(fs.read_bytes)
    assert not hasattr(fs.write_bytes, 'aio') and not hasattr(fs.read_bytes, 'aio')
    inverse = fs.write_bytes('/tmp/omni-input.zip', PAYLOAD)
    assert inspect.iscoroutine(inverse)
    count = len(exec_events)
    try:
        await hard_wait(inverse, 1)
    except TypeError as error:
        inverse_error = {'type': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError('original inverted write must fail')
    assert len(exec_events) == count and not memory
    # 真 SDK validation + 真 _exec_fs_tool_write + 受控 container.exec 邊界。
    await hard_wait(fs.write_bytes(data=PAYLOAD, remote_path='/tmp/omni-input.zip'), 1)
    marker = hashlib.sha256(PAYLOAD).hexdigest().encode()
    await hard_wait(fs.write_bytes(data=marker, remote_path='/tmp/omni-input.sha256'), 1)
    restored = await hard_wait(fs.read_bytes('/tmp/omni-input.zip'), 1)
    restored_marker = await hard_wait(fs.read_bytes('/tmp/omni-input.sha256'), 1)
    assert restored == PAYLOAD and restored_marker == marker
    raw_success_count = len(exec_events) - count
    # public namespace 只在 .aio 介面執行；不把 public sync 和 internal raw async 混用。
    public = SandboxFilesystem(sandbox)
    assert hasattr(public.write_bytes, 'aio') and hasattr(public.read_bytes, 'aio')
    await hard_wait(public.write_bytes.aio(data=b'public-aio', remote_path='/tmp/public-aio'), 1)
    assert await hard_wait(public.read_bytes.aio('/tmp/public-aio'), 1) == b'public-aio'
    return {'actual_internal_sandbox_class': type(sandbox).__name__, 'actual_filesystem_class': type(fs).__name__, 'internal_write_signature': str(inspect.signature(fs.write_bytes)), 'internal_read_signature': str(inspect.signature(fs.read_bytes)), 'internal_both_return_coroutines': True, 'internal_has_aio_attribute': False, 'public_write_signature': str(inspect.signature(SandboxFilesystem.write_bytes)), 'public_read_signature': str(inspect.signature(SandboxFilesystem.read_bytes)), 'public_methods_are_coroutine_functions': inspect.iscoroutinefunction(SandboxFilesystem.write_bytes), 'public_has_aio': True, 'inverted_original_write': inverse_error, 'inverted_original_write_exec_calls': 0, 'named_data_remote_path_true_sdk_roundtrip': True, 'raw_write2_read2_exec_calls': raw_success_count, 'public_aio_true_sdk_roundtrip': True}


class Stub:
    async def ImageFromId(self, request, **kwargs):
        rpc_events.append({'rpc': 'ImageFromId', 'retry': kwargs.get('retry')})
        return api_pb2.ImageFromIdResponse(image_id=staged.IMAGE_ID)

    async def SandboxCreate(self, request, **kwargs):
        rpc_events.append({'rpc': 'SandboxCreate', 'retry': kwargs.get('retry')})
        return api_pb2.SandboxCreateResponse(sandbox_id='sb-reviewfixture', metadata=api_pb2.SandboxHandleMetadata(app_id='ap-reviewfixture'))

    async def SandboxList(self, request, **kwargs):
        rpc_events.append({'rpc': 'SandboxList', 'retry': kwargs.get('retry')})
        return api_pb2.SandboxListResponse()

    async def SandboxTerminate(self, request, **kwargs):
        rpc_events.append({'rpc': 'SandboxTerminate', 'sandbox_id': request.sandbox_id, 'retry': kwargs.get('retry')})
        return api_pb2.SandboxTerminateResponse()

    async def SandboxWait(self, request, **kwargs):
        rpc_events.append({'rpc': 'SandboxWait', 'sandbox_id': request.sandbox_id, 'retry': kwargs.get('retry')})
        return api_pb2.SandboxWaitResponse(result=api_pb2.GenericResult(status=3))


class App:
    def __init__(self, name):
        self.app_id = 'ap-reviewfixture'

    @asynccontextmanager
    async def run(self, **kwargs):
        yield self


async def task_id_boundary(self, **kwargs):
    return 'ta-reviewfixture'


async def controller_probe():
    async def close():
        rpc_events.append({'rpc': 'client_close'})
    client = SimpleNamespace(_stub=Stub(), _close=close)
    output = HERE / 'controller-current-private-positive'
    output.mkdir(exist_ok=False)
    result_bytes = b'independent private non-neural result bytes'
    job = {'job': 'kev', 'sandbox_seconds': 2100, 'config_sha256': 'private-config', 'core_selection_sha256': 'private-selection', 'core_final_opening_sha256': 'private-opening'}
    result = {**{k:job[k] for k in ('job','config_sha256','core_selection_sha256','core_final_opening_sha256')}, 'success': True, 'final_opened': True, 'bytes': len(result_bytes), 'sha256': hashlib.sha256(result_bytes).hexdigest()}
    memory.update({'/tmp/omni-result.json': json.dumps(result).encode(), '/tmp/omni-result.zip': result_bytes})
    async def stdout(self):
        yield 'OMNI_JOB_READY\n'
    before = len(exec_events)
    with patch.object(_Sandbox, '_get_task_id', new=task_id_boundary), patch.object(_Sandbox, 'stdout', new=property(stdout)):
        receipt = await staged.run_staged_job(client, api_pb2, App, _Sandbox, {'app_name': 'omni-private-current-review', 'tags': {'review': 'private'}}, output, [], None, None, job, PAYLOAD)
    assert receipt['success'] is True and receipt['input_payload_uploaded'] is True and receipt['result_exported_before_stop'] is True
    assert receipt['input_payload_sha256'] == hashlib.sha256(PAYLOAD).hexdigest()
    assert memory['/tmp/omni-input.zip'] == PAYLOAD and memory['/tmp/omni-input.sha256'] == hashlib.sha256(PAYLOAD).hexdigest().encode()
    assert (output/'result.zip').read_bytes() == result_bytes
    events = exec_events[before:]
    assert [next(iter(row['command'])) for row in events] == ['WriteFile','WriteFile','ReadFile','ReadFile']
    assert receipt['stopped_sandboxes'] == {'sb-reviewfixture': 3} and receipt['remaining_owned_sandboxes'] == [] and all(receipt['cleanup_checks'].values())
    assert len([event for event in rpc_events if event['rpc'] == 'SandboxCreate']) == 1
    assert all(event.get('retry') is None for event in rpc_events)
    return {'actual_corrected_controller_with_real_internal_sdk_filesystem': True, 'positive': True, 'phase': receipt['phase'], 'filesystem_operations': events, 'payload_marker_and_result_bytes_verified': True, 'export_before_stop': True, 'single_simulated_create': True, 'simulated_known_id_terminated': True, 'rpc_events': rpc_events, 'private_fake_worker_result_not_real_nn': True}


async def main():
    api = await api_probe()
    control = await controller_probe()
    return api, control


original_staged_hard_wait = staged.hard_wait
gc_stress_calls = []
async def gc_stressed_hard_wait(coroutine, seconds):
    gc.collect()
    gc_stress_calls.append(True)
    return await original_staged_hard_wait(coroutine, seconds)

with patch.object(staged, 'hard_wait', new=gc_stressed_hard_wait), patch.object(_Sandbox, 'exec', new=exec_boundary), patch.object(socket.socket, 'connect', new=no_network), patch.object(socket, 'create_connection', new=no_network):
    api_result, controller_result = synchronizer.create_blocking(main)()
assert not network_events
dependencies = {}
for name in ['modal', 'synchronicity', 'grpclib', 'protobuf', 'aiohttp', 'certifi', 'typing_extensions']:
    dependencies[name] = importlib.metadata.version(name)
source_files = [Path(modal.__file__), Path(fs_module.__file__), Path(inspect.getfile(_Sandbox)), Path(inspect.getfile(async_utils)), Path(inspect.getfile(hard_wait)), ROOT / 'scripts/omni_revision/staged_execution.py', ROOT / 'tests/test_omni_mature_research.py']
source_manifest = []
for path in source_files:
    raw = path.read_bytes()
    source_manifest.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    if 'site-packages' in path.parts:
        target = HERE / 'dependency-snapshot' / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
result = {'schema': 'omni-mature-runtime-gc-stressed-production-original-v1', 'gc_collect_before_every_staged_hard_wait': True, 'gc_stress_calls': len(gc_stress_calls), 'reviewer_task': '/root/omni_development_execution_review', 'started_at': started, 'finished_at': datetime.now(UTC).isoformat(), 'all_assertions_passed': True, 'python': platform.python_version(), 'sdk': dependencies, 'production_matching_loop_entry': 'modal._utils.async_utils.synchronizer.create_blocking(async_function)', 'interface_and_roundtrip': api_result, 'current_controller_positive': controller_result, 'network_connect_attempts': network_events, 'real_modal_resource_create_calls': 0, 'real_rpc_calls': 0, 'nn_predictions': 0, 'source_manifest': source_manifest, 'limitations': ['Exec/RPC responses and App context are private offline fixtures, not provider observations.', 'The real installed Modal1.6.0 class/property/signatures/path validation/write/read bodies and the corrected reviewed run_staged_job/hard_wait execute without filesystem-method stubs.', 'No actual stopped-sandbox filesystem or provider-side neural execution history was read.']}
(HERE / 'gc-production-probe-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'all_assertions_passed': True, 'modal': modal.__version__, 'current_controller_phase': controller_result['phase'], 'original_error': api_result['inverted_original_write'], 'current_fs_operations': len(controller_result['filesystem_operations']), 'gc_stress_calls': len(gc_stress_calls), 'corrected_named_args_roundtrip': True, 'raw_read_bytes_awaitable': True, 'real_rpc_calls': 0}, ensure_ascii=False))
