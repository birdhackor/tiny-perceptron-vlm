"""真Modal1.6.0 internal filesystem契約；只替換exec與RPC界線，不連服務。"""

import asyncio
import gc
import hashlib
import importlib.metadata
import json
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from scripts.omni_revision import staged_execution as staged
from scripts.omni_revision.execution import hard_wait

pytest.importorskip("modal", reason="真SDK契約另由manual workflow以modal==1.6.0執行")
synchronizer = importlib.import_module("modal._utils.async_utils").synchronizer
_Sandbox = importlib.import_module("modal.sandbox")._Sandbox
_SandboxFilesystem = importlib.import_module("modal.sandbox_fs")._SandboxFilesystem
api_pb2 = importlib.import_module("modal_proto.api_pb2")


@pytest.fixture
def real_sdk(monkeypatch):
    assert importlib.metadata.version("modal") == "1.6.0"
    memory, events = {}, []

    def no_network(*args, **kwargs):
        raise AssertionError("SDK契約測試禁止真正網路／付費RPC")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    class Reader:
        def __init__(self, value):
            self.value = value

        async def read(self):
            return self.value

    class Process:
        def __init__(self, command, binary):
            self.command = command
            self.stderr = Reader(b"" if binary else "")
            self.stdout = Reader(memory.get(next(iter(command.values()))["path"], b""))

        async def _stdin_write_stream(self, source):
            raw = source.read()
            memory[self.command["WriteFile"]["path"]] = raw
            return len(raw)

        async def wait(self):
            return 0

    async def exec_boundary(self, binary, command, **kwargs):
        assert binary == "/__modal/.bin/modal-sandbox-fs-tools"
        value = json.loads(command)
        events.append(value)
        return Process(value, kwargs.get("text") is False)

    monkeypatch.setattr(_Sandbox, "exec", exec_boundary)

    def make():
        value = _Sandbox._new_hydrated(
            "sb-fixture", SimpleNamespace(), api_pb2.SandboxHandleMetadata(app_id="ap-fixture")
        )
        assert type(value.filesystem) is _SandboxFilesystem
        return value

    return memory, events, make


def test_real_sdk_rejects_original_inverse_arguments_before_exec(real_sdk):
    memory, events, make = real_sdk

    async def probe():
        sandbox = make()
        fs = sandbox.filesystem
        # 真filesystem只弱參照sandbox；測試與production一樣保留擁有者跨越await。
        gc.collect()
        with pytest.raises(TypeError):
            await hard_wait(fs.write_bytes("/tmp/input.zip", b"fixture"), 1)
        assert sandbox.object_id == "sb-fixture"

    synchronizer.create_blocking(probe)()
    assert events == [] and memory == {}


def test_actual_controller_uses_real_internal_write_and_read_contract(tmp_path, monkeypatch, real_sdk):
    memory, events, _ = real_sdk
    payload, result_bytes = b"fixed first-party input", b"fixed first-party output"
    job = {
        "job": "kev",
        "sandbox_seconds": 2100,
        "config_sha256": "fixture-config",
        "core_selection_sha256": "fixture-selection",
        "core_final_opening_sha256": "fixture-opening",
    }
    result = {
        **{k: job[k] for k in ("job", "config_sha256", "core_selection_sha256", "core_final_opening_sha256")},
        "success": True,
        "final_opened": True,
        "bytes": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    memory.update({"/tmp/omni-result.json": json.dumps(result).encode(), "/tmp/omni-result.zip": result_bytes})
    calls = []

    class App:
        def __init__(self, name):
            self.app_id = "ap-fixture"

        @asynccontextmanager
        async def run(self, **kwargs):
            yield self

    class Stub:
        async def ImageFromId(self, request, **kwargs):
            return api_pb2.ImageFromIdResponse(image_id=staged.IMAGE_ID)

        async def SandboxCreate(self, request, **kwargs):
            assert kwargs == {"retry": None, "timeout": 480}
            calls.append("create")
            return api_pb2.SandboxCreateResponse(
                sandbox_id="sb-fixture", metadata=api_pb2.SandboxHandleMetadata(app_id="ap-fixture")
            )

        async def SandboxList(self, request, **kwargs):
            return api_pb2.SandboxListResponse()

        async def SandboxTerminate(self, request, **kwargs):
            assert request.sandbox_id == "sb-fixture"
            calls.append("terminate")
            return api_pb2.SandboxTerminateResponse()

        async def SandboxWait(self, request, **kwargs):
            return api_pb2.SandboxWaitResponse(result=api_pb2.GenericResult(status=3))

    async def task_id(self, **kwargs):
        return "ta-fixture"

    async def stdout(self):
        yield "OMNI_JOB_READY\n"

    async def close():
        calls.append("close")

    monkeypatch.setattr(_Sandbox, "_get_task_id", task_id)
    monkeypatch.setattr(_Sandbox, "stdout", property(stdout))

    async def probe():
        return await staged.run_staged_job(
            SimpleNamespace(_stub=Stub(), _close=close),
            api_pb2,
            App,
            _Sandbox,
            {"app_name": "first-party-fixture", "tags": {"fixture": "no-network"}},
            tmp_path,
            [],
            None,
            None,
            job,
            payload,
        )

    receipt = synchronizer.create_blocking(probe)()
    assert receipt["success"] and receipt["input_payload_uploaded"] and receipt["result_exported_before_stop"]
    assert memory["/tmp/omni-input.zip"] == payload
    assert memory["/tmp/omni-input.sha256"] == hashlib.sha256(payload).hexdigest().encode()
    assert (tmp_path / "result.zip").read_bytes() == result_bytes
    assert [next(iter(row)) for row in events] == ["WriteFile", "WriteFile", "ReadFile", "ReadFile"]
    assert calls == ["create", "terminate", "close"]
    assert receipt["stopped_sandboxes"] == {"sb-fixture": 3} and all(receipt["cleanup_checks"].values())


@pytest.mark.parametrize("operation", ["write", "read"])
def test_real_sdk_filesystem_timeout_cancels_before_fake_exec_completes(monkeypatch, real_sdk, operation):
    _, _, make = real_sdk
    cancelled = []

    async def blocked_exec(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(_Sandbox, "exec", blocked_exec)

    async def probe():
        sandbox = make()
        fs = sandbox.filesystem
        gc.collect()
        pending = (
            fs.write_bytes(data=b"fixture", remote_path="/tmp/input")
            if operation == "write"
            else fs.read_bytes("/tmp/input")
        )
        with pytest.raises(TimeoutError):
            await hard_wait(pending, 0.01)
        assert sandbox.object_id == "sb-fixture"

    synchronizer.create_blocking(probe)()
    assert cancelled == [True]
