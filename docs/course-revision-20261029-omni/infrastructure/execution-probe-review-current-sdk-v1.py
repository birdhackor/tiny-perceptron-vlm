"""回核用真 Modal 1.6.0 lifecycle／streams／FS；所有 transport 都是離線 fake。"""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["MODAL_ENVIRONMENT"] = "main"
os.environ["MODAL_SANDBOX_V2"] = "0"

from scripts.omni_revision import execution as runner  # noqa: E402
from scripts.omni_revision import finance  # noqa: E402


def forbidden(*args, **kwargs):
    raise AssertionError("real client or network forbidden")


def payloads():
    raw = bytes(range(256)) * 4
    result = {
        "schema": "omni-runtime-probe-v1",
        "python": "3.12.10",
        "packages": {
            "torch": "2.8.0+cu128",
            "transformers": "4.57.6",
            "numpy": "2.2.6",
            "safetensors": "0.7.0",
            "accelerate": "1.12.0",
        },
        "gpu": "NVIDIA L4",
        "gpu_count": 1,
        "gpu_memory_bytes": 24 * 1024**3,
        "finite_identity_matmul": True,
        "model_loaded": False,
        "training_started": False,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {"/tmp/omni-probe.json": json.dumps(result).encode(), "/tmp/omni-probe-payload.bin": raw}


class OfflineRouter:
    """只替代 worker transport，保留 SDK exec／ContainerProcess／StreamReader／FS。"""

    def __init__(self, events):
        self.files, self.processes, self.events = payloads(), {}, events

    async def exec_start(self, request):
        assert request.command_args[0] == "/__modal/.bin/modal-sandbox-fs-tools"
        assert not request.secret_ids and not request.env
        operation = json.loads(request.command_args[1])
        self.processes[request.exec_id] = operation
        self.events.append({"router": "exec_start", "operation": operation})

    async def exec_stdio_read(self, task_id, process_id, file_descriptor, deadline, *, start_offset=0):
        from modal_proto import api_pb2

        operation = self.processes[process_id]
        if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT and "ReadFile" in operation:
            raw = self.files[operation["ReadFile"]["path"]]
            # Deliberately split bytes to exercise SDK stream assembly.
            for fragment in (raw[start_offset : start_offset + 17], raw[max(start_offset, 17) :]):
                if fragment:
                    yield SimpleNamespace(data=fragment)

    async def exec_stdin_write_stream(self, *, task_id, exec_id, source):
        raw = source.read()
        self.files[self.processes[exec_id]["WriteFile"]["path"]] = raw
        self.events.append({"router": "exec_stdin_write_stream", "bytes": len(raw)})
        return len(raw)

    async def exec_wait(self, task_id, process_id, deadline):
        return SimpleNamespace(WhichOneof=lambda field: "code", code=0)


async def run_case(mode, output):
    from google.protobuf.empty_pb2 import Empty
    from grpclib import GRPCError, Status
    from modal._grpc_client import UnaryUnaryWrapper
    from modal._resources import convert_fn_config_to_resources_config
    from modal._utils.async_utils import synchronizer
    from modal.app import _App
    from modal.sandbox import _build_outbound_network_access, _Sandbox
    from modal_proto import api_pb2

    assert asyncio.get_running_loop() is synchronizer._get_loop(start=False)
    app_id, sandbox_id = "ap-aaaaaaaaaaaaaaaaaaaaaa", "sb-aaaaaaaaaaaaaaaaaaaaaa"
    events, calls = [], {}
    router = OfflineRouter(events)
    client = SimpleNamespace(_snapshotted=False)
    stopped = False

    async def rpc(name, request, *, timeout=None, metadata=None):
        nonlocal stopped
        calls[name] = calls.get(name, 0) + 1
        events.append({"rpc": name, "timeout": timeout})
        if name == "ImageFromId":
            return api_pb2.ImageFromIdResponse(image_id=runner.IMAGE_ID)
        if name == "AppCreate":
            assert request.app_state == api_pb2.APP_STATE_EPHEMERAL
            assert request.environment_name == "main"
            return api_pb2.AppCreateResponse(app_id=app_id)
        if name == "EnvironmentGetOrCreate":
            assert request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED
            return api_pb2.EnvironmentGetOrCreateResponse(
                environment_id="en-offline", metadata=api_pb2.EnvironmentMetadata(name="main")
            )
        if name == "AppPublish":
            assert request.app_state == api_pb2.APP_STATE_EPHEMERAL
            return api_pb2.AppPublishResponse()
        if name in {"AppClientDisconnect", "AppHeartbeat"}:
            return Empty()
        if name == "SandboxCreate":
            # No retry metadata means retry=None reached the real UnaryUnaryWrapper.
            assert not metadata
            assert timeout == 180
            definition = request.definition
            assert definition.image_id == runner.IMAGE_ID
            assert definition.timeout_secs == 600
            assert definition.resources.milli_cpu == definition.resources.milli_cpu_max == 2000
            assert definition.resources.memory_mb == definition.resources.memory_mb_max == 8192
            assert definition.resources.gpu_config.count == 1
            assert definition.network_access.network_access_type == api_pb2.NetworkAccess.BLOCKED
            assert definition.runtime == "runc"
            assert not definition.secret_ids and not definition.mount_ids and not definition.volume_mounts
            assert not definition.enable_snapshot and not definition.include_oidc_identity_token
            assert definition.entrypoint_args[-1] == (ROOT / "scripts/omni_revision/probe_worker.py").read_text()
            assert request.app_id == app_id
            if mode.startswith("ambiguous"):
                raise GRPCError(Status.UNAVAILABLE, "offline lost response")
            return api_pb2.SandboxCreateResponse(
                sandbox_id=sandbox_id, metadata=api_pb2.SandboxHandleMetadata(app_id=app_id)
            )
        if name == "SandboxGetTaskId":
            return api_pb2.SandboxGetTaskIdResponse(task_id="ta-offline")
        if name == "SandboxList":
            assert request.app_id == app_id and request.environment_name == "main"
            assert {x.tag_name: x.tag_value for x in request.tags} == runner.identifiers("probe-123-1")["tags"]
            if stopped or mode == "ambiguous_empty" or request.before_timestamp:
                return api_pb2.SandboxListResponse()
            return api_pb2.SandboxListResponse(sandboxes=[api_pb2.SandboxInfo(id=sandbox_id, created_at=1)])
        if name == "SandboxTerminate":
            assert request.sandbox_id == sandbox_id
            stopped = True
            return Empty()
        if name == "SandboxWait":
            assert stopped and request.sandbox_id == sandbox_id
            return api_pb2.SandboxWaitResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
            )
        raise AssertionError("unexpected RPC: " + name)

    def wrap(name):
        wrapper = UnaryUnaryWrapper(SimpleNamespace(name="/modal.client.ModalClient/" + name), client, "offline")

        async def direct(request, *, timeout=None, metadata=None):
            return await rpc(name, request, timeout=timeout, metadata=metadata)

        wrapper.direct = direct
        return wrapper

    async def logs(request):
        assert request.sandbox_id == sandbox_id
        events.append({"rpc": "SandboxGetLogs", "stream": "real SDK async iterator"})
        yield api_pb2.TaskLogsBatch(items=[api_pb2.TaskLogs(data="OMNI_PROBE_READY\n")], eof=True)

    client._stub = SimpleNamespace(
        **{
            name: wrap(name)
            for name in [
                "ImageFromId",
                "AppCreate",
                "EnvironmentGetOrCreate",
                "AppPublish",
                "AppHeartbeat",
                "AppClientDisconnect",
                "SandboxCreate",
                "SandboxGetTaskId",
                "SandboxList",
                "SandboxTerminate",
                "SandboxWait",
            ]
        },
        SandboxGetLogs=SimpleNamespace(unary_stream=logs),
    )

    async def close():
        events.append({"client": "closed"})

    client._close = close

    async def get_router(self, task_id):
        assert self.object_id == sandbox_id and task_id == "ta-offline"
        return router

    with patch.object(_Sandbox, "_get_command_router_client", get_router):
        receipt = await runner.run_owned_probe(
            client,
            api_pb2,
            _App,
            _Sandbox,
            runner.identifiers("probe-123-1"),
            output,
            [],
            convert_fn_config_to_resources_config(cpu=(2, 2), memory=(8192, 8192), gpu="L4"),
            _build_outbound_network_access(True, None, None),
        )
    rows = [{"App ID": app_id, "Description": "omni-probe-123-1", "State": "stopped", "Tasks": "0"}]
    final = runner.reconcile_stop(receipt, rows, runner.identifiers("probe-123-1"))
    terminal = finance.finish({"attempts": [{"attempt_id": "probe-123-1", "status": "reserved"}]}, "probe-123-1", final)
    assert calls["SandboxCreate"] == 1
    assert calls["AppClientDisconnect"] == 1
    assert final["success"] is (mode == "normal"), final
    assert final["resources_verified"] is (mode != "ambiguous_empty"), final
    assert (
        terminal["attempts"][0]["status"]
        == {"normal": "completed", "ambiguous_owned": "failed-stopped", "ambiguous_empty": "unresolved"}[mode]
    )
    if mode == "normal":
        assert (output / "stdout.txt").read_text() == "OMNI_PROBE_READY\n"
        assert len([x for x in events if x.get("router") == "exec_start"]) == 4
    return {
        "mode": mode,
        "calls": calls,
        "events": events,
        "receipt": final,
        "ledger_status": terminal["attempts"][0]["status"],
    }


if __name__ == "__main__":
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client

    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    _Client.__init__ = forbidden
    _Client.from_env = forbidden
    _Client.from_credentials = forbidden
    results = []
    for mode in ["normal", "ambiguous_owned", "ambiguous_empty"]:
        with tempfile.TemporaryDirectory(prefix="omni-current-sdk-review-") as temporary:
            results.append(synchronizer.create_blocking(run_case)(mode, Path(temporary)))
    print(
        json.dumps(
            {
                "modal_version": importlib.metadata.version("modal"),
                "client_constructed": False,
                "network_called": False,
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
