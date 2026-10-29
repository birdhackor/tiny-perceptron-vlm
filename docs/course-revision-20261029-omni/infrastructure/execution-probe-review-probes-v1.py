"""獨立離線 SDK 探針；只用 fake stubs，不建立 Client 或派工。"""

import asyncio
import importlib.metadata
import json
import socket
import time
from types import SimpleNamespace


def forbidden(*args, **kwargs):
    raise AssertionError("network or real Modal client construction forbidden")


async def main(native=False):
    from grpclib import GRPCError, Status
    from modal._grpc_client import UnaryUnaryWrapper
    from modal.client import _Client
    from modal.image import _Image
    from modal.sandbox import _Sandbox
    from modal_proto import api_pb2

    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    _Client.__init__ = forbidden
    _Client.from_env = forbidden
    _Client.from_credentials = forbidden
    events = []

    async def secret(request):
        events.append({"rpc": "SecretGetOrCreate", "creation_type": request.object_creation_type})
        return api_pb2.SecretGetOrCreateResponse(secret_id="st-offline")

    async def create(request):
        events.append(
            {
                "rpc": "SandboxCreate",
                "secret_ids": list(request.definition.secret_ids),
                "timeout_secs": request.definition.timeout_secs,
                "cpu_request": request.definition.resources.milli_cpu,
                "cpu_limit": request.definition.resources.milli_cpu_max,
                "memory_request_mib": request.definition.resources.memory_mb,
                "memory_limit_mib": request.definition.resources.memory_mb_max,
                "gpu_count": request.definition.resources.gpu_config.count,
                "tags": {x.tag_name: x.tag_value for x in request.tags},
            }
        )
        return api_pb2.SandboxCreateResponse(sandbox_id="sb-aaaaaaaaaaaaaaaaaaaaaa")

    async def task_id(request):
        return api_pb2.SandboxGetTaskIdResponse(task_id="ta-offline")

    async def terminate(request):
        events.append({"rpc": "SandboxTerminate", "delay_seconds": 0.08})
        await asyncio.sleep(0.08)
        return api_pb2.SandboxTerminateResponse()

    stub = SimpleNamespace(
        SecretGetOrCreate=secret, SandboxCreate=create, SandboxGetTaskId=task_id, SandboxTerminate=terminate
    )
    client = SimpleNamespace(_stub=stub, _snapshotted=False)
    app = SimpleNamespace(app_id="ap-offline", _client=client)
    image = _Image._new_hydrated("im-offline", client, None)
    try:
        await _Sandbox.create(
            "python", "-c", "pass", app=app, image=image, client=client,
            name="omni-probe-123-1", tags={"omni-attempt": "probe-123-1"},
            gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=600,
            block_network=True, env={"OMP_NUM_THREADS": "2"},
        )
    except AssertionError as error:
        if not native:
            raise
        import traceback
        return {"error_type": type(error).__name__, "traceback": traceback.format_exc(), "events": events}
    if native:
        raise AssertionError("Expected raw-native-loop hydration assertion was not reproduced")
    assert [x["rpc"] for x in events] == ["SecretGetOrCreate", "SandboxCreate"]
    env_events = list(events)
    events.clear()

    async def never_task_id(request):
        await asyncio.Event().wait()

    stub.SandboxGetTaskId = never_task_id
    started = time.monotonic()
    try:
        await asyncio.wait_for(
            _Sandbox.create("python", "-c", "pass", app=app, image=image, client=client, gpu="L4"), 0.01
        )
    except TimeoutError:
        pass
    elapsed = time.monotonic() - started
    assert elapsed >= 0.075, "SDK shielding must remain visible in this original probe"
    cancellation_events = list(events)

    wrapped = UnaryUnaryWrapper(SimpleNamespace(name="/modal.client.ModalClient/SandboxCreate"), client, "offline")
    retry_events = []

    async def direct(request, *, timeout=None, metadata=None):
        retry_events.append({"timeout": timeout, "metadata": dict(metadata or [])})
        if len(retry_events) == 1:
            raise GRPCError(Status.UNAVAILABLE, "offline response-lost fixture")
        return api_pb2.SandboxCreateResponse(sandbox_id="sb-aaaaaaaaaaaaaaaaaaaaaa")

    wrapped.direct = direct
    await wrapped(api_pb2.SandboxCreateRequest())
    assert len(retry_events) == 2
    assert retry_events[0]["metadata"]["x-idempotency-key"] == retry_events[1]["metadata"]["x-idempotency-key"]
    for item in retry_events:
        item["metadata"].pop("x-idempotency-key")
        item["metadata"].pop("x-modal-timestamp")
    return {
        "modal_version": importlib.metadata.version("modal"),
        "real_client_constructed": False,
        "network_called": False,
        "env_creates_anonymous_secret": env_events,
        "wait_for_timeout_seconds": 0.01,
        "wait_for_elapsed_seconds": elapsed,
        "cancellation_events": cancellation_events,
        "default_create_retry": retry_events,
        "same_idempotency_key_preserved": True,
    }


if __name__ == "__main__":
    native = asyncio.run(main(native=True))
    from modal._utils.async_utils import synchronizer
    synchronized = synchronizer.create_blocking(main)()
    print(json.dumps({"native_loop": native, "sdk_synchronizer_loop": synchronized}, ensure_ascii=False, indent=2))
