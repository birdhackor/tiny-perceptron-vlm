"""付費邊界、模糊 create 與清理失敗；所有測試都離線。"""

import asyncio
import hashlib
import json
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts.omni_revision import execution as runner


def valid_result():
    return {
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
        "payload_sha256": hashlib.sha256(bytes(range(256)) * 4).hexdigest(),
    }


def test_bound_uses_sandbox_rates_and_includes_creation_and_cleanup():
    proof = runner.cost_proof(runner.RATE_CEILINGS)
    assert proof["hourly_resource_bound_usd"] == "1.2758"
    assert proof["resource_wall_bound_seconds"] == 900
    assert proof["reserved_usd"] == "0.40"
    # Function CPU/RAM 費率不足以作 Sandbox 的上界；多餘舊欄也不會覆蓋。
    rates = {**runner.RATE_CEILINGS, "cpu_hour_cost": "0.00001", "mem_gib_hour_cost": "0.00001"}
    assert runner.cost_proof(rates)["hourly_resource_bound_usd"] == "1.2758"


@pytest.mark.parametrize("bad", ["missing", "increased", "nan"])
def test_missing_increased_or_nonfinite_rates_block_before_dispatch(bad):
    rates = dict(runner.RATE_CEILINGS)
    if bad == "missing":
        rates.pop("cpu_hour_cost_sandbox")
    else:
        rates["cpu_hour_cost_sandbox"] = "0.15" if bad == "increased" else "NaN"
    with pytest.raises(ValueError):
        runner.cost_proof(rates)


@pytest.mark.parametrize("age", [-1, 121])
def test_stale_or_future_snapshot_rejected_at_actual_paid_boundary(age):
    with pytest.raises(ValueError):
        runner.check_fresh({"observed_at": (datetime.now(UTC) - timedelta(seconds=age)).isoformat()})


class App:
    def __init__(self, name):
        self.app_id = "ap-owned"

    @asynccontextmanager
    async def run(self, **kwargs):
        assert kwargs["detach"] is False
        assert kwargs["environment_name"] == "main"
        yield self


class Sandbox:
    def __init__(self, mode):
        self.mode = mode
        self.object_id = "sb-owned"
        self.stopped = False
        self.terminate_count = 0
        self.filesystem = self
        self.challenge = None

    @property
    def stdout(self):
        async def parts():
            yield "OMNI_PROBE_"
            yield "READY\n"

        return parts()

    async def read_bytes(self, path):
        if path.endswith("omni-probe.json"):
            result = valid_result()
            if self.mode == "wrong_gpu":
                result["gpu"] = "NVIDIA H100"
            return json.dumps(result).encode()
        if path.endswith("payload.bin"):
            return bytes(range(256)) * 4
        return b"wrong" if self.mode == "bad_roundtrip" else self.challenge

    async def write_bytes(self, raw, path):
        self.challenge = raw

    async def _get_task_id(self, **kwargs):
        return "ta-owned"


def run_case(tmp_path, mode):
    sandbox = Sandbox(mode)
    creates, lookups = [], []

    async def create(request, **kwargs):
        creates.append((request, kwargs))
        assert kwargs == {"retry": None, "timeout": 180}
        assert request.definition.image_id == runner.IMAGE_ID
        assert request.definition.timeout_secs == 600
        assert not hasattr(request.definition, "env")
        assert not hasattr(request.definition, "name")
        assert not hasattr(request.definition, "runtime")
        if mode in {"ambiguous_create", "ambiguous_empty"}:
            raise TimeoutError("response lost after container create")
        return SimpleNamespace(sandbox_id=sandbox.object_id, metadata=None)

    async def listing(request, **kwargs):
        lookups.append(request)
        assert request.app_id == "ap-owned"
        assert {x.tag_name: x.tag_value for x in request.tags} == runner.identifiers("probe-123-1")["tags"]
        rows = (
            []
            if sandbox.stopped or mode == "ambiguous_empty"
            else [SimpleNamespace(id=sandbox.object_id, created_at=1)]
        )
        # 模擬既有 SDK 的 before_timestamp 分頁。
        return SimpleNamespace(sandboxes=[] if request.before_timestamp else rows)

    async def terminate(request, **kwargs):
        assert request.sandbox_id == sandbox.object_id
        sandbox.terminate_count += 1
        if mode == "cleanup_failure":
            raise RuntimeError("fixture token-like secret should not be copied to receipt")
        sandbox.stopped = True

    async def poll(request, **kwargs):
        assert request.sandbox_id == sandbox.object_id
        return SimpleNamespace(result=SimpleNamespace(status=4 if sandbox.stopped else 0))

    async def image_from_id(request, **kwargs):
        return SimpleNamespace(image_id=request.image_id)

    async def close():
        return None

    client = SimpleNamespace(
        _stub=SimpleNamespace(
            ImageFromId=image_from_id,
            SandboxCreate=create,
            SandboxList=listing,
            SandboxTerminate=terminate,
            SandboxWait=poll,
        ),
        _close=close,
    )
    factory = SimpleNamespace(_new_hydrated=lambda identifier, client, metadata: sandbox)
    api = SimpleNamespace(
        **{
            name: SimpleNamespace
            for name in (
                "ImageFromIdRequest",
                "SandboxTag",
                "Sandbox",
                "SandboxCreateRequest",
                "SandboxListRequest",
                "SandboxTerminateRequest",
                "SandboxWaitRequest",
            )
        }
    )
    receipt = asyncio.run(
        runner.run_owned_probe(
            client,
            api,
            App,
            factory,
            runner.identifiers("probe-123-1"),
            tmp_path,
            [],
            "fixture-resource",
            "fixture-network",
        )
    )
    return receipt, sandbox, creates, lookups


@pytest.mark.parametrize(
    "mode", ["normal", "wrong_gpu", "bad_roundtrip", "ambiguous_create", "ambiguous_empty", "cleanup_failure"]
)
def test_single_create_exports_before_stop_and_ambiguity_never_replays(tmp_path, mode):
    receipt, sandbox, creates, lookups = run_case(tmp_path, mode)
    assert len(creates) == 1
    assert lookups and sandbox.terminate_count == (0 if mode == "ambiguous_empty" else 1)
    assert receipt["success"] is (mode == "normal")
    assert sandbox.stopped is (mode not in {"cleanup_failure", "ambiguous_empty"})
    rows = [{"app_id": "ap-owned", "description": "omni-probe-123-1", "state": "stopped", "tasks": "0"}]
    final = runner.reconcile_stop(receipt, rows, runner.identifiers("probe-123-1"))
    assert final["resources_verified"] is (mode not in {"cleanup_failure", "ambiguous_empty"})
    if mode == "cleanup_failure":
        assert "token-like" not in json.dumps(receipt)
    if mode == "normal":
        assert runner.check_probe((tmp_path / "probe-result.json").read_bytes())["finite_identity_matmul"]


@pytest.mark.parametrize("rows", [[], [{"app_id": "ap-owned", "state": "running", "tasks": "1"}]])
def test_missing_or_running_app_is_not_verified_cleanup(tmp_path, rows):
    receipt, *_ = run_case(tmp_path, "normal")
    final = runner.reconcile_stop(receipt, rows, runner.identifiers("probe-123-1"))
    assert final["resources_verified"] is False


def test_failed_reservation_cas_never_enters_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("MODAL_SANDBOX_V2", "0")
    monkeypatch.setattr(
        sys, "argv", ["execution", "runtime-probe", "--attempt-id", "probe-123-1", "--output", str(tmp_path)]
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "1.6.0")
    workspace = SimpleNamespace(billing=SimpleNamespace(rates=lambda: runner.RATE_CEILINGS))
    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(Workspace=SimpleNamespace(from_context=lambda: workspace))
    )
    monkeypatch.setattr(runner, "checked_sources", lambda: {"revision": "a" * 40})
    monkeypatch.setattr(runner.preflight, "inventory", lambda *args: {})
    monkeypatch.setattr(runner.preflight, "load_reconciliation", lambda *args: {})
    monkeypatch.setattr(runner.preflight, "official_snapshot", lambda *args: {})
    monkeypatch.setattr(runner, "reserve", lambda *args: {"attempts": [{"attempt_id": "probe-123-1"}]})

    class FailedCAS:
        def read(self):
            return {"origin": {"baseline": {"summaries": [{"cycle": "2026-10"}]}}}, "old-blob"

        def write(self, value, old_blob):
            raise RuntimeError("CAS conflict")

    monkeypatch.setattr(runner, "GitHubLedger", FailedCAS)
    monkeypatch.setattr(runner, "execute_async", lambda *args: pytest.fail("CAS 未成功卻呼叫 paid runner"))
    with pytest.raises(RuntimeError, match="CAS conflict"):
        runner.main()


@pytest.mark.parametrize("late_cleanup", [False, True])
def test_hard_wait_does_not_wait_for_a_cancellation_shield(late_cleanup):
    import time

    async def main():
        async def shielded():
            try:
                await asyncio.Event().wait()
            finally:
                if late_cleanup:
                    await asyncio.sleep(0.15)

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await runner.hard_wait(shielded(), 0.01)
        assert time.monotonic() - started < 0.1

    asyncio.run(main())


def test_error_redaction_precedes_truncation_and_longer_overlapping_values(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-prefix")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-prefix-long-secret")
    monkeypatch.setenv("GH_TOKEN", "test-cross-boundary-token")
    error = SimpleNamespace(_grpc_message="test-prefix-long-secret " + "x" * 3960 + "test-cross-boundary-token")
    detail = runner.redacted_create_error(error)["redacted_server_detail"]
    assert "long-secret" not in detail and "test-cross" not in detail and "test-prefix" not in detail
    assert len(detail) <= 4000
