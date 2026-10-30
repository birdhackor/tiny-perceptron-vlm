"""單次 create、完整結果先匯出、未知身分不視為停止，以及核心預留。"""

import asyncio
import hashlib
import io
import json
import sys
import zipfile
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

import pytest

from scripts.omni_revision import execution
from scripts.omni_revision import experiment_execution as runner


def storage_fixture():
    return runner.storage_quote(b"<p>Price per GB/month Shared storage (artifacts and GitHub Packages) $0.25 USD</p>")


def test_development_bundle_pins_exact_worker_configuration_and_local_sources():
    config, raw = runner.job_bundle(runner.job_config("dev-av"))
    archive = zipfile.ZipFile(io.BytesIO(raw))
    assert set(archive.namelist()) == {*runner.WORKER_FILES, "job.json"}
    assert hashlib.sha256(archive.read("job.json")).hexdigest() == config["config_sha256"]
    assert json.loads(archive.read("job.json"))["final_opened"] is False
    for name in runner.WORKER_FILES:
        assert archive.read(name) == (runner.ROOT / name).read_bytes()
    proof = runner.cost_proof(execution.RATE_CEILINGS, config, storage_fixture())
    assert proof["reserved_usd"] == "0.65" and proof["resource_wall_bound_seconds"] == 1500
    with pytest.raises(ValueError):
        runner.job_config("formal-final-without-freeze")


def test_profile_keeps_future_core_and_final_bounds_even_when_billing_has_not_arrived():
    following = {
        "reserved_total_usd": "1.45",
        "attempts": [
            {
                "purpose": json.dumps({"job": "dev-av"}),
                "status": "reserved",
                "guard": {"retained_official_increment_usd": "0", "exact_legacy_carry_usd": "4.73"},
            }
        ],
    }
    proof = runner.protect_core(following)
    assert proof["future_bounds_usd"] == "24.10"
    following["reserved_total_usd"] = "5"
    with pytest.raises(ValueError, match="保留"):
        runner.protect_core(following)


@pytest.mark.parametrize(
    "mode",
    ["normal", "worker_failure", "corrupted", "oversized", "lost_owned", "lost_empty", "stop_failure", "list_failure"],
)
def test_owned_runtime_exports_before_termination_and_never_replays_create(tmp_path, mode):
    job, bundle = runner.job_bundle(runner.job_config("dev-av"))
    names = {"app_name": "omni-job-123-1", "tags": {"omni-round": "omni-20261029", "omni-attempt": "job-123-1"}}
    calls, stopped = [], set()
    payload = b"actual fixture zip bytes"
    result = {
        "job": "dev-av",
        "config_sha256": job["config_sha256"],
        "final_opened": False,
        "success": mode != "worker_failure",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if mode == "oversized":
        result["bytes"] = runner.MAX_RESULT_BYTES + 1

    class App:
        def __init__(self, name):
            self.app_id = "ap-owned"

        @asynccontextmanager
        async def run(self, **kwargs):
            yield self

    class Sandbox:
        filesystem = None

        def __init__(self):
            self.filesystem = self

        @property
        def stdout(self):
            async def parts():
                yield "training log\nOMNI_JOB_READY\n"

            return parts()

        async def _get_task_id(self, **kwargs):
            return "ta-owned"

        async def read_bytes(self, path):
            assert "sb-owned" not in stopped
            calls.append(("read", path))
            if path.endswith(".json"):
                return json.dumps(result).encode()
            return payload + (b"corruption" if mode == "corrupted" else b"")

    sandbox = Sandbox()

    async def create(request, **kwargs):
        calls.append(("create", request))
        assert kwargs == {"retry": None, "timeout": 480}
        assert request.definition.timeout_secs == 900
        assert not hasattr(request.definition, "name") and not hasattr(request.definition, "runtime")
        if mode.startswith("lost_"):
            raise TimeoutError("response lost")
        return SimpleNamespace(sandbox_id="sb-owned", metadata=None)

    async def listing(request, **kwargs):
        if mode == "list_failure":
            raise ConnectionError("discovery failed after known ID was obtained")
        rows = (
            []
            if stopped or mode == "lost_empty" or request.before_timestamp
            else [SimpleNamespace(id="sb-owned", created_at=1)]
        )
        return SimpleNamespace(sandboxes=rows)

    async def terminate(request, **kwargs):
        calls.append(("terminate", request.sandbox_id))
        if mode == "stop_failure":
            raise RuntimeError("do not include arbitrary raw exception details")
        stopped.add(request.sandbox_id)

    async def poll(request, **kwargs):
        return SimpleNamespace(result=SimpleNamespace(status=4 if stopped else 0))

    async def image(request, **kwargs):
        return SimpleNamespace(image_id=request.image_id)

    async def close():
        pass

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
    client = SimpleNamespace(
        _stub=SimpleNamespace(
            ImageFromId=image, SandboxCreate=create, SandboxList=listing, SandboxTerminate=terminate, SandboxWait=poll
        ),
        _close=close,
    )
    factory = SimpleNamespace(_new_hydrated=lambda *_: sandbox)
    receipt = asyncio.run(
        runner.run_owned_job(client, api, App, factory, names, tmp_path, [], "resources", "network", job, bundle)
    )
    final = execution.reconcile_stop(
        receipt, [{"app_id": "ap-owned", "description": names["app_name"], "state": "stopped", "tasks": "0"}], names
    )
    assert sum(row[0] == "create" for row in calls) == 1
    assert final["resources_verified"] is (mode not in {"lost_empty", "stop_failure", "list_failure"})
    assert final["success"] is (mode == "normal")
    if mode in {"normal", "worker_failure", "stop_failure", "list_failure"}:
        assert (tmp_path / "result.zip").read_bytes() == payload
        assert final["result_exported_before_stop"]
    else:
        assert not (tmp_path / "result.zip").exists()
    if mode == "list_failure":
        assert stopped == {"sb-owned"}
        assert final["cleanup_checks"]["owned_resource_discovery_completed"] is False


def test_storage_bound_checks_price_retention_and_aggregate_before_upload(tmp_path):
    proof = runner.cost_proof(execution.RATE_CEILINGS, runner.job_config("dev-av"), storage_fixture())
    assert proof["storage"]["artifact_retention_days"] == 7
    assert proof["storage"]["artifact_storage_bound_usd"] == "0.04"
    with pytest.raises(ValueError):
        runner.storage_quote(b"Price per GB/month Shared storage (artifacts and GitHub Packages) $100 USD")
    with pytest.raises(ValueError):
        runner.cost_proof(execution.RATE_CEILINGS, runner.job_config("dev-av"))
    control = tmp_path / "oversized-control.json"
    with control.open("wb") as stream:
        stream.truncate(runner.MAX_CONTROL_BYTES + 1)
    with pytest.raises(ValueError, match="control"):
        runner.check_artifact_limits(tmp_path)
    control.unlink()
    archive = tmp_path / "result.zip"
    with archive.open("wb") as stream:
        stream.truncate(runner.MAX_RESULT_BYTES + 1)
    with pytest.raises(ValueError):
        runner.check_artifact_limits(tmp_path)


@pytest.mark.parametrize("fail_cas", [False, True])
def test_executor_exception_keeps_unknown_paid_state_unresolved_and_cas_rejection_never_executes(
    monkeypatch, tmp_path, fail_cas
):
    events = []
    state = {"origin": {"baseline": {"summaries": [{"cycle": "2026-10"}]}}, "attempts": []}
    attempt = {"attempt_id": "job-123-1", "status": "reserved"}
    following = {**state, "attempts": [attempt]}

    class Store:
        def read(self):
            return following, "blob"

        def write(self, value, blob):
            status = value["attempts"][-1]["status"]
            events.append(status)
            if fail_cas:
                raise RuntimeError("CAS conflict")
            return {"transaction": "fixture"}

    class Sync:
        def create_blocking(self, function):
            def execute(*args):
                events.append("executor_entered")
                raise OSError("receipt write failed after unknown paid call")

            return execute

    modal = ModuleType("modal")
    modal.Workspace = SimpleNamespace(
        from_context=lambda: SimpleNamespace(billing=SimpleNamespace(rates=lambda: execution.RATE_CEILINGS))
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", SimpleNamespace(synchronizer=Sync()))
    monkeypatch.setenv("MODAL_SANDBOX_V2", "0")
    monkeypatch.setattr(sys, "argv", ["executor", "dev-av", "--attempt-id", "job-123-1", "--output", str(tmp_path)])
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "1.6.0")
    monkeypatch.setattr(runner, "checked_sources", lambda: {"revision": "a" * 40})
    monkeypatch.setattr(runner, "GitHubLedger", Store)
    monkeypatch.setattr(runner, "reserve", lambda *args: following)
    monkeypatch.setattr(runner, "protect_core", lambda *args: {})
    monkeypatch.setattr(runner, "official_storage_quote", storage_fixture)
    monkeypatch.setattr(runner.preflight, "load_reconciliation", lambda *args: {})
    monkeypatch.setattr(runner.preflight, "inventory", lambda *args: {})
    monkeypatch.setattr(runner.preflight, "official_snapshot", lambda *args: {})
    monkeypatch.setattr(runner.execution, "read_stopped_apps", lambda *args: [])
    if fail_cas:
        with pytest.raises(RuntimeError, match="CAS conflict"):
            runner.main()
        assert events == ["reserved"]
    else:
        with pytest.raises(SystemExit):
            runner.main()
        assert events == ["reserved", "executor_entered", "unresolved"]
        receipt = json.loads((tmp_path / "final-receipt.json").read_bytes())
        assert receipt["paid_call_state"] == "unknown_after_executor_entry"
        assert receipt["resources_verified"] is False
