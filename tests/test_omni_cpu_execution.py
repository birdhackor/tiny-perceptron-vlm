"""標準CPU父控制端的子process停止、單次scope及封包邊界。"""

import json
import subprocess
import sys

import pytest

from scripts.omni_revision import cpu_execution as cpu


def test_actual_child_has_no_github_or_modal_tokens_and_exit_is_observed(tmp_path, monkeypatch):
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.setenv(name, "test-only-placeholder")
    state = {"all_training_processes_exited": True}
    code = "import os; assert all(os.environ.get(k) is None for k in ('GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')); print('child-safe')"
    cpu.bounded_child(["-c", code], tmp_path / "child.log", 10, state)
    assert state["all_training_processes_exited"]
    assert (tmp_path / "child.log").read_text().strip() == "child-safe"


def test_actual_deadline_kills_and_observes_child_before_reporting_stopped(tmp_path):
    state = {"all_training_processes_exited": True}
    with pytest.raises(TimeoutError):
        cpu.bounded_child(["-c", "import time; time.sleep(5)"], tmp_path / "deadline.log", 0.01, state)
    assert state["all_training_processes_exited"] is True


def test_child_wait_failure_cannot_claim_exited(tmp_path, monkeypatch):
    class Unknown:
        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, **kwargs):
            raise RuntimeError("unknown child state")

    monkeypatch.setattr(cpu.subprocess, "Popen", lambda *args, **kwargs: Unknown())
    state = {"all_training_processes_exited": True}
    with pytest.raises(RuntimeError, match="unknown"):
        cpu.bounded_child(["-c", "unused"], tmp_path / "unknown.log", 0, state)
    assert state["all_training_processes_exited"] is False


def test_real_child_created_before_interrupted_popen_return_remains_unknown(tmp_path, monkeypatch):
    original = cpu.subprocess.Popen
    held = []

    def interrupted(*args, **kwargs):
        held.append(
            original(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(cpu.subprocess, "Popen", interrupted)
    state = {"all_training_processes_exited": True}
    try:
        with pytest.raises(KeyboardInterrupt):
            cpu.bounded_child(["-c", "unused"], tmp_path / "interrupted.log", 1, state)
        assert held[0].poll() is None and state["all_training_processes_exited"] is False
    finally:
        held[0].kill()
        held[0].wait(timeout=10)


def test_core_attempt_is_durably_single_use_even_after_failure(tmp_path, monkeypatch):
    class Ledger:
        def read(self):
            return {
                "attempts": [{"purpose": json.dumps({"job": "cpu-core-formal"}), "status": "failed-stopped"}]
            }, "blob"

    monkeypatch.setattr(cpu, "identity", lambda *args: None)
    monkeypatch.setattr(cpu, "checked_sources", lambda: {"revision": "1" * 40})
    monkeypatch.setattr(cpu, "FROZEN", tmp_path / "frozen")
    monkeypatch.setattr(cpu, "GitHubLedger", Ledger)
    monkeypatch.setattr(cpu, "pricing", lambda: pytest.fail("no new paid boundary"))
    with pytest.raises(ValueError, match="已嘗試"):
        cpu.reserve_cpu("new-attempt", tmp_path / "control")


def test_zip_and_control_bounds_apply_before_artifact_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "MAX_PACKAGE", 5)
    (tmp_path / "model.zip").write_bytes(b"123456")
    with pytest.raises(ValueError):
        cpu.artifact_bound(tmp_path)


def test_owned_asset_is_not_overwritten_and_unknown_owner_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "model.zip"
    path.write_bytes(b"data")
    monkeypatch.setattr(cpu, "command", lambda *args: pytest.fail("must not upload"))
    monkeypatch.setattr(
        cpu, "api", lambda *args: {"draft": True, "tag_name": "tag", "body": "wrong-owner", "assets": []}
    )
    with pytest.raises(ValueError, match="ownership"):
        cpu.upload_owned("tag", "body", 1, path)
    monkeypatch.setattr(
        cpu, "api", lambda *args: {"draft": True, "tag_name": "tag", "body": "body", "assets": [{"name": path.name}]}
    )
    with pytest.raises(ValueError, match="覆寫"):
        cpu.upload_owned("tag", "body", 1, path)


def test_real_package_roundtrip_preserves_all_checkpoint_bytes(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"own-weight-fixture")
    (root / "continuation.pt").write_bytes(b"own-full-state-fixture")
    path = cpu.package(tmp_path, [("core/job", root)], "package.zip")
    import zipfile

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("package-file-manifest.json"))
        for name, record in manifest.items():
            assert record["bytes"] == len(archive.read(name))
            assert record["sha256"] == cpu.hashlib.sha256(archive.read(name)).hexdigest()
