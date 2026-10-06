"""Verify private attempt inspection without credentials or cloud mutation APIs."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selftrained_inspection", ROOT / "scripts/selftrained/inspect_attempt.py")
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)
PREFIX = "selftrained/selftrained-v1/pretrain/gha-37409271685-1/"


def fake_sdk(monkeypatch, contents, calls, error=None):
    class Volume:
        def read_file(self, path):
            calls.append(path)
            if error:
                raise error
            if path not in contents:
                raise FileNotFoundError(path)
            value = contents[path]
            return iter((value[:2], value[2:]))

    def lookup(name, create_if_missing):
        assert name == "tiny-perceptron-course" and create_if_missing is False
        return Volume()

    # No App, function, deploy, billing write or Volume mutation method exists.
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=SimpleNamespace(from_name=lookup)))


def test_only_reads_exact_metadata_allowlist_and_preserves_raw_bytes(tmp_path, monkeypatch):
    calls = []
    value = b'{"status":"running","stage":"pretrain"}\n'
    fake_sdk(monkeypatch, {PREFIX + "execution.json": value}, calls)
    receipt = inspection.inspect_attempt("selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "out", "a" * 40)
    assert calls == [PREFIX + name for name in inspection.FILES]
    assert (tmp_path / "out/execution.json").read_bytes() == value
    assert receipt["status"] == "complete" and receipt["total_downloaded_bytes"] == len(value)
    assert receipt["files"][0]["sha256"] == hashlib.sha256(value).hexdigest()
    assert all(item["status"] == "absent" for item in receipt["files"][1:])
    assert not any(receipt[key] for key in ("remote_container_started", "gpu_used", "ledger_written", "weights_read"))
    assert not any(path.endswith((".pt", ".safetensors")) for path in calls)


@pytest.mark.parametrize(
    "batch,stage,run,revision",
    [
        ("../other", "pretrain", "attempt", "a" * 40),
        ("batch", "unknown", "attempt", "a" * 40),
        ("batch", "pretrain", "../../budget.json", "a" * 40),
        ("batch", "pretrain", "attempt", "main"),
    ],
)
def test_rejects_unsafe_or_unpinned_inspection_before_client_lookup(tmp_path, monkeypatch, batch, stage, run, revision):
    fake_sdk(monkeypatch, {}, [], error=AssertionError("client must not be touched"))
    with pytest.raises(ValueError):
        inspection.inspect_attempt(batch, stage, run, tmp_path / "out", revision)
    assert not (tmp_path / "out").exists()


def test_byte_overflow_or_auth_failure_is_failed_not_a_fake_absent_file(tmp_path, monkeypatch):
    calls = []
    fake_sdk(monkeypatch, {PREFIX + "execution.json": b"abcdef"}, calls)
    monkeypatch.setattr(inspection, "MAX_BYTES", 3)
    with pytest.raises(ValueError, match="64 MiB"):
        inspection.inspect_attempt("selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "too-big", "a" * 40)
    receipt = json.loads((tmp_path / "too-big/inspection-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"
    assert len(calls) == 1
    fake_sdk(monkeypatch, {}, [], error=PermissionError("fixture authentication failure"))
    with pytest.raises(PermissionError):
        inspection.inspect_attempt("selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "denied", "a" * 40)
    receipt = json.loads((tmp_path / "denied/inspection-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "PermissionError" and receipt["files"] == []
