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


def test_interrupted_validation_reads_journal_metrics_and_raw_prefix_without_completed_receipt(tmp_path, monkeypatch):
    prefix = "selftrained/selftrained-v2/validation/gha-37426148065-1/"
    values = {
        "execution.json": b'{"status":"running","stage":"validation"}\n',
        "evaluation-receipt.json": b'{"status":"interrupted","completed_count":1}\n',
        "metrics.json": b'{"evaluation_complete":false,"count":1}\n',
        "outputs.jsonl": b'{"record":{"id":"fixture-completed-row"}}\n{"uncommitted-tail":',
    }
    calls = []
    fake_sdk(monkeypatch, {prefix + name: raw for name, raw in values.items()}, calls)
    output = tmp_path / "out"
    receipt = inspection.inspect_attempt("selftrained-v2", "validation", "gha-37426148065-1", output, "a" * 40)
    assert calls == [prefix + name for name in inspection.FILES]
    observed = {item["path"]: item for item in receipt["files"]}
    assert observed["receipt.json"]["status"] == "absent"
    assert receipt["status"] == "complete" and receipt["mode"] == "metadata"
    assert receipt["total_downloaded_bytes"] == sum(map(len, values.values()))
    for name, raw in values.items():
        assert (output / name).read_bytes() == raw
        assert observed[name] == {
            "path": name,
            "status": "present",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    assert not any(receipt[key] for key in ("remote_container_started", "gpu_used", "ledger_written", "weights_read"))
    assert not any(path.endswith((".pt", ".safetensors")) or path.endswith("budget.json") for path in calls)
    monkeypatch.setattr(inspection, "MAX_BYTES", receipt["total_downloaded_bytes"] - 1)
    calls.clear()
    with pytest.raises(ValueError, match="64 MiB"):
        inspection.inspect_attempt("selftrained-v2", "validation", "gha-37426148065-1", tmp_path / "bounded", "a" * 40)
    failed = json.loads((tmp_path / "bounded/inspection-receipt.json").read_text())
    assert failed["status"] == "failed" and failed["error_type"] == "ValueError"
    assert calls[-1] == prefix + "outputs.jsonl"


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


def completed_safe_source():
    def encode(value):
        return (json.dumps(value) + "\n").encode()

    config = {"architecture": "moe", "width": 32}
    execution = {
        "status": "completed",
        "batch_id": "selftrained-v1",
        "run_id": "gha-37409271685-1",
        "stage": "pretrain",
        "revision": "b" * 40,
        "manifest_sha256": "c" * 64,
        "job": {"architecture": "moe", "steps": 600},
    }
    training = {
        "stage": "pretrain",
        "architecture": "moe",
        "steps": 600,
        "completed_requested_steps": True,
        "selected_checkpoint_available": True,
        "inference_exported": True,
        "test_used_for_selection": False,
        "origin": {"kind": "all-neural-weights-random"},
        "config": config,
    }
    values = {
        "execution.json": encode(execution),
        "train-receipt.json": encode(training),
        "runner.log": b"actual fixture child log\n",
        "metrics.jsonl": b'{"step":600}\n',
        "model.safetensors": b"selected safe weights fixture",
        "model-config.json": encode(config),
        "tokenizer.json": encode({"chars": ["a", "b"]}),
    }
    values["inference-manifest.json"] = encode(
        {
            "stage": "pretrain",
            "selection": "validation_loss",
            "origin": training["origin"],
            "selected_checkpoint_sha256": "d" * 64,
            "files": {name: hashlib.sha256(values[name]).hexdigest() for name in inspection.SAFE_EXPORT_FILES[:3]},
        }
    )
    source = execution | {
        "files": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in values.items()
        ]
        + [{"path": "best.pt", "bytes": 100, "sha256": "d" * 64}]
    }
    values["receipt.json"] = encode(source)
    return {PREFIX + name: raw for name, raw in values.items()}, hashlib.sha256(values["receipt.json"]).hexdigest()


def update_pinned_json(contents, name, change):
    value = json.loads(contents[PREFIX + name])
    change(value)
    raw = (json.dumps(value) + "\n").encode()
    contents[PREFIX + name] = raw
    source = json.loads(contents[PREFIX + "receipt.json"])
    if name != "receipt.json":
        for item in source["files"]:
            if item["path"] == name:
                item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    else:
        source = value
    contents[PREFIX + "receipt.json"] = (json.dumps(source) + "\n").encode()
    return hashlib.sha256(contents[PREFIX + "receipt.json"]).hexdigest()


def test_safe_export_reads_only_pinned_selected_files_after_completed_provenance(tmp_path, monkeypatch):
    contents, receipt_sha = completed_safe_source()
    calls = []
    fake_sdk(monkeypatch, contents, calls)
    receipt = inspection.inspect_attempt(
        "selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "out", "a" * 40, True, receipt_sha
    )
    assert calls == [PREFIX + name for name in (*inspection.FILES, *inspection.SAFE_EXPORT_FILES[:3])]
    assert receipt["mode"] == "safe-export" and receipt["weights_read"] is True
    assert receipt["status"] == "complete" and receipt["source_receipt_sha256"] == receipt_sha
    assert {item["path"] for item in receipt["verified_safe_files"]} == set(inspection.SAFE_EXPORT_FILES)
    assert not any(path.endswith(".pt") or "optimizer" in path or "/assets/" in path for path in calls)
    assert (tmp_path / "out/model.safetensors").read_bytes() == contents[PREFIX + "model.safetensors"]


@pytest.mark.parametrize("receipt_sha", [None, "main", "a" * 63])
def test_safe_export_requires_reviewed_receipt_pin_before_any_client_lookup(tmp_path, monkeypatch, receipt_sha):
    fake_sdk(monkeypatch, {}, [], error=AssertionError("client must not be touched"))
    with pytest.raises(ValueError, match="receipt SHA"):
        inspection.inspect_attempt(
            "selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "out", "a" * 40, True, receipt_sha
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["wrong-pin", "partial", "nonrandom", "manifest-mismatch", "metadata-tamper"])
def test_invalid_safe_export_provenance_fails_before_reading_weights(tmp_path, monkeypatch, mutation):
    contents, pin = completed_safe_source()
    if mutation == "wrong-pin":
        pin = "e" * 64
    elif mutation == "partial":
        pin = update_pinned_json(contents, "receipt.json", lambda value: value.update(status="failed"))
    elif mutation == "nonrandom":
        pin = update_pinned_json(contents, "train-receipt.json", lambda value: value.update(origin={"kind": "mature"}))
    elif mutation == "manifest-mismatch":
        pin = update_pinned_json(
            contents, "inference-manifest.json", lambda value: value["files"].update({"model.safetensors": "e" * 64})
        )
    else:
        contents[PREFIX + "train-receipt.json"] += b" "
    calls = []
    fake_sdk(monkeypatch, contents, calls)
    with pytest.raises(ValueError):
        inspection.inspect_attempt(
            "selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "out", "a" * 40, True, pin
        )
    assert calls == [PREFIX + name for name in inspection.FILES]
    receipt = json.loads((tmp_path / "out/inspection-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["weights_read"] is False


def test_safe_weight_tamper_and_total_allowance_are_fail_closed(tmp_path, monkeypatch):
    contents, pin = completed_safe_source()
    contents[PREFIX + "model.safetensors"] += b"changed"
    calls = []
    fake_sdk(monkeypatch, contents, calls)
    with pytest.raises(ValueError, match="pinned completed receipt"):
        inspection.inspect_attempt(
            "selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "tampered", "a" * 40, True, pin
        )
    assert calls[-1] == PREFIX + "model.safetensors" and not any(path.endswith(".pt") for path in calls)
    contents, pin = completed_safe_source()
    metadata_size = sum(len(contents.get(PREFIX + name, b"")) for name in inspection.FILES)
    monkeypatch.setattr(inspection, "MAX_BYTES", metadata_size)
    calls.clear()
    fake_sdk(monkeypatch, contents, calls)
    with pytest.raises(ValueError, match="64 MiB"):
        inspection.inspect_attempt(
            "selftrained-v1", "pretrain", "gha-37409271685-1", tmp_path / "bounded", "a" * 40, True, pin
        )
    assert calls == [PREFIX + name for name in inspection.FILES]
