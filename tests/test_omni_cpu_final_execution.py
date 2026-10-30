"""CPU final外層的單次／來源／partial保存控制；全為虛構資料，不開真final。"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.omni_revision import cpu_final_execution as runner


def environment(monkeypatch):
    for key, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": runner.REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "a" * 40,
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_EVENT_NAME",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    ],
)
def test_final_identity_is_exact_authorized_manual_run(monkeypatch, bad):
    environment(monkeypatch)
    if bad is None:
        runner.identity("cpu-final-123-1")
    else:
        monkeypatch.setenv(bad, "other")
        with pytest.raises(ValueError):
            runner.identity("cpu-final-123-1")


@pytest.mark.parametrize("status", ["reserved", "completed", "failed-stopped", "unresolved"])
def test_no_previous_final_attempt_can_be_retried(status):
    state = {"attempts": [{"status": status, "purpose": json.dumps({"job": runner.JOB})}]}
    with pytest.raises(ValueError, match="重跑"):
        runner.prior_attempts(state)
    runner.prior_attempts({"attempts": [{"purpose": json.dumps({"job": "cpu-core-formal"})}]})


def core_fixture(tmp_path, monkeypatch, failure=None):
    root = tmp_path / "checkout"
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    files = {
        "core/fixture/training/weights.safetensors": b"explicit bytes only, never parsed",
        "selection-evidence/calibration.json": b"{}",
    }
    for name in [
        "data-definition.json",
        "metrics.json",
        "formal-recipes.json",
        "codec-selection.json",
        "model-selection.json",
    ]:
        raw = json.dumps({"fixture": name}).encode()
        (frozen / name).write_bytes(raw)
        files["frozen/" + name] = b"different" if failure == "frozen" and name == "metrics.json" else raw
    if failure == "escape":
        files["core/../escape"] = b"outside"
    package = tmp_path / "source.zip"
    with zipfile.ZipFile(package, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr(
            "package-file-manifest.json",
            json.dumps(
                {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()}
            ),
        )
    asset = {"name": "core-checkpoints.zip", "bytes": package.stat().st_size, "sha256": runner.file_sha(package)}
    if failure == "outer":
        asset["sha256"] = "f" * 64
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "FROZEN", frozen)
    monkeypatch.setattr(runner, "check_metadata", lambda _: {"core_selection_sha256": "c" * 64})
    monkeypatch.setattr(runner, "completed_core", lambda _: ("owned-fixture", asset))
    monkeypatch.setattr(
        runner, "command", lambda args: (Path(args[-1]) / asset["name"]).write_bytes(package.read_bytes())
    )
    opened = []
    monkeypatch.setattr(runner.core_final, "opening", lambda core, selected: opened.append((core, selected)))
    return root, opened


def test_prepare_core_delegates_actual_strict_opener_only_after_verified_inputs(tmp_path, monkeypatch):
    root, calls = core_fixture(tmp_path, monkeypatch)
    result = runner.prepare_core("a" * 40, {})
    assert result["core_asset"]["name"] == "core-checkpoints.zip"
    assert calls == [(root / "outputs/omni-formal-core-v1", "c" * 64)]
    assert (root / "outputs/omni-core-selection-v1/calibration.json").read_bytes() == b"{}"


@pytest.mark.parametrize("failure", ["outer", "frozen", "escape", "existing"])
def test_bad_core_never_reaches_opener_or_overwrites_existing(tmp_path, monkeypatch, failure):
    root, calls = core_fixture(tmp_path, monkeypatch, failure)
    if failure == "existing":
        p = root / "outputs/omni-formal-core-v1"
        p.mkdir(parents=True)
        (p / "user.txt").write_bytes(b"keep")
    with pytest.raises(ValueError):
        runner.prepare_core("a" * 40, {})
    assert calls == []
    assert not (root / "outputs/escape").exists()
    if failure == "existing":
        assert (root / "outputs/omni-formal-core-v1/user.txt").read_bytes() == b"keep"


def execution_fixture(tmp_path, monkeypatch, failed_part=None):
    environment(monkeypatch)
    output = tmp_path / "final"
    output.mkdir()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "model-selection.json").write_bytes(b"explicit selection bytes")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "FROZEN", frozen)
    monkeypatch.setattr(runner, "checked_sources", lambda: {"revision": "a" * 40})
    reserved = {
        "attempt_id": "cpu-final-123-1",
        "revision": "a" * 40,
        "purpose": json.dumps(
            {
                "job": runner.JOB,
                "ownership": {"attempt": "cpu-final-123-1"},
                "sources_sha256": runner.digest({"revision": "a" * 40}),
            }
        ),
        "components": dict(runner.transport.COMPONENTS),
        "bound_usd": "0.10",
        "status": "reserved",
    }
    (output / "reservation.json").write_text(json.dumps({"attempt": reserved}))
    monkeypatch.setattr(
        runner,
        "reservation_identity",
        lambda row: {key: row[key] for key in ("attempt_id", "revision", "purpose", "components", "bound_usd")},
    )
    monkeypatch.setattr(
        runner,
        "GitHubLedger",
        lambda: type("FixtureLedger", (), {"read": lambda _: ({"attempts": [reserved]}, "fixture-blob")})(),
    )
    monkeypatch.setattr(runner, "find_draft", lambda *args: None)
    provider = {}

    def api(endpoint, method="GET", payload=None):
        if method == "POST":
            provider.update(id=7, **payload)
        return dict(provider)

    monkeypatch.setattr(runner, "api", api)
    calls = []

    def child(arguments, log, seconds, state):
        log.write_text("explicit mocked child, no NN or training\n")
        state["all_training_processes_exited"] = True
        if arguments[1].endswith("core_report"):
            Path(arguments[-1]).write_text('{"explicit_fixture":true}')
            return
        part = arguments[2]
        calls.append(part)
        if part == failed_part:
            raise RuntimeError("explicit failed part")
        target = Path(arguments[arguments.index("--output") + 1])
        target.mkdir(parents=True)
        (target / "results.json").write_text('{"explicit_fixture":true}')

    monkeypatch.setattr(runner.transport, "bounded_child", child)
    exported = []

    def upload(tag, body, release_id, path):
        exported.append((tag, body, release_id, path))
        return {
            "id": 9,
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": runner.file_sha(path),
            "roundtrip_verified": True,
        }

    monkeypatch.setattr(runner.transport, "upload_owned", upload)
    return output, calls, exported


def test_four_parts_run_once_and_record_second_entry_is_rejected(tmp_path, monkeypatch):
    output, calls, exported = execution_fixture(tmp_path, monkeypatch)
    runner.run_final("cpu-final-123-1", output)
    assert calls == ["av", "sound", "choice", "integration"]
    assert len(exported) == 1
    receipt = (output / "cpu-final-receipt.json").read_bytes()
    assert json.loads(receipt)["success"] is True
    with pytest.raises(FileExistsError):
        runner.run_final("cpu-final-123-1", output)
    assert (output / "cpu-final-receipt.json").read_bytes() == receipt
    assert len(calls) == 4


def test_failed_measurement_keeps_partial_and_never_runs_later_parts(tmp_path, monkeypatch):
    output, calls, exported = execution_fixture(tmp_path, monkeypatch, "sound")
    with pytest.raises(RuntimeError, match="不完整"):
        runner.run_final("cpu-final-123-1", output)
    assert calls == ["av", "sound"]
    assert len(exported) == 1
    receipt = json.loads((output / "cpu-final-receipt.json").read_bytes())
    assert receipt["success"] is False and receipt["completed_parts"] == ["av"]
    assert (output / "measurements/av/results.json").is_file()
    assert not (output / "measurements/choice").exists()


def finish_fixture(tmp_path, monkeypatch):
    output, _, _ = execution_fixture(tmp_path, monkeypatch)
    runner.run_final("cpu-final-123-1", output)
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    ownership = json.loads((output / "release-ownership.json").read_bytes())
    receipt = json.loads((output / "cpu-final-receipt.json").read_bytes())
    writes, provider_calls = [], []

    class Ledger:
        def read(self):
            return {"attempts": [reserved]}, "fixture-blob"

        def write(self, following, blob):
            assert blob == "fixture-blob"
            writes.append(following)
            return {"fixture": True}

    def api(endpoint):
        provider_calls.append(endpoint)
        if endpoint.startswith("releases/"):
            return {
                "id": ownership["release_id"],
                "tag_name": ownership["tag"],
                "body": ownership["body"],
                "draft": True,
                "target_commitish": receipt["source_revision"],
                "assets": [{"id": row["id"], "name": row["name"], "size": row["bytes"]} for row in receipt["assets"]],
            }
        return {
            "total_count": 1,
            "artifacts": [
                {
                    "id": 99,
                    "name": "omni-final-123-1",
                    "size_in_bytes": 1024,
                    "created_at": "2026-10-01T00:00:00Z",
                    "expires_at": "2026-10-08T00:00:00Z",
                    "expired": False,
                }
            ],
        }

    monkeypatch.setattr(runner, "GitHubLedger", Ledger)
    monkeypatch.setattr(runner, "api", api)
    monkeypatch.setenv("OMNI_ARTIFACT_OUTCOME", "success")
    return output, reserved, ownership, receipt, writes, provider_calls


def test_finish_binds_this_attempt_source_and_exact_owned_provider(tmp_path, monkeypatch):
    output, _, _, _, writes, calls = finish_fixture(tmp_path, monkeypatch)
    runner.finish_final("cpu-final-123-1", output)
    assert len(writes) == 1 and calls
    assert writes[0]["attempts"][0]["status"] == "completed"


@pytest.mark.parametrize(
    "failure",
    [
        "attempt",
        "source",
        "foreign_draft",
        "receipt_id",
        "ownership_id",
        "reserved_revision",
        "sources_digest",
        "job",
        "schema",
    ],
)
def test_foreign_receipts_rejected_before_provider_lookup_or_ledger_write(tmp_path, monkeypatch, failure):
    output, reserved, ownership, receipt, writes, calls = finish_fixture(tmp_path, monkeypatch)
    if failure == "attempt":
        receipt["attempt"] = "cpu-final-999-1"
    elif failure == "source":
        receipt["source_revision"] = "b" * 40
    elif failure == "foreign_draft":
        tag, body = runner.release_identity("cpu-final-999-1", "a" * 40)
        receipt["release_tag"] = tag
        ownership.update(tag=tag, body=body)
    elif failure == "receipt_id":
        receipt["release_id"] = 888
    elif failure == "ownership_id":
        ownership["release_id"] = 888
    elif failure == "reserved_revision":
        reserved["revision"] = "b" * 40
    elif failure in {"sources_digest", "job"}:
        purpose = json.loads(reserved["purpose"])
        purpose["sources_sha256" if failure == "sources_digest" else "job"] = "foreign"
        reserved["purpose"] = json.dumps(purpose)
    else:
        receipt["schema"] = "foreign"
    (output / "reservation.json").write_text(json.dumps({"attempt": reserved}))
    (output / "release-ownership.json").write_text(json.dumps(ownership))
    (output / "cpu-final-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="身分不同"):
        runner.finish_final("cpu-final-123-1", output)
    assert writes == [] and calls == []


def test_missing_owned_release_marks_export_failed_immediately(tmp_path, monkeypatch):
    output, _, _ = execution_fixture(tmp_path, monkeypatch)

    def api(endpoint, method="GET", payload=None):
        return {"id": 7} if method == "POST" else None

    monkeypatch.setattr(runner, "api", api)
    with pytest.raises(RuntimeError, match="不完整"):
        runner.run_final("cpu-final-123-1", output)
    receipt = json.loads((output / "cpu-final-receipt.json").read_bytes())
    assert receipt["success"] is False
    assert receipt["export_error_type"] == "ValueError"
