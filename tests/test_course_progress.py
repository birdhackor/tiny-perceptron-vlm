"""進度重建不能遺失費用歷史，亦不能把規劃旗標當成公開權重證據。"""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "course_progress_candidate", PROJECT / "scripts/update_course_progress.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempt(run_id="run-1", reserved="0.22", cumulative="0.22", status="failed"):
    return {
        "run_id": run_id,
        "batch_id": "course-v1",
        "experiment_id": "example",
        "mode": "run",
        "reserved_usd": reserved,
        "status": status,
        "observed_cumulative_reserved_usd": cumulative,
    }


def receipt(entry, cumulative):
    return {"billing": {"entry": entry, "reserved_total_usd": cumulative}}


def test_empty_local_receipts_preserve_durable_history_and_failed_reservations(builder):
    history = {"observed_reserved_total_usd": "8.22", "attempts": [attempt()]}
    result = builder.merge_budget_reservations(history, [], 10)
    assert result["observed_reserved_total_usd"] == "8.22"
    assert result["attempts"] == history["attempts"]
    assert result["attempts"][0]["status"] == "failed"


def test_merge_is_unique_by_run_and_decimal_amounts_not_float_strings(builder):
    old = attempt(status="completed")
    duplicate = {**old, "reserved_usd": "0.220", "account_only_metadata": "not exported"}
    result = builder.merge_budget_reservations(
        {"attempts": [old]}, [receipt(duplicate, "0.44"), receipt(attempt("run-2"), "0.44")], 10
    )
    assert len(result["attempts"]) == 2
    assert result["observed_reserved_total_usd"] == "0.44"
    assert result["attempts"][0]["observed_cumulative_reserved_usd"] == "0.44"
    assert all("account_only_metadata" not in entry for entry in result["attempts"])


def test_cumulative_total_keeps_maximum_and_cannot_be_lower_than_unique_attempt_sum(builder):
    entries = [receipt(attempt(f"run-{i}"), "0.22") for i in range(3)]
    assert builder.merge_budget_reservations({}, entries, 10)["observed_reserved_total_usd"] == "0.66"
    history = {"observed_reserved_total_usd": "0.70", "attempts": []}
    assert builder.merge_budget_reservations(history, entries, 10)["observed_reserved_total_usd"] == "0.70"


@pytest.mark.parametrize(
    "field,value",
    [("batch_id", "different"), ("experiment_id", "different"), ("mode", "release"), ("reserved_usd", "0.04")],
)
def test_same_run_cannot_be_reassigned_or_change_its_reservation(builder, field, value):
    old = attempt()
    with pytest.raises(ValueError, match="Conflicting reservation"):
        builder.merge_budget_reservations({"attempts": [old]}, [receipt({**old, field: value}, "0.22")], 10)


def test_pending_status_can_finish_but_failure_cannot_be_erased(builder):
    old = attempt(status="reserved")
    result = builder.merge_budget_reservations({"attempts": [old]}, [receipt(attempt(status="failed"), "0.22")], 10)
    assert result["attempts"][0]["status"] == "failed"
    with pytest.raises(ValueError, match="terminal status"):
        builder.merge_budget_reservations(result, [receipt(attempt(status="completed"), "0.22")], 10)


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-0.01", True, None])
def test_invalid_currency_cannot_silently_reset_or_corrupt_the_budget(builder, amount):
    with pytest.raises(ValueError, match="finite nonnegative"):
        builder.merge_budget_reservations({}, [receipt(attempt(reserved=amount), "0.22")], 10)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def verified_release():
    return {
        "schema_version": 1,
        "experiment_id": "capstone_joint",
        "batch_id": "course-integration-v2",
        "release_status": "completed",
        "anonymous_download_verified": True,
        "source_run_id": "synthetic-test-receipt",
        "public_manifest": {
            "repo": "birdhackor/tiny-perceptron-course-models",
            "revision": "a" * 40,
            "files": [{"path": "course/test/model.pt", "output": "model.pt", "sha256": "b" * 64, "bytes": 100}],
        },
    }


def release_spec():
    return {
        "id": "capstone_joint",
        "batch_id": "course-integration-v2",
        "student_model_release": True,
        "release_evidence": "docs/course-experiments/public-releases/capstone_joint.json",
    }


def test_planned_model_release_does_not_imply_upload_or_public_verification(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    spec = release_spec()
    assert not builder.student_release_evidence({"id": "planned", "student_model_release": True})["published"]
    assert not builder.student_release_evidence(spec)["published"]
    pending = {**verified_release(), "anonymous_download_verified": False}
    write_json(tmp_path / spec["release_evidence"], pending)
    assert not builder.student_release_evidence(spec)["published"]


def test_validated_release_receipt_supplies_public_pin_and_evidence_hash(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    spec = release_spec()
    write_json(tmp_path / spec["release_evidence"], verified_release())
    result = builder.student_release_evidence(spec)
    assert result["published"] and result["verified_checkpoints"] == ["model.pt"]
    assert result["revision"] == "a" * 40 and len(result["evidence_sha256"]) == 64


@pytest.mark.parametrize(
    "damage",
    ["experiment", "batch", "pin", "hash", "bytes", "fixture", "duplicate", "path", "repository", "source_run"],
)
def test_claimed_verified_release_must_have_consistent_identity_and_actual_weight_manifest(
    builder, tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    spec, evidence = release_spec(), verified_release()
    file = evidence["public_manifest"]["files"][0]
    if damage == "experiment":
        evidence["experiment_id"] = "flash_probe"
    elif damage == "batch":
        evidence["batch_id"] = "old-batch"
    elif damage == "pin":
        evidence["public_manifest"]["revision"] = "main"
    elif damage == "hash":
        file["sha256"] = "wrong"
    elif damage == "bytes":
        file["bytes"] = 0
    elif damage == "fixture":
        file["output"] = "fixture.pt"
    elif damage == "duplicate":
        evidence["public_manifest"]["files"].append(dict(file))
    elif damage == "path":
        file["output"] = "../model.pt"
    elif damage == "repository":
        evidence["public_manifest"]["repo"] = "other/private"
    elif damage == "source_run":
        evidence["source_run_id"] = ""
    write_json(tmp_path / spec["release_evidence"], evidence)
    with pytest.raises(ValueError):
        builder.student_release_evidence(spec)


def test_actual_legacy_nested_release_receipt_remains_supported(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    spec = release_spec()
    flat = verified_release()
    nested = {
        "experiment_id": flat["experiment_id"],
        "batch_id": flat["batch_id"],
        "mode": "release",
        "release": {
            "experiment_id": flat["experiment_id"],
            "anonymous_download_verified": True,
            "repo": flat["public_manifest"]["repo"],
            "revision": flat["public_manifest"]["revision"],
            "public_manifest": flat["public_manifest"],
        },
    }
    write_json(tmp_path / spec["release_evidence"], nested)
    assert builder.student_release_evidence(spec)["published"]


def test_release_evidence_cannot_escape_repository(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="repository-relative"):
        builder.student_release_evidence({"id": "bad", "release_evidence": "../receipt.json"})
    with pytest.raises(ValueError, match="repository-relative"):
        builder.student_release_evidence({"id": "bad", "release_evidence": str(tmp_path / "receipt.json")})


def test_builder_merges_nested_integration_receipts_and_keeps_supporting_counts_separate(
    builder, tmp_path, monkeypatch
):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    docs = tmp_path / "docs/course-experiments"
    plan = {
        "budget": {"maximum_compute_cost": 10},
        "sequence": [{"id": "formal", "lessons": []}],
        "supporting_experiments": [
            {
                "id": "capstone_joint",
                "kind": "integration_training",
                "scope": "small test",
                "lessons": [],
                "student_model_release": True,
            }
        ],
    }
    write_json(docs / "plan.json", plan)
    write_json(docs / "budget-reservations.json", {"observed_reserved_total_usd": "0.22", "attempts": [attempt()]})
    write_json(
        docs / "results/capstone_joint.json",
        {"evidence_status": "complete_run", "results": {"schedule_completed": True}},
    )
    write_json(
        tmp_path / "outputs/integration-runs/nested/new/result.json",
        receipt(attempt("run-2", status="completed"), "0.44"),
    )
    for name in ("README.md", "first-steps.md", "training.md", "glossary.md"):
        file = tmp_path / "course" / name
        file.parent.mkdir(exist_ok=True)
        file.write_text("")
    # 只在隔離 temp ROOT 建 report，不執行 repo 的正式進度或帳本重建。
    builder.main()
    report = json.loads((docs / "progress.json").read_text())
    budget = json.loads((docs / "budget-reservations.json").read_text())
    assert report["counts"]["experiments"] == 1 and report["counts"]["complete_runs"] == 0
    assert report["counts"]["supporting_experiments"] == report["counts"]["complete_supporting_runs"] == 1
    supporting = report["supporting_evidence"][0]
    assert supporting["probe_outcome"] == "complete_run" and not supporting["student_model_release"]
    assert len(budget["attempts"]) == 2 and budget["observed_reserved_total_usd"] == "0.44"
    assert budget["attempts"][0]["status"] == "failed"
    builder.main()
    assert json.loads((docs / "budget-reservations.json").read_text()) == budget
