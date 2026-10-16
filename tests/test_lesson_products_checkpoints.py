"""中途HF不能只備份last；best、程式及配方都必須同組且可驗證。"""

import hashlib
import json

import pytest

from scripts.lesson_products.checkpoints import (
    DATA_MANIFEST,
    FILES,
    TRAIN_SOURCES,
    bundle,
    prepare_resume,
    save_transferred,
    transfer_payload,
    validate_resume,
)
from scripts.lesson_products.modal_runner import validate_job


def fixture(tmp_path):
    job = json.loads(open("docs/lesson-products/jobs/train-b-v1.json").read())
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in TRAIN_SOURCES}
    hashes[DATA_MANIFEST] = "d" * 64
    source = tmp_path / "lesson-products" / "original"
    source.mkdir(parents=True)
    for name in FILES:
        (source / name).write_bytes(("fixture bytes, not a model: " + name).encode())
    value = bundle(source, "original", 1000, hashes, job)
    resume = {
        "run_id": "original",
        "source": "volume",
        "hf_revision": None,
        "bundle_sha256": hashlib.sha256((source / "checkpoint-bundle.json").read_bytes()).hexdigest(),
    }
    ledger = {
        "reservations": [
            {
                "run_id": "original",
                "experiment_id": "selftrained_lesson_products",
                "status": "failed",
                "manifest_sha256": hashes[DATA_MANIFEST],
                "source_sha256": hashes.copy(),
                "runtime_options": job.copy(),
            }
        ]
    }
    return job, hashes, source, value, resume, ledger


@pytest.mark.parametrize("source_kind", ["volume", "hf"])
def test_paired_bundle_can_restore_from_volume_or_pinned_hf(tmp_path, monkeypatch, source_kind):
    job, hashes, source, _, resume, ledger = fixture(tmp_path)
    if source_kind == "hf":
        import sys
        from types import SimpleNamespace

        resume.update(source="hf", hf_revision="a" * 40)
        calls = []

        def fake_download(repo, name, revision, token):
            calls.append((repo, name, revision, token))
            return source / name.rsplit("/", 1)[1]

        monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
        monkeypatch.setenv("HF_TOKEN", "fictional-test-token")
    job["resume"] = resume
    validate_job(job)
    out = tmp_path / "restored"
    result = prepare_resume(resume, out, tmp_path, "fixture/private", hashes, job, ledger)
    assert result["files_verified"] == 5 and result["step"] == 1000
    assert all((out / name).read_bytes() == (source / name).read_bytes() for name in (*FILES, "checkpoint-bundle.json"))
    if source_kind == "hf":
        assert len(calls) == 6 and all(x[2] == "a" * 40 for x in calls)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_best",
        "corrupt_last",
        "wrong_bundle",
        "wrong_recipe",
        "wrong_source",
        "wrong_data",
        "wrong_original_data",
        "wrong_original_source",
        "wrong_original_recipe",
        "still_running",
    ],
)
def test_resume_preflight_rejects_incomplete_foreign_or_unstopped_source(tmp_path, fault):
    job, hashes, source, _, resume, ledger = fixture(tmp_path)
    if fault == "missing_best":
        (source / "best.pt").unlink()
    elif fault == "corrupt_last":
        (source / "last.pt").write_bytes(b"wrong")
    elif fault == "wrong_bundle":
        resume["bundle_sha256"] = "0" * 64
    elif fault == "wrong_recipe":
        job["seed"] += 1
    elif fault == "wrong_source":
        hashes[TRAIN_SOURCES[0]] = "0" * 64
    elif fault == "wrong_data":
        hashes[DATA_MANIFEST] = "0" * 64
    elif fault == "wrong_original_data":
        ledger["reservations"][0]["manifest_sha256"] = "0" * 64
    elif fault == "wrong_original_source":
        ledger["reservations"][0]["source_sha256"][TRAIN_SOURCES[0]] = "0" * 64
    elif fault == "wrong_original_recipe":
        ledger["reservations"][0]["runtime_options"]["seed"] += 1
    else:
        ledger["reservations"][0]["status"] = "running"
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_resume(resume, tmp_path / "restored", tmp_path, "fixture/private", hashes, job, ledger)


def test_hf_resume_cannot_use_moving_main_revision():
    with pytest.raises(ValueError, match="固定commit"):
        validate_resume({"run_id": "previous", "source": "hf", "hf_revision": "main", "bundle_sha256": "a" * 64})


def test_actual_transfer_checks_all_binary_hashes_and_does_not_put_bytes_in_receipt(tmp_path):
    _, _, source, _, _, _ = fixture(tmp_path)
    payload, hashes = transfer_payload(source)
    envelope = {"receipt": {"client_artifacts": hashes}, "artifacts": payload}
    json.dumps(envelope["receipt"])
    save_transferred(envelope, tmp_path / "local")
    assert (tmp_path / "local/best.pt").read_bytes() == payload["best.pt"]
    payload["last.pt"] = b"corruption"
    with pytest.raises(ValueError, match="指紋"):
        save_transferred(envelope, tmp_path / "corrupt")
