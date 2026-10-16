"""下載完成紀錄必須綁原帳本；離線假Volume不啟動任何容器。"""

import hashlib
import json

import pytest

from scripts.lesson_products.inspect import inspect_attempt, read_bounded
from scripts.selftrained.modal_runner import canonical_sha


class Volume:
    def __init__(self, files):
        self.files = files
        self.reads = []

    def read_file(self, name):
        self.reads.append(name)
        raw = self.files[name]
        yield raw[:3]
        yield raw[3:]


def fixture():
    raw = b"actual model bytes"
    receipt = {
        "run_id": "existing-attempt",
        "code_revision": "a" * 40,
        "status": "completed",
        "source_hashes": {"source.py": "b" * 64},
        "client_artifacts": {"model.pt": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
    }
    entry = {
        "run_id": receipt["run_id"],
        "experiment_id": "selftrained_lesson_products",
        "revision": receipt["code_revision"],
        "status": receipt["status"],
        "source_sha256": receipt["source_hashes"],
        "receipt_sha256": canonical_sha(receipt),
    }
    volume = Volume(
        {
            "lesson-products/existing-attempt/execution.json": json.dumps(receipt).encode(),
            "lesson-products/existing-attempt/model.pt": raw,
        }
    )
    live = {"ledger": {"reservations": [entry]}, "live": {"gross_quota_snapshot": {"actual": "7.5"}}}
    return volume, live


def test_download_requires_original_terminal_receipt_and_verifies_actual_artifact(tmp_path):
    volume, live = fixture()
    before = json.dumps(live, sort_keys=True)
    result = inspect_attempt(volume, live, "existing-attempt", "a" * 40, tmp_path)
    assert (tmp_path / "files/model.pt").read_bytes() == b"actual model bytes"
    assert result["remote_container_started"] is False
    assert result["ledger_written"] is False
    assert json.dumps(live, sort_keys=True) == before


@pytest.mark.parametrize(
    "field,value", [("status", "running"), ("revision", "c" * 40), ("receipt_sha256", "0" * 64), ("source_sha256", {})]
)
def test_mismatched_or_nonterminal_attempt_is_rejected(tmp_path, field, value):
    volume, live = fixture()
    live["ledger"]["reservations"][0][field] = value
    with pytest.raises(ValueError):
        inspect_attempt(volume, live, "existing-attempt", "a" * 40, tmp_path)
    assert not (tmp_path / "files/model.pt").exists()


def test_tampered_payload_and_oversized_chunks_are_rejected(tmp_path):
    volume, live = fixture()
    volume.files["lesson-products/existing-attempt/model.pt"] = b"wrong model bytes!"
    with pytest.raises(ValueError):
        inspect_attempt(volume, live, "existing-attempt", "a" * 40, tmp_path)
    with pytest.raises(ValueError, match="上限"):
        read_bounded(Volume({"x": b"1234"}), "x", 3)
