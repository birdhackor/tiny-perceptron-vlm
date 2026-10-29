"""One-off reconciliation 的原件、完整 Git blob、source gate 與單次 CAS 離線回核。"""

import hashlib
import json
import socket
import sys
import tempfile
import zipfile
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import reconcile_failed_probe as operation  # noqa: E402
from scripts.omni_revision.finance import (  # noqa: E402
    canonical,
    digest,
    reservation_identity,
    usd,
    utc,
    validate_state,
)

ARTIFACT = Path("/workspace/work/omni-attempt-inspection-40958605486")
INITIAL = Path("/workspace/work/omni-p1-live-initialize/initialization.json")
ORIGINAL = Path("/workspace/work/omni-runtime-probe-40957487215/extracted/finish-transaction.json")
EXPECTED_BLOB = "b856c74e7cf4d6c06f789335acef25a5d6d70dfd"


def forbidden(*args, **kwargs):
    raise AssertionError("network forbidden")


def original_state():
    state = deepcopy(json.loads(INITIAL.read_text())["state"])
    original = json.loads(ORIGINAL.read_text())["attempt"]
    fixture = ROOT / "docs/course-revision-20261029-omni/infrastructure/failed-probe-attempt-v1.json"
    assert json.loads(fixture.read_text()) == original
    state["attempts"] = [original]
    state["reserved_total_usd"] = "0.40"
    state["last_observed_at"] = original["billing_snapshot"]["observed_at"]
    state["month_gross_highwater_usd"] = {
        row["cycle"]: str(usd(row["metered_cost_usd"])) for row in original["billing_snapshot"]["summaries"]
    }
    raw = canonical(state) + b"\n"
    actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert actual_blob == EXPECTED_BLOB
    assert digest(original["receipt"]) == operation.ORIGINAL_RECEIPT_SHA256
    return state


def inspection():
    assert hashlib.sha256((ARTIFACT / "artifact.zip").read_bytes()).hexdigest() == (
        "46b60d9a779c0a2a7ce2aad8575a5ba65372622b5f3dab621f252716f721821a"
    )
    with zipfile.ZipFile(ARTIFACT / "artifact.zip") as archive:
        for path in (ARTIFACT / "extracted").iterdir():
            assert archive.read(path.name) == path.read_bytes()
    return json.loads((ARTIFACT / "extracted/attempt-inspection.json").read_text())


def transition_probe(state, evidence):
    # 用原觀察時間重放既有原件；線上 operation 仍會另取 fresh inspection。
    at = utc(evidence["observed_at"]) + timedelta(seconds=1)
    validate_state(state, at)
    following = operation.resolve(state, evidence, at)
    validate_state(following, at)
    old, new = state["attempts"][0], following["attempts"][0]
    assert reservation_identity(old) == reservation_identity(new)
    assert new["receipt"]["original_receipt"] == old["receipt"]
    assert new["receipt"]["success"] is False and new["status"] == "failed-stopped"
    assert following["reserved_total_usd"] == state["reserved_total_usd"] == "0.40"
    assert {key: value for key, value in following.items() if key != "attempts"} == {
        key: value for key, value in state.items() if key != "attempts"
    }
    return {
        "before_and_after_validate_state": True,
        "reservation_identity_unchanged": True,
        "all_nonattempt_state_unchanged": True,
        "full_bound_retained": "0.40",
        "new_status": new["status"],
    }


def main_cas_probe(state, evidence, fail):
    at = utc(evidence["observed_at"]) + timedelta(seconds=1)
    calls = {"read": 0, "inspection": 0, "write": 0}

    class Clock(datetime):
        @classmethod
        def now(cls, zone):
            assert zone is UTC
            return at

    class Ledger:
        def read(self):
            calls["read"] += 1
            return deepcopy(state), EXPECTED_BLOB

        def write(self, following, blob):
            calls["write"] += 1
            assert blob == EXPECTED_BLOB
            validate_state(following, at)
            assert following["reserved_total_usd"] == "0.40"
            assert following["attempts"][0]["receipt"]["original_receipt"] == state["attempts"][0]["receipt"]
            if fail:
                raise RuntimeError("offline CAS rejection")
            return {"commit": "offline-only", "blob": "offline-only"}

    async def inspect(attempt):
        calls["inspection"] += 1
        assert attempt == state["attempts"][0]
        return deepcopy(evidence)

    with tempfile.TemporaryDirectory(prefix="omni-reconciliation-main-") as temporary:
        folder = Path(temporary)
        (folder / "fixture-audit.json").write_text("{}")
        with (
            patch.object(operation, "ROOT", folder),
            patch.object(operation, "AUDIT", "fixture-audit.json"),
            patch.object(operation, "checked_audit", lambda: ("a" * 40, {})),
            patch.object(operation, "datetime", Clock),
            patch.object(operation, "GitHubLedger", Ledger),
            patch.object(operation.attempt_inspection, "inspect", inspect),
            patch.object(sys, "argv", ["reconcile", "--output", str(folder / "result")]),
        ):
            try:
                operation.main()
            except RuntimeError as error:
                assert fail and str(error) == "offline CAS rejection"
            else:
                assert not fail
        assert (folder / "result/reconciliation-before-write.json").exists()
        assert (folder / "result/reconciliation-transaction.json").exists() is not fail
    assert calls == {"read": 1, "inspection": 1, "write": 1}
    return {"cas_rejected": fail, "calls": calls, "paid_or_stop_rpc": False, "automatic_retry": False}


def source_gate_probe():
    raw_sources = {name: (ROOT / name).read_bytes() for name in operation.SOURCES}
    results = {}
    with tempfile.TemporaryDirectory(prefix="omni-reconciliation-audit-") as temporary:
        folder = Path(temporary)
        committed = dict(raw_sources)
        for name, raw in raw_sources.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        valid = {
            "verdict": "pass",
            "reviewer_task": "/root/omni_execution_probe_review",
            "only_attempt_id": operation.ATTEMPT,
            "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in raw_sources.items()},
        }

        def git(arguments, **kwargs):
            if arguments == ["git", "rev-parse", "HEAD"]:
                return "a" * 40
            return committed[arguments[-1].split(":", 1)[1]]

        with patch.object(operation, "ROOT", folder), patch.object(operation.subprocess, "check_output", git):
            for case in [
                "valid_fixture",
                "not_pass",
                "wrong_reviewer",
                "wrong_attempt",
                "missing_source",
                "wrong_hash",
                "dirty_source",
                "dirty_audit",
            ]:
                document = deepcopy(valid)
                for name, raw in raw_sources.items():
                    (folder / name).write_bytes(raw)
                if case == "not_pass":
                    document["verdict"] = "blocked"
                elif case == "wrong_reviewer":
                    document["reviewer_task"] = "/root"
                elif case == "wrong_attempt":
                    document["only_attempt_id"] = "probe-other"
                elif case == "missing_source":
                    document["files"].pop(operation.SOURCES[0])
                elif case == "wrong_hash":
                    document["files"][operation.SOURCES[0]] = "0" * 64
                elif case == "dirty_source":
                    (folder / operation.SOURCES[0]).write_bytes(b"dirty")
                audit = folder / operation.AUDIT
                audit.parent.mkdir(parents=True, exist_ok=True)
                raw = json.dumps(document).encode()
                committed[operation.AUDIT] = raw
                audit.write_bytes(raw + (b" " if case == "dirty_audit" else b""))
                try:
                    operation.checked_audit()
                except ValueError:
                    results[case] = "rejected"
                else:
                    results[case] = "accepted"
    assert results == {"valid_fixture": "accepted", **{case: "rejected" for case in results if case != "valid_fixture"}}
    return results


if __name__ == "__main__":
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    state, evidence = original_state(), inspection()
    print(
        json.dumps(
            {
                "original_full_git_blob_verified": EXPECTED_BLOB,
                "transition": transition_probe(state, evidence),
                "main": [main_cas_probe(state, evidence, False), main_cas_probe(state, evidence, True)],
                "source_gate": source_gate_probe(),
            },
            indent=2,
        )
    )
