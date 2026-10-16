"""獨立離線inspection審核；真finish_entry與下載控制流、只提供read_file的Volume。"""
import contextlib
import copy
import hashlib
import io
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import inspect as inspection
from scripts.lesson_products.checkpoints import TRANSFER_FILES
from scripts.selftrained.modal_runner import canonical_sha, finish_entry


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(status="completed", names=TRANSFER_FILES):
    run, revision = "synthetic-terminal-attempt", "b" * 40
    artifacts = {name: ("explicit synthetic artifact " + name).encode() for name in names}
    source = {"synthetic/source.py": "a" * 64}
    receipt = {"run_id": run, "code_revision": revision, "status": status, "source_hashes": source,
               "client_artifacts": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in artifacts.items()}}
    ledger = {"reserved_total_usd": "10.10", "reservations": [{"run_id": "old", "experiment_id": "selftrained_history", "reserved_usd": "10.00", "status": "completed"},
        {"run_id": run, "experiment_id": "selftrained_lesson_products", "reserved_usd": ".10", "revision": revision, "source_sha256": source, "status": "running"}]}
    finish_entry(ledger, run, status, receipt)
    contents = {f"lesson-products/{run}/execution.json": json.dumps(receipt, separators=(",", ":")).encode(),
                **{f"lesson-products/{run}/{name}": raw for name, raw in artifacts.items()}}
    live = {"ledger": ledger, "live": {"gross_quota_snapshot": {"explicit_synthetic_readonly_snapshot": True}}}
    return run, revision, receipt, live, contents


class Volume:
    def __init__(self, contents):
        self.contents = contents
        self.calls = []
    def read_file(self, name):
        self.calls.append(name)
        raw = self.contents[name]
        return (raw[start:start+65536] for start in range(0, len(raw), 65536))
    # Intentionally no commit, upload, batch_upload, write_file, App or container.


def main():
    started = datetime.now(UTC).isoformat()
    checked_names = ["scripts/lesson_products/inspect.py", ".github/workflows/lesson-products-inspect.yml", "scripts/lesson_products/checkpoints.py", "scripts/lesson_products/modal_runner.py", "scripts/selftrained/modal_runner.py", "scripts/selftrained/finance.py", inspection.POLICY_PATH]
    before = {name: sha((ROOT / name).read_bytes()) for name in checked_names}
    results = []
    raw_observation = None
    with tempfile.TemporaryDirectory(dir=ROOT / "work", prefix="inspection-audit-") as temporary:
        work = Path(temporary)
        for status, names in (("completed", TRANSFER_FILES), ("failed", ("last.pt", "progress.json")), ("failed-or-cancelled", ())):
            run, revision, receipt, live, contents = fixture(status, names)
            original_live, original_contents = copy.deepcopy(live), contents.copy()
            volume = Volume(contents)
            destination = work / status
            result = inspection.inspect_attempt(volume, live, run, revision, destination)
            assert result["status"] == status and result["receipt_canonical_sha256"] == canonical_sha(receipt)
            assert live == original_live and contents == original_contents
            assert volume.calls == [f"lesson-products/{run}/execution.json", *[f"lesson-products/{run}/{name}" for name in names]]
            assert not any(result[key] for key in ("gpu_started_by_inspection", "remote_container_started", "ledger_written"))
            for name in names: assert (destination / "files" / name).read_bytes() == contents[f"lesson-products/{run}/{name}"]
            assert json.loads((destination / "execution.json").read_bytes()) == receipt
            if status == "completed":
                saved_sha = sha((destination / "execution.json").read_bytes())
                raw_observation = {"remote_raw_sha256": result["execution_sha256"], "saved_json_sha256": saved_sha,
                    "same_raw_bytes": result["execution_sha256"] == saved_sha, "same_canonical_receipt": canonical_sha(json.loads((destination / "execution.json").read_bytes())) == result["receipt_canonical_sha256"], "note": "execution.json is reserialized by write_json, not byte-exact remote raw; canonical ledger binding is retained."}
            results.append({"id": "terminal_" + status + "_positive", "passed": True, "files": len(names), "read_paths": volume.calls, "ledger_and_volume_unchanged": True})

        for kind in ("unsafe_run", "not_reserved", "running", "wrong_experiment", "receipt_tamper", "receipt_wrong_run", "receipt_wrong_revision", "entry_wrong_revision", "status_mismatch", "source_mismatch", "unsafe_artifact", "unlisted_artifact", "total_oversize", "artifact_overlong", "artifact_short", "artifact_badsha", "receipt_oversize"):
            run, revision, receipt, live, contents = fixture()
            base_volume = Volume(contents)
            inspection.inspect_attempt(base_volume, live, run, revision, work / (kind + "-valid-base"))
            entry = live["ledger"]["reservations"][-1]
            if kind == "unsafe_run": run = "../budget.json"
            elif kind == "not_reserved": live["ledger"]["reservations"].pop()
            elif kind == "running": entry["status"] = "running"
            elif kind == "wrong_experiment": entry["experiment_id"] = "selftrained_unrelated"
            elif kind == "receipt_tamper": receipt["unbound_extra"] = True
            elif kind == "receipt_wrong_run": receipt["run_id"] = "other-run"
            elif kind == "receipt_wrong_revision": receipt["code_revision"] = "c" * 40
            elif kind == "entry_wrong_revision": entry["revision"] = "c" * 40
            elif kind == "status_mismatch": receipt["status"] = "failed"
            elif kind == "source_mismatch": receipt["source_hashes"] = {"other.py": "a" * 64}
            elif kind == "unsafe_artifact": receipt["client_artifacts"]["../budget.json"] = {"sha256": "a" * 64, "bytes": 1}
            elif kind == "unlisted_artifact": receipt["client_artifacts"]["not-allowed.txt"] = {"sha256": "a" * 64, "bytes": 1}
            elif kind == "total_oversize": receipt["client_artifacts"]["best.pt"]["bytes"] = 20 * 1024 * 1024 + 1
            elif kind == "artifact_overlong": contents[f"lesson-products/{run}/best.pt"] += b"unexpected extra bytes"
            elif kind == "artifact_short": contents[f"lesson-products/{run}/best.pt"] = b""
            elif kind == "artifact_badsha": receipt["client_artifacts"]["best.pt"]["sha256"] = "0" * 64
            if kind not in ("receipt_tamper", "unsafe_run", "not_reserved"):
                entry["receipt_sha256"] = canonical_sha(receipt)
            if kind != "unsafe_run":
                contents[f"lesson-products/{run}/execution.json"] = json.dumps(receipt).encode()
            if kind == "receipt_oversize": contents[f"lesson-products/{run}/execution.json"] = b" " * (4 * 1024 * 1024 + 1)
            original_live, original_contents = copy.deepcopy(live), contents.copy()
            volume = Volume(contents)
            caught = None
            try:
                inspection.inspect_attempt(volume, live, run, revision, work / (kind + "-negative"))
            except ValueError as error:
                caught = {"type": type(error).__name__, "message": str(error)}
            assert caught, kind
            assert live == original_live and contents == original_contents
            assert all(path.startswith("lesson-products/synthetic-terminal-attempt/") for path in volume.calls)
            if kind in ("unsafe_run", "not_reserved", "running", "wrong_experiment"): assert not volume.calls
            if kind in ("unsafe_artifact", "unlisted_artifact", "total_oversize"): assert len(volume.calls) == 1
            results.append({"id": "reject_" + kind, "base_passed": True, "passed": True, "error": caught, "read_paths": volume.calls, "ledger_and_volume_unchanged": True})

        # Run actual main through an SDK with only Volume lookup/read_file available.
        run, revision, receipt, live, contents = fixture()
        volume = Volume(contents)
        lookups, reads = [], []
        def lookup(name, create_if_missing):
            lookups.append({"volume": name, "create_if_missing": create_if_missing})
            assert name == "tiny-perceptron-course" and create_if_missing is False
            return volume
        def read_live(policy, policy_sha):
            reads.append(policy_sha)
            assert policy_sha == inspection.POLICY_SHA
            return live
        sdk = SimpleNamespace(Volume=SimpleNamespace(from_name=lookup))
        stdout = io.StringIO()
        destination = work / "cli"
        argv = ["inspect", "--run-id", run, "--expected-revision", revision, "--output", str(destination)]
        with patch.dict(sys.modules, {"modal": sdk}), patch.object(inspection, "read_live_ledger", side_effect=read_live), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            inspection.main()
        assert len(lookups) == len(reads) == 1
        summary = json.loads(stdout.getvalue())
        assert summary == {"run_id": run, "status": "completed", "files": len(TRANSFER_FILES), "remote_container_started": False}
        results.append({"id": "main_readonly_sdk_positive", "passed": True, "lookup": lookups, "fixed_policy_sha": reads, "stdout": summary, "no_mutation_or_compute_methods_available": True})
    assert before == {name: sha((ROOT / name).read_bytes()) for name in checked_names}
    report = {"reviewer_task": "/root/tech_20261016_review_tools", "context": "informed engineering original", "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform()}, "checked_files": [{"path": p, "sha256": digest} for p, digest in before.items()], "results": results, "all_expected": True, "raw_receipt_reserialization_observation": raw_observation, "limits": ["No actual Volume, billing service, Modal container, paid compute, HF write, model or GPU; main ledger provider is mocked and actual read_live_ledger implementation was inspected separately.", "Fixtures use real finish_entry to form canonical ledger receipt bindings and explicit synthetic artifact bytes, not real model results.", "Terminal ledger/receipt consistency does not prove a running cloud call has stopped independently; this command neither polls nor cancels calls."]}
    path = ART / "engineering-inspection-probes-results-original.json"
    with path.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"cases": len(results), "all_expected": True, "receipt_sha256": sha(path.read_bytes()), "raw_receipt_preserved": raw_observation["same_raw_bytes"]}))


if __name__ == "__main__": main()
