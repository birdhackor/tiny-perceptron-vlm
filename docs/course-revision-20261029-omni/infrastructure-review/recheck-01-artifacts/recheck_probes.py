"""Independent offline recheck, with every external mutation replaced by a fake.

Defect-reproduction tests explicitly assert observed bad behavior. Their passing
result does not approve that behavior. Sources are the frozen recheck candidate.
"""

import hashlib
import io
import json
import os
import random
import signal
import struct
import subprocess
import sys
import tarfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

FROZEN = Path("/tmp/omni-infra-review-recheck-01-20261029")
sys.path.insert(0, str(FROZEN))
from scripts.omni_revision import finance as f
from scripts.omni_revision import github_transport_probe as gt
from scripts.omni_revision import preflight as p

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)
REV = "1" * 40


def snapshot(gross="10"):
    return {"source": f.SOURCE, "workspace": "birdhackor", "observed_at": NOW.isoformat(),
            "summaries": [{"cycle": "2026-10", "start": "2026-10-01T00:00:00+00:00",
                           "end": "2026-11-01T00:00:00+00:00", "metered_cost_usd": gross}]}


def inventory():
    report = {"retained_bounds": [{"run_id": "actual-pending", "status": "completed", "reserved_usd": "0.50"}],
              "carry_usd": "0.50", "historical_audit_total_usd": "100",
              "historical_audit_excluded_from_carry": True,
              "month_gross_highwater_usd": {"2026-10": "9"}}
    reconciliation = {
        "schema": "omni-legacy-reconciliation-v1", "reviewer_task": "/root/independent_recheck",
        "ledger_sha256": "a" * 64, "quota_sha256": "b" * 64, "legacy_policy_sha256": "c" * 64,
        "legacy_verification_sha256": f.digest(report), "existing_volumes_sha256": f.digest([]),
        "retained_bounds": [{"run_id": "actual-pending", "reserved_usd": "0.50",
                             "stop_evidence": [{"path": "docs/independently-reviewed-stop.json", "sha256": "d" * 64}]}],
    }
    return {"workspace": "birdhackor", "environment": "main", "observed_at": NOW.isoformat(),
            "ledger_sha256": "a" * 64, "quota_sha256": "b" * 64, "legacy_policy_sha256": "c" * 64,
            "legacy_verification": report, "reconciliation": reconciliation, "existing_volumes": [],
            "active_apps": [], "unknown_attempts": [], "errors": []}


def initialized():
    return f.new_state(snapshot(), inventory(), REV, now=NOW)


def reserve(state, gross="10", amount="1", attempt="one"):
    components = dict(compute=amount, startup_idle="0", build="0", egress="0", storage_cleanup="0")
    return f.reserve(state, snapshot(gross), inventory(), attempt, components, REV, "independent recheck", now=NOW)


def terminal(state):
    return f.finish(state, "one", {"success": True, "compute_stopped": True, "resources_verified": True,
                                  "cleanup_checks": {"all_actual_resources_gone": True}})


def test_original_gross_80_counterexample_now_rejected_without_using_audit_100_as_carry():
    state = initialized()
    assert state["origin"]["legacy_carry_usd"] == "0.50"
    with pytest.raises(ValueError, match="Unexplained"):
        reserve(state, gross="90")


def test_original_official_59_plus_retained_59_plus_one_now_rejected():
    state = terminal(reserve(initialized(), amount="59"))
    with pytest.raises(ValueError, match="Retained official increment"):
        reserve(state, gross="69", amount="1", attempt="two")


def test_original_deleted_highwater_map_now_rejected():
    state = initialized()
    state["month_gross_highwater_usd"] = {}
    with pytest.raises(ValueError, match="high-water/total"):
        reserve(state, gross="9")


def test_cleanup_failure_remains_unresolved_and_blocks_next_reservation():
    state = reserve(initialized())
    state = f.finish(state, "one", {"success": False, "compute_stopped": True, "resources_verified": True,
                                    "cleanup_checks": {"all_actual_resources_gone": False}})
    assert state["attempts"][0]["status"] == "unresolved"
    with pytest.raises(ValueError, match="running or unresolved"):
        reserve(state, attempt="two")


def readonly_fake():
    raw_ledger = b'{"reservations":[{"run_id":"history","status":"completed","reserved_usd":"48.4"}]}'
    raw_quota = b"{}"
    calls = []
    class Volume:
        objects = SimpleNamespace(list=lambda **kw: [])
        @staticmethod
        def from_name(name, create_if_missing):
            calls.append((name, create_if_missing))
            assert name == "tiny-perceptron-course" and create_if_missing is False
            return SimpleNamespace(read_file=lambda name: iter([raw_ledger if name == "budget.json" else raw_quota]))
    modal = SimpleNamespace(Volume=Volume, Workspace=SimpleNamespace(from_context=lambda: SimpleNamespace(
        billing=SimpleNamespace(rates=lambda: {"cpu_hour_cost": "0.0472", "mem_gib_hour_cost": "0.008"}))))
    return modal, calls, raw_quota


def test_original_empty_quota_terminal_word_cannot_initialize_but_raw_inventory_survives(tmp_path, monkeypatch):
    modal, calls, raw_quota = readonly_fake()
    monkeypatch.setattr(p, "app_rows", lambda: [])
    actual = p.inventory(modal, tmp_path)
    assert actual["errors"] and (tmp_path / "legacy-gross-quota.json").read_bytes() == raw_quota
    with pytest.raises(ValueError, match="reconciled"):
        f.new_state(snapshot(), actual, REV, now=datetime.now(UTC))
    assert calls == [("tiny-perceptron-course", False)]


def test_inventory_main_does_not_need_reconciliation_or_create_any_resource(tmp_path, monkeypatch):
    modal, calls, raw_quota = readonly_fake()
    class Store:
        def read(self): return None, None
        def initialize(self, *a): raise AssertionError("No GitHub write during inventory")
    def forbidden(*args, **kwargs):
        raise AssertionError("Inventory must not depend on reconciliation or committed initialization gate")
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setattr(p, "GitHubLedger", Store)
    monkeypatch.setattr(p, "app_rows", lambda: [])
    monkeypatch.setattr(p, "load_reconciliation", forbidden)
    monkeypatch.setattr(p, "source_revision", forbidden)
    monkeypatch.setattr(p, "official_snapshot", lambda *a: snapshot())
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.setattr(sys, "argv", ["preflight", "inventory", "--output", str(tmp_path)])
    p.main()
    assert (tmp_path / "billing.json").exists() and (tmp_path / "inventory.json").exists()
    assert calls == [("tiny-perceptron-course", False)]


def test_fixture_has_binary_weights_full_optimizer_rng_and_exact_same_environment_continuation(tmp_path):
    path = tmp_path / "fixture.tar.gz"
    expected = gt.make_fixture(path)
    assert 0 < path.stat().st_size < 32768
    gt.verify_fixture(path, expected)
    with tarfile.open(path, "r:gz") as tar:
        files = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    assert len(files["weights.bin"]) == 16
    training = json.loads(files["training-state.json"])
    assert training["next_step"] == 1 and len(training["optimizer"]["velocity"]) == 2
    rng = random.Random()
    rng.setstate(gt.tuples(training["rng"]))
    weights = list(struct.unpack("<2d", files["weights.bin"]))
    gt.update(weights, training["optimizer"], rng)
    assert struct.pack("<2d", *weights) == expected


@pytest.mark.parametrize("alter", ["optimizer", "rng"])
def test_missing_optimizer_or_changed_rng_is_detected_even_with_recomputed_inner_manifest(tmp_path, alter):
    path = tmp_path / "fixture.tar.gz"
    expected = gt.make_fixture(path)
    with tarfile.open(path, "r:gz") as tar:
        files = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    state = json.loads(files["training-state.json"])
    if alter == "optimizer":
        state["optimizer"]["velocity"] = [0.0, 0.0]
    else:
        state["rng"] = random.Random(99).getstate()
    files["training-state.json"] = gt.canonical(state)
    files["manifest.json"] = gt.canonical({name: gt.checksum(raw) for name, raw in files.items() if name != "manifest.json"})
    with tarfile.open(path, "w:gz") as tar:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    with pytest.raises(ValueError, match="Optimizer/RNG continuation"):
        gt.verify_fixture(path, expected)


class FakeGitHub:
    """Only the documented client paths, never a network request."""
    def __init__(self, lose_create_response=False, foreign_identity=False):
        self.release = None
        self.calls = []
        self.archive = None
        self.lose_create_response = lose_create_response
        self.foreign_identity = foreign_identity
    def api(self, endpoint, method="GET", payload=None, allow_missing=False):
        self.calls.append((endpoint, method))
        if endpoint == "releases" and method == "POST":
            self.release = dict(payload, id=41, assets=[])
            if self.foreign_identity:
                self.release["body"] = "belongs to someone else"
            if self.lose_create_response:
                raise TimeoutError("Response lost after successful server mutation")
            return deepcopy(self.release)
        if endpoint.startswith("releases/tags/"):
            assert self.release is None or self.release["draft"] is True
            return None  # REST tag lookup cannot find a pending draft tag
        if endpoint == "releases/41":
            if method == "DELETE":
                self.release = None
                return None
            return deepcopy(self.release)
        raise AssertionError((endpoint, method))
    def command(self, arguments, payload=None, allow_missing=False):
        assert arguments[:2] == ["gh", "release"]
        if arguments[2] == "upload":
            self.archive = Path(arguments[4]).read_bytes()
            self.release["assets"] = [{"name": "checkpoint-fixture.tar.gz", "id": 44}]
        else:
            assert arguments[2] == "download"
            target = Path(arguments[arguments.index("--dir") + 1]) / "checkpoint-fixture.tar.gz"
            target.write_bytes(self.archive)
        return b""


def setup_transport(monkeypatch, server):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", gt.REPOSITORY)
    monkeypatch.setattr(gt, "api", server.api)
    monkeypatch.setattr(gt, "command", server.command)
    monkeypatch.setattr(gt.subprocess, "check_output", lambda *a, **k: REV.encode())


def test_full_mock_draft_roundtrip_exact_continuation_and_owned_cleanup(tmp_path, monkeypatch, capsys):
    server = FakeGitHub()
    setup_transport(monkeypatch, server)
    result = gt.probe(tmp_path)
    assert result["success"] and result["cleanup_completed"] and result["optimizer_rng_exact_continuation"]
    assert result["archive_bytes"] < 32768 and server.release is None
    assert ("releases/41", "DELETE") in server.calls
    stdout = capsys.readouterr().out
    assert "https://" not in stdout and "token" not in stdout.lower()


def test_foreign_identity_is_never_deleted(tmp_path, monkeypatch):
    server = FakeGitHub(foreign_identity=True)
    setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    result = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert not result["cleanup_completed"]
    assert server.release is not None and all(method != "DELETE" for _, method in server.calls)


def test_reproduce_lost_create_response_falsely_claims_draft_cleanup_from_tag_404(tmp_path, monkeypatch):
    server = FakeGitHub(lose_create_response=True)
    setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    result = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert result["cleanup_completed"] is True  # defect: draft still exists below
    assert server.release is not None and server.release["draft"] is True
    assert all(method != "DELETE" for _, method in server.calls)


def test_command_rejection_does_not_expose_stderr_redirect_or_credentials(monkeypatch):
    marker = b"DUMMY_CREDENTIAL https://redirect.example.test/signed?secret=DUMMY"
    monkeypatch.setattr(gt.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr=marker, stdout=marker))
    with pytest.raises(RuntimeError) as caught:
        gt.command(["gh", "api", "dummy"])
    assert "DUMMY" not in str(caught.value) and "https://" not in str(caught.value)
