"""Independent, offline falsification probes for the immutable P1 initial source.

These assertions reproduce observed behavior; a PASS can confirm a defect.
All Modal/GitHub/HF mutations in this file are in-memory fakes.
"""

import ast
import base64
import hashlib
import io
import json
import os
import random
import signal
import struct
import sys
import tarfile
import types
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE_ROOT = Path(os.environ.get("OMNI_REVIEW_SOURCE_ROOT", "/tmp/omni-infra-review-initial-20261029"))
sys.path.insert(0, str(SOURCE_ROOT))
from scripts.omni_revision import finance as f
from scripts.omni_revision import preflight as p

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)
REV = "1" * 40


def snapshot(gross="10", when=NOW):
    return {
        "source": f.SOURCE,
        "workspace": "birdhackor",
        "observed_at": when.isoformat(),
        "summaries": [{"cycle": "2026-10", "start": "2026-10-01T00:00:00+00:00",
                       "end": "2026-11-01T00:00:00+00:00", "metered_cost_usd": gross}],
    }


def inventory(carry="0"):
    return {
        "workspace": "birdhackor", "environment": "main", "observed_at": NOW.isoformat(),
        "ledger_sha256": "a" * 64, "quota_sha256": "b" * 64,
        "terminated_pending": [] if carry == "0" else [
            {"run_id": "historical-fully-metered", "status": "completed", "reserved_usd": carry}],
        "unknown_attempts": [], "active_apps": [], "errors": [],
    }


def components(amount):
    return dict(compute=amount, startup_idle="0", build="0", egress="0", storage_cleanup="0")


def reserve(state, gross="10", amount="1", attempt="probe"):
    return f.reserve(state, snapshot(gross), inventory("100"), attempt, components(amount), REV,
                     "independent audit", now=NOW)


def test_reproduce_gross_over_60_and_history_misclassified_as_carry():
    state = f.new_state(snapshot(), inventory("100"), REV, now=NOW)
    accepted = reserve(state, gross="90")
    assert accepted["reserved_total_usd"] == "1"
    assert accepted["legacy_carry_usd"] == "100"
    assert float(accepted["attempts"][-1]["billing_snapshot"]["summaries"][0]["metered_cost_usd"]) - 10 > 60


def test_reproduce_fully_spent_round_still_reserves_when_official_plus_pending_exceeds_60():
    state = f.new_state(snapshot(), inventory("100"), REV, now=NOW)
    state = reserve(state, amount="59", attempt="first")
    state = f.finish(state, "first", {"success": True, "compute_stopped": True, "volume_deleted": True})
    following = reserve(state, gross="69", amount="1", attempt="second")
    assert following["reserved_total_usd"] == "60"
    assert 59 + 59 + 1 > 60  # official increment + retained prior bound + proposal, excluding carry


@pytest.mark.parametrize("bad", [True, "NaN", "sNaN", "Infinity", "-Infinity", "-0.1"])
def test_official_gross_invalid_values_are_rejected(bad):
    with pytest.raises((ValueError, ArithmeticError)):
        f.new_state(snapshot(bad), inventory(), REV, now=NOW)


def test_stale_future_missing_cycle_and_duplicate_cycle_fail_closed():
    for when in (NOW - timedelta(seconds=121), NOW + timedelta(seconds=1)):
        with pytest.raises(ValueError):
            f.new_state(snapshot(when=when), inventory(), REV, now=NOW)
    for rows in ([], snapshot()["summaries"] * 2):
        bad = snapshot()
        bad["summaries"] = rows
        with pytest.raises(ValueError):
            f.new_state(bad, inventory(), REV, now=NOW)


def test_reproduce_missing_highwater_history_is_not_detected():
    state = f.new_state(snapshot(), inventory("100"), REV, now=NOW)
    state["month_gross_highwater_usd"] = {}
    following = reserve(state, gross="9")
    assert following["month_gross_highwater_usd"] == {"2026-10": "9"}


def test_reproduce_hf_cleanup_failure_is_terminal_for_next_paid_attempt():
    state = f.new_state(snapshot(), inventory("100"), REV, now=NOW)
    state = reserve(state)
    state = f.finish(state, "probe", {"success": False, "compute_stopped": True,
                                    "volume_deleted": True, "hf_branch_created": True,
                                    "hf_branch_deleted": False})
    assert state["attempts"][0]["status"] == "failed-stopped"
    assert reserve(state, attempt="next")["reserved_total_usd"] == "2"


def test_reproduce_inventory_accepts_unvalidated_quota_and_missing_termination_evidence(tmp_path, monkeypatch):
    ledger = {"reservations": [{"run_id": "legacy", "status": "completed", "reserved_usd": "48.4"}]}
    raw = {"budget.json": json.dumps(ledger).encode(), "gross-quota.json": b"{}"}
    class Volume:
        objects = types.SimpleNamespace(list=lambda **kw: [])
        @staticmethod
        def from_name(*args, **kwargs):
            return types.SimpleNamespace(read_file=lambda name: iter([raw[name]]))
    monkeypatch.setattr(p, "app_rows", lambda: [])
    actual = p.inventory(types.SimpleNamespace(Volume=Volume), tmp_path)
    accepted = f.new_state(snapshot(when=datetime.now(UTC)), actual, REV)
    assert accepted["legacy_carry_usd"] == "48.4"
    assert accepted["legacy_inventory"]["terminated_pending"][0]["app_id"] is None


def test_main_cas_failure_precedes_any_paid_boundary(tmp_path, monkeypatch):
    events = []
    state = f.new_state(snapshot(), inventory(), REV, now=NOW)
    class Store:
        def read(self):
            return state, "old-sha"
        def write(self, *args):
            events.append("CAS-rejected")
            raise RuntimeError("independent fake CAS conflict")
    fake = types.SimpleNamespace(Workspace=types.SimpleNamespace(from_context=lambda: types.SimpleNamespace(
        billing=types.SimpleNamespace(rates=lambda: {"cpu_hour_cost": "0.0472", "mem_gib_hour_cost": "0.008"}))))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(p, "GitHubLedger", Store)
    monkeypatch.setattr(p, "inventory", lambda *args: inventory())
    monkeypatch.setattr(p, "official_snapshot", lambda *args: snapshot())
    monkeypatch.setattr(p, "source_revision", lambda: REV)
    monkeypatch.setattr(p, "reserve", lambda *args: {"reserved": "fake"})
    monkeypatch.setattr(p, "run_cpu_probe", lambda *args: events.append("PAID"))
    monkeypatch.setenv("HF_RELEASE_REPO", "review/dummy")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.setattr(sys, "argv", ["preflight", "hf-probe", "--attempt-id", "new", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="CAS conflict"):
        p.main()
    assert events == ["CAS-rejected"]


def test_reproduce_volume_name_check_then_create_if_missing_can_delete_foreign_volume(monkeypatch):
    events = []
    own_name = "tiny-perceptron-omni-probe-" + hashlib.sha256(b"race").hexdigest()[:16]
    class Objects:
        def list(self, **kwargs):
            return []  # prior list was empty; another actor can create before lazy hydration
        def delete(self, name, **kwargs):
            events.append(("deleted-by-name", name))
    class Volume:
        objects = Objects()
        @staticmethod
        def from_name(name, **kwargs):
            events.append(("lazy-get-or-create", name, kwargs["create_if_missing"]))
            return types.SimpleNamespace()  # actual Modal 1.6 allows binding an existing Volume
    class App:
        app_id = "ap-review-fake"
        def __init__(self, *args): pass
        def function(self, **kwargs):
            return lambda fn: types.SimpleNamespace(remote=lambda: {"success": False})
        def run(self, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
    modal = types.SimpleNamespace(App=App, Volume=Volume,
        Secret=types.SimpleNamespace(from_name=lambda *a, **k: object()),
        Image=types.SimpleNamespace(debian_slim=lambda **k: object()))
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(p, "app_rows", lambda: [{"App ID": "ap-review-fake", "State": "stopped", "Tasks": "0"}])
    result = p.run_cpu_probe(modal, "race", "review/dummy", "dummy")
    assert ("lazy-get-or-create", own_name, True) in events
    assert ("deleted-by-name", own_name) in events
    assert result["volume_deleted"] is True


def extracted_remote(directory, volume):
    tree = ast.parse((SOURCE_ROOT / "scripts/omni_revision/preflight.py").read_text())
    remote = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "remote_probe")
    remote = deepcopy(remote)
    remote.decorator_list = []
    for node in ast.walk(remote):
        if isinstance(node, ast.Constant) and node.value == "/probe":
            node.value = str(directory)
    module = ast.fix_missing_locations(ast.Module(body=[remote], type_ignores=[]))
    ns = {"attempt_id": "local-review", "repo": "review/dummy", "volume": volume}
    exec(compile(module, "<extracted-initial-remote-only-local-I/O>", "exec"), ns)
    return ns["remote_probe"]


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def test_hf_protocol_roundtrip_bytes_and_cross_host_authorization_stripping(tmp_path, monkeypatch):
    calls, files, handlers = [], {}, []
    commits = []
    original = "2" * 40
    commit = "3" * 40
    deleted = False
    class Opener:
        def open(self, request, timeout):
            nonlocal deleted
            calls.append((request.method, request.full_url))
            path = request.full_url.split("https://huggingface.co", 1)[-1]
            if request.method == "DELETE":
                deleted = True
                data = b""
            elif "/revision/main" in path:
                data = json.dumps({"sha": original}).encode()
            elif "/revision/omni-preflight-" in path and deleted:
                raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
            elif "/branch/" in path:
                data = b""
            elif "/preupload/" in path:
                payload = json.loads(request.data)
                data = json.dumps({"files": [dict(path=v["path"], uploadMode="regular", shouldIgnore=False)
                                             for v in payload["files"]]}).encode()
            elif "/commit/" in path:
                records = [json.loads(line) for line in request.data.splitlines()]
                assert records[0]["value"]["parentCommit"] == original
                for item in records[1:]:
                    files[item["value"]["path"]] = base64.b64decode(item["value"]["content"])
                commits.append(commit)
                data = json.dumps({"commitOid": commit}).encode()
            elif "/resolve/" in path:
                assert f"/resolve/{commit}/" in path
                data = files[path.split(f"/resolve/{commit}/", 1)[1]]
            else:
                raise AssertionError(path)
            return FakeResponse(data)
    def build(handler):
        handlers.append(handler)
        return Opener()
    monkeypatch.setenv("HF_TOKEN", "INDEPENDENT_DUMMY_NO_REAL_CREDENTIAL")
    monkeypatch.setattr(urllib.request, "build_opener", build)
    remote = extracted_remote(tmp_path, types.SimpleNamespace(commit=lambda: None))
    before = signal.getsignal(signal.SIGALRM)
    try:
        result = remote()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)
    assert result["success"] is True and result["hf_branch_deleted"] is True
    assert struct.unpack("<2d", files["probe/weights.bin"])
    with tarfile.open(fileobj=io.BytesIO(files["probe/resume.tar"])) as archive:
        resume = json.load(archive.extractfile("resume.json"))
    state = resume["rng"]
    restored = random.Random()
    restored.setstate((state[0], tuple(state[1]), state[2]))
    assert resume["optimizer"]["step"] == 2
    request = urllib.request.Request("https://huggingface.co/x", headers={"Authorization": "Bearer DUMMY"})
    redirected = handlers[0].redirect_request(request, None, 302, "", {}, "https://cdn.example.test/file")
    assert redirected.get_header("Authorization") is None
    assert all("Bearer" not in json.dumps(item) for item in [result, calls])


def test_reproduce_hf_successful_branch_creation_with_lost_response_is_left_unresolved(tmp_path, monkeypatch):
    created, deleted = [], []
    class Opener:
        def open(self, request, timeout):
            if "/revision/main" in request.full_url:
                return FakeResponse(json.dumps({"sha": "4" * 40}).encode())
            if "/branch/" in request.full_url and request.method == "POST":
                created.append(True)  # server mutation succeeded, response never reached client
                raise TimeoutError("dummy response loss")
            if request.method == "DELETE":
                deleted.append(True)
            raise AssertionError("unexpected request")
    monkeypatch.setenv("HF_TOKEN", "INDEPENDENT_DUMMY_NO_REAL_CREDENTIAL")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    remote = extracted_remote(tmp_path, types.SimpleNamespace(commit=lambda: None))
    before = signal.getsignal(signal.SIGALRM)
    try:
        result = remote()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)
    assert created and not deleted
    assert result["hf_branch_created"] is False and result["hf_branch_deleted"] is False
