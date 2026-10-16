"""發布600秒CPU及取消收尾的獨立離線反例。"""

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import publish
from scripts.selftrained.modal_runner import reservation_guard


def main():
    spec = importlib.util.spec_from_file_location("publication_finish_fake_classes", ROOT / "docs/lesson-products/reviews/artifacts/engineering-runner-probes-original.py")
    classes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classes)
    app = classes.FakeApp()
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    class RemoteError(Exception):
        pass
    sdk = SimpleNamespace(App=lambda _: app, Volume=SimpleNamespace(from_name=lambda *a, **k: volume), Secret=SimpleNamespace(from_name=lambda *a, **k: "synthetic"), Image=SimpleNamespace(debian_slim=lambda **k: classes.FakeImage()), exception=SimpleNamespace(FunctionTimeoutError=type("RuntimeTimeout", (Exception,), {}), RemoteError=RemoteError, OutputExpiredError=type("Expired", (Exception,), {})))
    with patch.dict(sys.modules, {"modal": sdk}):
        _, _, finish = publish.register(ROOT / "work/nonexistent-fixture-only.tar.gz")
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT / "work", prefix="publish-finish-audit-") as folder:
        path = Path(folder) / "budget.json"
        for kind in ("poll-timeout", "network", "terminal", "completed-race", "wrong-id", "unrelated"):
            state = {"reserved_total_usd": "10.10", "reservations": [{"run_id": "old", "reserved_usd": "10.00", "status": "completed"}, {"run_id": "audit", "reserved_usd": ".10", "status": "running", "experiment_id": "selftrained_lesson_products_release", "function_call_id": "fc-synthetic"}]}
            if kind == "unrelated":
                state["reservations"][-1]["experiment_id"] = "selftrained_other"
            publish.write_json(path, state)
            events = []
            class Call:
                def cancel(self, terminate_containers):
                    events.append(["cancel", terminate_containers])
                def get(self, timeout):
                    events.append(["get", timeout])
                    if kind == "poll-timeout":
                        raise TimeoutError("still unfinished")
                    if kind == "network":
                        raise ConnectionError("unknown status")
                    if kind == "terminal":
                        raise RemoteError("terminal cancellation result")
                    updated = json.loads(path.read_bytes())
                    updated["reservations"][-1]["status"] = "completed"
                    publish.write_json(path, updated)
                    return {"status": "completed"}
            sdk.FunctionCall = SimpleNamespace(from_id=lambda _: Call())
            caught = None
            with patch.dict(sys.modules, {"modal": sdk}), patch.object(publish, "LEDGER", path):
                try:
                    finish("audit", "fc-wrong" if kind == "wrong-id" else "fc-synthetic")
                except Exception as error:
                    caught = {"type": type(error).__name__, "message": str(error)}
            after = json.loads(path.read_bytes())
            status = after["reservations"][-1]["status"]
            if kind in ("poll-timeout", "network", "wrong-id", "unrelated"):
                assert caught and status == "running"
            else:
                assert caught is None and status == ("completed" if kind == "completed-race" else "failed-or-cancelled")
            assert after["reservations"][0] == state["reservations"][0] and after["reserved_total_usd"] == "10.10"
            results.append({"id": kind, "status": status, "error": caught, "events": events, "prior_entry_and_total_unchanged": True})
    guard = reservation_guard("release", {"cpu_hour_cost": ".04730", "mem_gib_hour_cost": ".00800", "egress_gib_cost": "0"}, 600)
    assert guard["resource_spec"] == {"cpu": 1, "memory_gib": 2, "seconds": 600, "gpu": False}
    output = {"reviewer_task": "/root/tech_20261016_review_tools", "results": results, "all_expected": True, "release_guard_fixture": guard, "limits": ["Fake SDK calls only; pure guard fixture rates are not current observed rates or available account balance.", "Actual publish CPU600 + optional finish CPU180 are <= original compute600 + two control180 bounds."]}
    path = Path(__file__).with_name("engineering-publication-finish-results-original.json")
    with path.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"probes": len(results), "all_expected": True, "reserved_bound_fixture_usd": guard["reserved_usd"], "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
