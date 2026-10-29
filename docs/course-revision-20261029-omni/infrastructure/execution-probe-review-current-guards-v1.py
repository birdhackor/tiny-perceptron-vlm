"""回核 source/audit guard 與同一 cleanup deadline；不做外部 I/O。"""

import json
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import execution as runner  # noqa: E402

current = datetime.now(UTC)


class Clock:
    @classmethod
    def now(cls, tz):
        return current


def deadline_probe(expire=False, sdk_ok=True):
    global current
    current = datetime.now(UTC)
    deadline = current + timedelta(seconds=0.08)
    receipt = {
        "paid_call_started": True,
        "app_id": "ap-owned",
        "cleanup_deadline_at": deadline.isoformat(),
        "cleanup_checks": {
            "sdk_cleanup_completed_within_deadline": sdk_ok,
            "owned_sandboxes_stopped": True,
        },
    }
    timeouts = []

    def official(command, **kwargs):
        global current
        assert command == ["python", "-m", "modal", "app", "list", "--env", "main", "--json"]
        timeouts.append(kwargs["timeout"])
        current += timedelta(seconds=min(0.02, kwargs["timeout"]))
        state = "running" if expire or len(timeouts) == 1 else "stopped"
        return SimpleNamespace(
            stdout=json.dumps(
                [{"App ID": "ap-owned", "State": state, "Tasks": "0" if state == "stopped" else "1"}]
            ).encode()
        )

    def advance(seconds):
        global current
        current += timedelta(seconds=min(seconds, 0.001))

    with (
        patch.object(runner, "datetime", Clock),
        patch.object(runner.subprocess, "run", official),
        patch.object(runner.time, "sleep", advance),
    ):
        try:
            rows = runner.read_stopped_apps(receipt)
            final = runner.reconcile_stop(receipt, rows, runner.identifiers("probe-123-1"))
            result = {"returned": True, "resources_verified": final["resources_verified"]}
        except TimeoutError:
            result = {"returned": False, "error_type": "TimeoutError"}
    assert all(0 < value <= 0.08 for value in timeouts)
    return {"expire": expire, "sdk_ok": sdk_ok, "timeouts": timeouts, "result": result}


if __name__ == "__main__":
    results = [deadline_probe(), deadline_probe(expire=True), deadline_probe(sdk_ok=False)]
    assert results[0]["result"]["resources_verified"] is True
    assert results[1]["result"]["error_type"] == "TimeoutError"
    assert results[2]["result"]["resources_verified"] is False
    # 重放原先純本地的 audit fixtures；不改寫原探針，也不執行它的舊 lifecycle main。
    original = ROOT / "docs/course-revision-20261029-omni/infrastructure/execution-probe-review-guard-probes-v1.py"
    guards = runpy.run_path(str(original))["audit_probes"]()
    print(json.dumps({"deadline_probes": results, "audit_probes": guards}, indent=2))
