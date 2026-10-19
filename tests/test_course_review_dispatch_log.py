"""派工紀錄（docs/course-review-20261019/tools/dispatch_log.py）`finished` 事件的離線測試。

驗證：turn.failed 而 Codex 沒寫 -o 輸出時，事件照記、輸出記成 null；turn.completed 卻找不到輸出時拒絕記。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dispatch_log", ROOT / "docs/course-review-20261019/tools/dispatch_log.py"
)
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)

THREAD = "01a15300-d4b6-7250-9b1e-f37408f9426a"
USAGE = {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2, "reasoning_output_tokens": 1}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run_finished(tmp_path, monkeypatch, end):
    monkeypatch.setattr(DISPATCH, "LOG", tmp_path / "dispatch-log.jsonl")
    monkeypatch.setattr(DISPATCH, "ROLLOUT_COPIES", tmp_path / "rollouts")
    monkeypatch.setattr(DISPATCH, "CODEX_SESSIONS", tmp_path / "sessions")
    token_count = {"type": "token_count", "info": {"total_token_usage": USAGE}, "rate_limits": {}}
    rollout = [{"payload": {"type": "task_started", "turn_id": "turn-1", "started_at": 1}}, {"payload": token_count}]
    if end["type"] == "turn.completed":
        rollout.append({"payload": {"type": "task_complete", "completed_at": 2, "duration_ms": 1000}})
    jsonl(tmp_path / f"sessions/2026/10/19/rollout-2026-10-19T15-12-05-{THREAD}.jsonl", rollout)
    log = jsonl(tmp_path / "codex.log", [{"type": "thread.started", "thread_id": THREAD}, end])
    argv = ["dispatch_log.py", "finished", "/root/rv_test", "--log", str(log), "--output", str(tmp_path / "out.md")]
    monkeypatch.setattr(sys, "argv", argv)
    DISPATCH.main()
    return [json.loads(line) for line in (tmp_path / "dispatch-log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_failed_turn_without_output_is_recorded_with_null_output(tmp_path, monkeypatch):
    entries = run_finished(tmp_path, monkeypatch, {"type": "turn.failed", "error": {"message": "at capacity"}})
    assert [(e["event"], e["log_end"], e["output"], e["turn"]["status"]) for e in entries] == [
        ("finished", "turn.failed", None, "incomplete")
    ]


def test_a_completed_turn_without_output_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="-o 輸出"):
        run_finished(tmp_path, monkeypatch, {"type": "turn.completed", "usage": USAGE})
    assert not (tmp_path / "dispatch-log.jsonl").exists()
