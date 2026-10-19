"""記下每次派出或續談 Codex 審閱任務的實際時間、身分與用量；不產生審閱判斷。

每次派工前跑 `before`，派出後跑 `dispatched`，續談後跑 `resumed`；每一次（派工或續談）結束後
跑 `finished`。四種事件都追加到 docs/course-review-20261019/dispatch-log.jsonl，不改寫既有行。
派工時間窗取 `before` 與 `dispatched` 兩個實際時鐘之間；它只界定派工發生的區間，不聲稱精確的
啟動時刻。

`finished` 讀 Codex 原生 rollout 的最後一個 turn，複製一份到 outputs/course-review-20261019/
rollouts/（不進 Git），記下路徑與 SHA-256。
驗證：這個 turn 的起訖時間、累計 token 的前後差、帳號用量百分比的讀數，以及 exec 程式碼裡
`tools.<名稱>(` 形式的工具呼叫次數與 `tools.view_image` 的圖路徑。
不涵蓋：帳號用量百分比是整個帳號同時段的讀數，有別的任務並行時不能全算在這一個任務上；
工具呼叫是從程式碼字面辨認，寫成別種形式的呼叫數不到，`view_image_unparsed` 記下數到呼叫
卻讀不出路徑的次數。
"""

import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "docs/course-review-20261019/dispatch-log.jsonl"
ROLLOUT_COPIES = ROOT / "outputs/course-review-20261019/rollouts"
CODEX_SESSIONS = Path.home() / ".codex/sessions"
TASK = re.compile(r"/root/[A-Za-z0-9_-]+")
TOOL_CALL = re.compile(r"\btools\.([A-Za-z_][A-Za-z0-9_]*)\(")
VIEW_IMAGE = re.compile(r"\btools\.view_image\(\s*\{\s*path\s*:\s*\"((?:[^\"\\]|\\.)*)\"")
USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
META_KEYS = (
    "job_id",
    "kind",
    "name",
    "repo_root",
    "launched_at",
    "prompt_sha256",
    "workorder_prefix_sha256",
    "workorder_suffix_sha256",
    "model",
    "effort",
    "service_tier",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_meta(jobdir):
    """launch.meta 是 NUL 分隔的 key／value 交錯序列。"""
    parts = (Path(jobdir) / "launch.meta").read_bytes().decode().split("\0")
    pairs = dict(zip(parts[0::2], parts[1::2]))
    return {key: pairs.get(key, "") for key in META_KEYS}


def json_events(path):
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def first_thread(log):
    for event in json_events(log):
        if event.get("type") == "thread.started":
            return event.get("thread_id")
    raise ValueError("log 裡沒有 thread.started：" + str(log))


def log_turn_end(log):
    """`--json` log 最後一個 turn 的結束事件；turn.completed 的 usage 是整個 thread 的累計。"""
    ends = [e for e in json_events(log) if e.get("type") in ("turn.completed", "turn.failed")]
    if not ends:
        raise SystemExit("log 裡沒有 turn.completed 或 turn.failed，這一次還沒結束：" + str(log))
    return ends[-1]


def rollout_of(thread):
    found = sorted(CODEX_SESSIONS.glob(f"*/*/*/rollout-*-{thread}.jsonl"))
    if len(found) != 1:
        raise SystemExit(f"thread {thread} 的原生 rollout 應該恰好一個，找到 {len(found)} 個")
    return found[0]


def payload(event):
    return event.get("payload") or {}


def usage_of(event):
    info = payload(event).get("info") or {}
    usage = info.get("total_token_usage") or {}
    return {key: int(usage.get(key, 0)) for key in USAGE_KEYS}


def rate_reading(event):
    limits = payload(event).get("rate_limits") or {}
    reading = {"at": event.get("timestamp"), "limit_id": limits.get("limit_id")}
    for window in ("primary", "secondary"):
        value = limits.get(window)
        if value:
            reading[window] = {key: value.get(key) for key in ("used_percent", "window_minutes", "resets_at")}
    return reading


def last_turn(rollout):
    """rollout 裡最後一個 task_started 到檔尾的那一段。"""
    events = json_events(rollout)
    starts = [i for i, e in enumerate(events) if payload(e).get("type") == "task_started"]
    if not starts:
        raise SystemExit("rollout 裡沒有 task_started：" + str(rollout))
    start = starts[-1]
    counts = [e for e in events[:start] if payload(e).get("type") == "token_count" and payload(e).get("info")]
    before = usage_of(counts[-1]) if counts else dict.fromkeys(USAGE_KEYS, 0)
    segment = events[start:]
    readings = [e for e in segment if payload(e).get("type") == "token_count"]
    with_usage = [e for e in readings if payload(e).get("info")]
    after = usage_of(with_usage[-1]) if with_usage else before
    complete = [e for e in segment if payload(e).get("type") == "task_complete"]
    tools, images, view_calls = {}, [], 0
    for event in segment:
        body = payload(event)
        if body.get("type") == "custom_tool_call":
            code = body.get("input") or ""
            for name in TOOL_CALL.findall(code):
                tools[name] = tools.get(name, 0) + 1
            view_calls += code.count("tools.view_image(")
            images.extend(json.loads(f'"{path}"') for path in VIEW_IMAGE.findall(code))
        elif body.get("type") == "function_call":
            name = body.get("name") or ""
            tools[name] = tools.get(name, 0) + 1
    started = payload(events[start])
    turn = {
        "turn_id": started.get("turn_id"),
        "started_at": started.get("started_at"),
        "status": "complete" if complete else "incomplete",
        "usage_before": before,
        "usage_after": after,
        "usage_delta": {key: after[key] - before[key] for key in USAGE_KEYS},
        "account_rate_limit_readings": [rate_reading(e) for e in (readings[:1] + readings[-1:])],
        "tool_calls": dict(sorted(tools.items())),
        "view_image_paths": images,
        "view_image_unparsed": view_calls - len(images),
    }
    if complete:
        done = payload(complete[-1])
        turn.update(completed_at=done.get("completed_at"), duration_ms=done.get("duration_ms"))
    return turn


def append(entry):
    entry = {"at": datetime.now(UTC).isoformat(), **entry}
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False))


def repo_prompt(path):
    prompt = path.resolve()
    if not prompt.is_relative_to(ROOT):
        raise SystemExit("提示詞原文要放在 repo 內")
    return {"path": prompt.relative_to(ROOT).as_posix(), "sha256": sha(prompt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="event", required=True)
    before = sub.add_parser("before")
    before.add_argument("task")
    before.add_argument("--role", required=True, choices=("reader", "technical", "continuity"))
    before.add_argument("--wave", required=True)
    scope = before.add_mutually_exclusive_group(required=True)
    scope.add_argument("--pages", nargs="+")
    scope.add_argument("--lesson-id")
    before.add_argument("--prompt", required=True, type=Path, help="repo 內提示詞原文")
    dispatched = sub.add_parser("dispatched")
    dispatched.add_argument("task")
    dispatched.add_argument("--jobdir", required=True, type=Path)
    resumed = sub.add_parser("resumed")
    resumed.add_argument("task")
    resumed.add_argument("--thread", required=True)
    resumed.add_argument("--prompt", required=True, type=Path)
    resumed.add_argument("--log", required=True, type=Path, help="這次續談的 --json 輸出")
    resumed.add_argument("--output", required=True, type=Path, help="這次續談的 -o 輸出")
    resumed.add_argument("--model", required=True)
    resumed.add_argument("--effort", required=True)
    resumed.add_argument("--service-tier", required=True)
    finished = sub.add_parser("finished")
    finished.add_argument("task")
    finished.add_argument("--log", required=True, type=Path, help="這一次的 --json 輸出")
    finished.add_argument("--output", required=True, type=Path, help="這一次的 -o 輸出")
    args = parser.parse_args()
    if not TASK.fullmatch(args.task):
        raise SystemExit("task 必須是 /root/ 加上英數字、底線或連字號")
    if args.event == "before":
        entry = {"event": "before", "task": args.task, "role": args.role, "wave": args.wave}
        entry.update({"pages": args.pages} if args.pages else {"lesson_id": args.lesson_id})
        entry["prompt"] = repo_prompt(args.prompt)
        append(entry)
    elif args.event == "dispatched":
        jobdir = args.jobdir.resolve()
        append(
            {
                "event": "dispatched",
                "task": args.task,
                "jobdir": str(jobdir),
                "thread": first_thread(jobdir / "codex.log"),
                "launch_meta": launch_meta(jobdir),
            }
        )
    elif args.event == "resumed":
        if first_thread(args.log) != args.thread:
            raise SystemExit("續談 log 的 thread 與指定的不同")
        append(
            {
                "event": "resumed",
                "task": args.task,
                "thread": args.thread,
                "prompt": repo_prompt(args.prompt),
                "log": {"path": str(args.log.resolve()), "sha256": sha(args.log)},
                "output": {"path": str(args.output.resolve()), "sha256": sha(args.output)},
                "model": args.model,
                "effort": args.effort,
                "service_tier": args.service_tier,
            }
        )
    else:
        thread = first_thread(args.log)
        end = log_turn_end(args.log)
        source = rollout_of(thread)
        turn = last_turn(source)
        ROLLOUT_COPIES.mkdir(parents=True, exist_ok=True)
        copy = ROLLOUT_COPIES / f"{thread}-{turn['turn_id']}.jsonl"
        shutil.copyfile(source, copy)
        log_usage = {key: int((end.get("usage") or {}).get(key, 0)) for key in USAGE_KEYS[:-1]}
        mismatch = {
            key: [log_usage[key], turn["usage_after"][key]]
            for key in log_usage
            if end.get("type") == "turn.completed" and log_usage[key] != turn["usage_after"][key]
        }
        if mismatch:
            raise SystemExit(
                "rollout 最後一個 turn 的累計用量與這份 log 不同，那不是這一次的 turn；"
                "finished 要在這一次結束後、下一次續談前執行。差異（log, rollout）：" + json.dumps(mismatch)
            )
        append(
            {
                "event": "finished",
                "task": args.task,
                "thread": thread,
                "log": {"path": str(args.log.resolve()), "sha256": sha(args.log)},
                "output": {"path": str(args.output.resolve()), "sha256": sha(args.output)},
                "log_end": end.get("type"),
                "rollout": {"source": str(source), "copy": str(copy), "sha256": sha(copy)},
                "turn": turn,
            }
        )


if __name__ == "__main__":
    main()
