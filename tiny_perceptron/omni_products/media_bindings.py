"""把 Auto 的四個舊 A 媒體案例綁到固定資料包；不依模型成績挑題。"""

import hashlib
import json
from pathlib import Path

from tiny_perceptron.selftrained.tools import ToolCallError, parse_tool_call

from .data import av_pair, fingerprint, split_a

OLD_A = {
    "repo": "birdhackor/tiny-perceptron-course-models",
    "revision": "979cdfacc588ad0536f1c64fff96f264571cf054",
    "prefix": "selftrained/v2/moe-joint",
    "manifest_sha256": "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e",
}
OLD_B = {
    "repo": "birdhackor/tiny-perceptron-course-models",
    "revision": "2c6a1062f3e66a602cf83eda6c703456ec18bca0",
    "prefix": "lesson-products/v1-20261016/b",
    "manifest_sha256": "dd077fd7a0f11fa5d95bdcf76a73bf6e035819ab135fa3191ff94f4e3643a1a3",
}
MEDIA_TASKS = {
    "clothing": ("vision-test.jsonl", "vision_clothing"),
    "relation": ("vision-test.jsonl", "vision_relation"),
    "ocr": ("ocr-test.jsonl", "ocr"),
    "voice": ("voice-test.jsonl", "voice_qa"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(data_root, manifest, name):
    expected = next(row for row in manifest["records"] if row["path"] == name)
    path = Path(data_root) / name
    if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
        raise ValueError("固定舊 A 記錄與資料 manifest 不符")
    return [json.loads(line) for line in path.read_text().splitlines()]


def public_input(record):
    if record["messages"][-1]["role"] != "assistant":
        raise ValueError("舊固定記錄缺少末尾標準答案")
    task = record["task"]
    messages = record["messages"][:-1]
    if task == "tool_reply":
        # 訓練中的正確 assistant call／tool result 不是當前公開輸入。
        # 從原user問題起跑，讓固定舊A自己產call並接真工具結果。
        last_user = max(i for i, message in enumerate(messages) if message["role"] == "user")
        messages = messages[: last_user + 1]
    return {
        "messages": messages,
        "task": "tool_call" if task == "tool_reply" else task,
        "tools": task in {"tool_call", "tool_reply"},
        "metadata": {
            key: record[key]
            for key in ("image", "audio", "roi", "image_layout", "modality_message_index")
            if key in record
        },
    }


def bind_media(data_root, manifest):
    selected = {}
    assets = {row["path"]: row for row in manifest["assets"]}
    for name, (filename, task) in MEDIA_TASKS.items():
        record = min(
            (row for row in records(data_root, manifest, filename) if row["task"] == task), key=lambda row: row["id"]
        )
        bound = {
            "selection": "lexicographically first fixed old-test record ID; no model predictions",
            "source_record": record["id"],
            "parent": record["group_id"],
            "record_file": filename,
            "record_file_sha256": next(row["sha256"] for row in manifest["records"] if row["path"] == filename),
            "input": public_input(record),
            "gold": record["messages"][-1]["content"],
            "assets": {},
        }
        for key in ("image", "audio"):
            if key in record:
                path = record[key]
                expected = assets[path]
                if sha(Path(data_root) / path) != expected["sha256"]:
                    raise ValueError("Auto 使用的固定媒體 SHA 不符")
                bound["assets"][path] = expected
        selected[name] = bound
    # 這題明記與 A200 的共同祖先，不能把兩處評分加成獨立樣本。
    pattern, variant, panel, which = split_a()["final"][0], 73, 1, 0
    _, label = av_pair(pattern, variant, panel)[which]
    selected["video-audio"] = {
        "selection": "predeclared first A-final ancestor with secondary variant; not an independent AV family",
        "parent": ["A-pattern", pattern],
        "pattern": pattern,
        "variant": variant,
        "panel": panel,
        "which": which,
        "gold": ("circle", "square")[label],
        "shared_A_final_family_count": 1,
    }
    return {
        "schema": "omni-auto-media-bindings-v1",
        "old_A": OLD_A,
        "old_B": OLD_B,
        "media": selected,
        "binding_sha256": fingerprint(selected),
        "model_predictions_used": False,
    }


def retention_panel(data_root, manifest, families_per_task=30):
    if type(families_per_task) is not int or families_per_task < 1:
        raise ValueError("retention 家族數須為正整數")
    grouped = {}
    for name in ("text-tools-test.jsonl", "vision-test.jsonl", "ocr-test.jsonl", "voice-test.jsonl"):
        for record in sorted(records(data_root, manifest, name), key=lambda row: row["id"]):
            task = record["task"]
            if task == "tool_reply" and record.get("supervision", {}).get("replay_kind") != "actual_executor":
                continue  # 反事實工具結果另評；不能拿其gold評真實算術往返。
            grouped.setdefault(task, {}).setdefault(record["group_id"], record)
    rows = []
    for task, families in sorted(grouped.items()):
        for family, record in sorted(families.items())[:families_per_task]:
            gold_call = None
            if task == "tool_call":
                gold_call = json.loads(record["messages"][-1]["content"])
            elif task == "tool_reply":
                candidates = [
                    message["content"]
                    for message in record["messages"][:-1]
                    if message["role"] == "assistant" and message["content"].lstrip().startswith("{")
                ]
                gold_call = json.loads(candidates[-1])
            rows.append(
                {
                    "task": task,
                    "family": family,
                    "record": record["id"],
                    "input": public_input(record),
                    "gold": record["messages"][-1]["content"],
                    "scoring_field": "tool_trace.initial_output (strict parsed call)"
                    if task == "tool_call"
                    else "answer",
                    "gold_tool_call": gold_call,
                    "requires_actual_tool_round": task == "tool_reply",
                    "metric_scope": "Live user/model/tool/model round, not teacher-forced internal tool-reply context"
                    if task == "tool_reply"
                    else "Original task target at its declared output stage",
                    "source_replay_kind": record.get("supervision", {}).get("replay_kind"),
                }
            )
    return rows


def retention_result(row, response):
    """標準答案只在這個評分端使用，不能流回公開輸入／模型。"""
    if row["task"] == "tool_call":
        actual = response["tool_trace"]["initial_output"]
        try:
            correct = parse_tool_call(actual).to_dict() == row["gold_tool_call"]
        except ToolCallError:
            correct = False
    elif row["task"] == "tool_reply":
        actual = response["answer"]
        trace = response["tool_trace"]
        correct = trace["executed"] and trace["tool_call"] == row["gold_tool_call"] and actual == row["gold"]
    else:
        actual = response["answer"]
        correct = actual == row["gold"]
    return {"correct": bool(correct), "actual": actual, "scoring_field": row["scoring_field"]}


def counterfactual_tool_panel(data_root, manifest):
    """保留舊受控工具結果診斷，與真正的計算器往返分開。"""
    selected = {}
    for record in sorted(records(data_root, manifest, "text-tools-test.jsonl"), key=lambda row: row["id"]):
        if (
            record["task"] == "tool_reply"
            and record.get("supervision", {}).get("replay_kind") == "counterfactual_test_only"
        ):
            selected.setdefault(record["group_id"], record)
    return [
        {
            "family": family,
            "record": record["id"],
            "controlled_prefix": record["messages"][:-1],
            "gold": record["messages"][-1]["content"],
            "source_replay_kind": "counterfactual_test_only",
            "actual_executor_result": record["supervision"]["actual_result"],
            "provided_tool_result": record["supervision"]["expected_result"],
            "scope": "Research-only controlled continuation with provided altered tool result; not public input or actual arithmetic; shared parents with live retention, not extra independent families",
        }
        for family, record in sorted(selected.items())
    ]
