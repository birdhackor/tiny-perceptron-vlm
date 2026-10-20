"""保存工具協議的短程式驗證與原始來源讀取收據。"""

import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import applications, tool_choice
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.retrieval import call_tool, tool_loop


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def save_new(name, value):
    with (ARTIFACTS / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def fetch(url):
    command = ["curl", "-fLsS", "--max-time", "45", url]
    process = subprocess.run(command, check=True, capture_output=True)
    return process.stdout, {
        "command": subprocess.list2cmdline(command),
        "url": url,
        "exit_code": process.returncode,
        "received_bytes": len(process.stdout),
        "received_sha256": hashlib.sha256(process.stdout).hexdigest(),
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
    }


paper_bytes, paper_receipt = fetch("https://arxiv.org/html/2302.04761v1")
parser = Text()
parser.feed(paper_bytes.decode("utf-8"))
paper = " ".join(" ".join(parser.parts).split())
paper_quotes = [
    "we interrupt the decoding process, call the appropriate API",
    "allowing the model to make API calls more than doubles performance for all tasks",
]
for quote in paper_quotes:
    assert quote in paper, "原始論文摘錄必須存在於讀取的版本"
assert "Results are always rounded to two decimal places." in paper
assert "inability of Toolformer to use tools in a chain" in paper
paper_receipt.update(
    {
        "version": "arXiv:2302.04761v1，2023-02-09，arXiv HTML",
        "excerpts": paper_quotes,
        "inspected_locators": ["Approach", "Inference", "Tools: Calculator", "Math Datasets", "Limitations"],
        "inspection_note": (
            "Approach 將工具名稱、輸入與結果表成文字；Inference 在請求處暫停解碼，"
            "實際呼叫 API，插入結果後繼續解碼。Calculator 僅支援基本四則，結果取小數兩位。"
            "Math Datasets 在該論文的模型與基準上，允許工具呼叫優於禁用工具。"
            "Limitations 說明工具串接與互動等限制；不據此推論本專案模型的成績或所有模型的優劣。"
        ),
    }
)

docs_bytes, docs_receipt = fetch(
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview.md"
)
docs = docs_bytes.decode("utf-8")
docs_quote = "Your code executes the operation and sends back a `tool_result`."
assert docs_quote in docs, "官方文件摘錄必須存在於讀取的版本"
assert "a second request sends the result back" in docs
docs_receipt.update(
    {
        "version": "官方 Markdown 動態文件，以本收據的讀取日期與回應 SHA-256 識別",
        "excerpts": [docs_quote],
        "inspected_locators": ["How tool use works", "When Claude uses tools", "When required parameters are missing"],
        "inspection_note": (
            "核對 client tools 的模型請求、應用程式執行、tool_result 回填及第二次模型回覆。"
            "文件也區分 server tools；本教材採用自己的本地外層程式。"
            "參數不足時可以追問，但文件不保證模型必然追問。"
            "此文件的 API 格式不直接等同於本專案的 JSON 與 user 訊息回填協議。"
        ),
    }
)
save_new("source-evidence.json", {"original_sources": [paper_receipt, docs_receipt]})

source_path = "docs/course-review-20261019/freeze-01/sources/chapter-0B.md"
freeze_path = "docs/course-review-20261019/freeze-04.json"
body = (ROOT / source_path).read_text(encoding="utf-8")
assert digest(source_path) == "d2426e22eeb0d0bfb8412524ca20400f23c1c4d06a8539060388df5a17aa770d"
assert digest(freeze_path) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-0B")
assert page["snapshot"] == source_path and page["figures_sha256"] == {}
assert "```" not in body and "![" not in body
assert (ROOT / "course/chapters/0B.md").read_text(encoding="utf-8").split("\n## ", 1)[0] + "\n" == body

request = '{"name":"multiply","arguments":{"a":123,"b":45}}'
result = call_tool(request)
assert type(result) is int and result == 123 * (40 + 5) == 4920 + 615 == 5535
rejections = []
for invalid in (
    '{"name":"delete_all","arguments":{"a":2,"b":3}}',
    '{"name":"add","arguments":{"a":true,"b":3}}',
):
    try:
        call_tool(invalid)
    except ValueError as error:
        rejections.append({"input": invalid, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("不允許的工具或 bool 必須被拒絕")

actions = [{"name": "add", "arguments": {"a": 2, "b": 3}}, {"done": True, "answer": 5}]
loop_cases = {}
for name, requests, limit in (
    ("correct_answer", actions, 3),
    ("step_limit", actions, 1),
    ("missing_done", actions[:1], 3),
    ("incorrect_answer_with_done", [actions[0], {"done": True, "answer": 6}], 3),
):
    outcome = tool_loop(requests, max_steps=limit)
    loop_cases[name] = {
        "status": outcome["status"],
        "actual_tool_calls": len(outcome["trace"]),
        "answer": outcome.get("answer"),
        "answer_correct": outcome.get("answer") == 5,
    }
assert loop_cases["correct_answer"] == {
    "status": "done", "actual_tool_calls": 1, "answer": 5, "answer_correct": True
}
assert loop_cases["incorrect_answer_with_done"] == {
    "status": "done", "actual_tool_calls": 1, "answer": 6, "answer_correct": False
}
assert loop_cases["step_limit"]["status"] == "step_limit"
assert loop_cases["missing_done"]["status"] == "needs_more_model_output"


def synthetic_episode(final_answer):
    observed_messages = []
    samples = iter([request, json.dumps({"done": True, "answer": final_answer})])

    def fake_sample(model, messages, ctx, **kwargs):
        observed_messages.append(copy.deepcopy(messages))
        return {
            "samples": [{"generated": next(samples), "invalid_special_tokens": [], "eos": True}],
            "generated_tokens": 1,
        }

    record = {
        "family": "synthetic_multiplication",
        "operation": "multiply",
        "a": 123,
        "b": 45,
        "answer": 5535,
        "messages": [
            {"role": "system", "content": applications._TOOLS_SYSTEM},
            {"role": "user", "content": "CALC:multiply(123,45)"},
        ],
    }
    original_sample = applications._sample
    applications._sample = fake_sample
    try:
        outcome = applications._tool_episode(None, record, None)
    finally:
        applications._sample = original_sample
    assert len(observed_messages) == 2
    assert observed_messages[1][-1] == {"role": "user", "content": "TOOL_RESULT:5535"}
    assert observed_messages[1][-2] == {"role": "assistant", "content": request}
    assert outcome["trace"][0]["tool_result"] == 5535
    assert outcome["actual_tool_calls"] == 1 and outcome["status"] == "done"
    return {
        "sampler": "人工指定 JSON，僅驗證外層控制流程，沒有執行模型推論或訓練",
        "messages_seen_by_sampler": observed_messages,
        **{key: outcome[key] for key in ("status", "actual_tool_calls", "answer", "answer_correct", "required_tool_behavior", "correct")},
    }


episodes = {"correct_answer": synthetic_episode(5535), "incorrect_answer": synthetic_episode(5536)}
assert episodes["correct_answer"]["correct"] is True
assert episodes["incorrect_answer"]["answer_correct"] is False
assert episodes["incorrect_answer"]["correct"] is False

records = tool_choice.build_records()
routing_examples = []
for intent, available, expected in (
    ("numerical_addition", True, "TOOL"),
    ("numerical_addition", False, "ASK"),
    ("explanation", True, "DIRECT"),
    ("copy", True, "DIRECT"),
    ("missing_quantity", True, "ASK"),
):
    row = next(
        item for item in records
        if item["operands"] == [1, 2]
        and item["intent"] == intent
        and item["calculator_available"] is available
    )
    assert row["expected_action"] == expected and row["messages"][-1]["content"] == expected
    routing_examples.append({"intent": intent, "calculator_available": available, "messages": row["messages"]})
assert tool_choice.parse_action({"generated": "TOOL", "eos": True, "invalid_special_tokens": []}) == "TOOL"
assert tool_choice.parse_action({"generated": "TOOL", "eos": False, "invalid_special_tokens": []}) is None

tok = ByteTokenizer()
_, labels = render_chat([
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "A"},
])
assert labels[-1].item() == tok.eos_id
prerequisites = []
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
    relative, _, anchor = target.partition("#")
    path = (ROOT / "course/chapters" / relative).resolve()
    assert path.is_file(), "導言連結的檔案必須存在"
    text = path.read_text(encoding="utf-8")
    heading = next((line for line in text.splitlines() if line.startswith("## " + anchor + " ")), None) if anchor else None
    assert not anchor or heading, "導言連結的錨點必須存在"
    prerequisites.append({"label": label, "target": target, "exists": True, "heading": heading})

input_paths = [
    source_path,
    freeze_path,
    "course/chapters/0B.md",
    "course/chapters/07.md",
    "course/glossary.md",
    "tiny_perceptron/retrieval.py",
    "tiny_perceptron/data.py",
    "scripts/course_experiments/applications.py",
    "scripts/course_experiments/tool_choice.py",
]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
        "curl": subprocess.run(["curl", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    },
    "input_sha256": {path: digest(path) for path in input_paths},
    "frozen_page": {"source_sha256": digest(source_path), "figures_sha256": {}, "contains_code_block": False},
    "arithmetic": {"input": request, "observed": result, "dtype": type(result).__name__, "expected": 5535, "tolerance": "精確整數相等"},
    "rejected_requests": rejections,
    "tool_loop": loop_cases,
    "synthetic_episodes": episodes,
    "routing_examples": routing_examples,
    "eos_supervision": {"last_target": labels[-1].item(), "eos_id": tok.eos_id},
    "prerequisite_links": prerequisites,
    "result": "所有確定性斷言通過；未執行訓練，未測量模型答對率或跨平台表現",
}
save_new("verification.json", output)
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
