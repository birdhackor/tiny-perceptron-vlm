"""核對工具迴圈、保存的生成軌跡與原始官方文件；不訓練模型。"""

import ast
import contextlib
import hashlib
import html
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import applications as app
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool, tool_loop


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (DEST / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def outcome(function, *args, **kwargs):
    try:
        return {"return": function(*args, **kwargs)}
    except Exception as error:
        return {"exception": type(error).__name__, "message": str(error)}


def fetch(url, anchors):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url]
    fetched = subprocess.run(command, check=True, capture_output=True).stdout
    text = fetched.decode("utf-8")
    inspected = []
    for anchor, excerpt, note in anchors:
        start = text.index(anchor)
        window = html.unescape(re.sub(r"<[^>]+>", " ", text[start : start + 16000]))
        window = re.sub(r"\s+", " ", window)
        assert excerpt in window, (url, anchor)
        inspected.append({"locator": anchor, "excerpt": excerpt, "inspection_note": note})
    return {
        "url": url,
        "command_argv": command,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_response_sha256": digest(fetched),
        "full_response_bytes": len(fetched),
        "stored_full_response": False,
        "version": "Python 3.13.16" if "python.org" in url else "Transformers v4.57.1",
        "inspected": inspected,
    }


requests = [{"name": "add", "arguments": {"a": 2, "b": 3}}, {"done": True, "answer": 5}]
cases = [
    tool_loop(requests, max_steps=3),
    tool_loop(requests, max_steps=1),
    tool_loop(requests[:1], max_steps=3),
    tool_loop([{"name": "unknown", "arguments": {"a": 2, "b": 3}}]),
]
assert [r["status"] for r in cases] == ["done", "step_limit", "needs_more_model_output", "invalid_request"]
assert [len(r["trace"]) for r in cases] == [1, 1, 1, 0]
assert [r.get("answer") for r in cases] == [5, None, None, None]
limits = {str(limit): tool_loop(requests, max_steps=limit)["status"] for limit in (1, 2, 3)}
assert list(limits.values()) == ["step_limit", "done", "done"]
wrong = [requests[0], {"done": True, "answer": 6}]
wrong_cases = [tool_loop(wrong, max_steps=limit) for limit in (1, 2, 3)]
assert [r["status"] for r in wrong_cases] == list(limits.values())
assert wrong_cases[-1]["answer"] == 6
assert tool_loop(requests[:1], max_steps=1)["status"] == "needs_more_model_output"

bad_requests = [
    {"name": "unknown", "arguments": {"a": 2, "b": 3}},
    {"name": "add", "arguments": {"a": 2}},
    {"name": "add", "arguments": {"a": True, "b": 3}},
]
validation = [outcome(call_tool, request) for request in bad_requests]
assert all(row.get("exception") == "ValueError" for row in validation)
string_action = json.dumps(requests[0])
exception_cases = {
    "字典可執行": outcome(call_tool, requests[0]),
    "JSON文字可直接交給call_tool": outcome(call_tool, string_action),
    "JSON文字不能直接放進tool_loop": outcome(tool_loop, [string_action]),
    "壞JSON解析": outcome(call_tool, "{"),
    "JSON數字不是動作物件": outcome(call_tool, "1"),
    "其他例外仍會傳出": outcome(tool_loop, [{"name": "add", "arguments": {"a": 10**400, "b": 1.0}}]),
}
assert exception_cases["JSON文字不能直接放進tool_loop"]["exception"] == "AttributeError"
assert exception_cases["壞JSON解析"]["exception"] == "JSONDecodeError"
assert exception_cases["JSON數字不是動作物件"]["exception"] == "TypeError"
assert exception_cases["其他例外仍會傳出"]["exception"] == "OverflowError"
assert list(enumerate(requests)) == [(0, requests[0]), (1, requests[1])]
assert {}.get("answer") is None
assert str(ValueError("驗證失敗")) == "驗證失敗"

frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/B.4.md"
body = frozen.read_text()
fences = re.findall(r"```python\n(.*?)```", body, re.S)
excerpt_namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(fences[0], excerpt_namespace)
library_binding = excerpt_namespace["tool_loop"]
exec(fences[1], excerpt_namespace)
assert excerpt_namespace["tool_loop"] is not library_binding
assert "call_tool" not in excerpt_namespace and "json" not in excerpt_namespace
with contextlib.redirect_stdout(io.StringIO()):
    exec(fences[2], excerpt_namespace)
assert excerpt_namespace["tool_loop"](requests) == cases[0]
library_tree = ast.parse((ROOT / "tiny_perceptron/retrieval.py").read_text())
library_node = next(node for node in library_tree.body if isinstance(node, ast.FunctionDef) and node.name == "tool_loop")
excerpt_node = ast.parse(fences[1]).body[0]
assert ast.dump(library_node) == ast.dump(excerpt_node)


class ReplayModel(torch.nn.Module):
    """只重播指定 token，以核對生成與執行順序，不當成模型品質證據。"""

    def __init__(self, replies):
        super().__init__()
        self.config = SimpleNamespace(max_length=384)
        self.replies = iter(replies)
        self.pending = None
        self.generation_count = 0

    def forward(self, ids, cache=None):
        if cache is None:
            self.pending = iter(ByteTokenizer().encode(next(self.replies)) + [ByteTokenizer.eos_id])
            self.generation_count += 1
        chosen = next(self.pending)
        logits = torch.full((ids.shape[0], 1, ByteTokenizer.vocab_size), -torch.inf)
        logits[:, :, chosen] = 0
        return {"logits": logits, "cache": True}


record = {
    "family": "pair:2:3", "operation": "add", "a": 2, "b": 3, "answer": 5,
    "messages": [{"role": "system", "content": app._TOOLS_SYSTEM}, {"role": "user", "content": "CALC:add(2,3)"}],
}
replayed = []
for replies, cap in [
    ([json.dumps(requests[0]), json.dumps(requests[1])], 3),
    ([json.dumps(requests[0])], 1),
    ([json.dumps({"done": True, "answer": 5})], 3),
]:
    model = ReplayModel(replies)
    episode = app._tool_episode(model, record, SimpleNamespace(device="cpu"), max_steps=cap)
    replayed.append({"status": episode["status"], "correct": episode["correct"], "answer_correct": episode["answer_correct"],
                     "actual_tool_calls": episode["actual_tool_calls"], "generation_count": model.generation_count,
                     "input_ids": [event["generation"]["input_ids"] for event in episode["trace"]]})
assert [r["generation_count"] for r in replayed] == [2, 1, 1]
assert [r["correct"] for r in replayed] == [True, False, False]
assert [r["status"] for r in replayed] == ["done", "step_limit", "done"]

raw_result = (ROOT / "docs/course-experiments/results/tools.json").read_bytes()
saved = json.loads(raw_result)
results = saved["results"]
tokenizer = ByteTokenizer()
audited = {}
all_events = []
for group in ("samples", "one_step_cap_samples"):
    audited[group] = []
    for sample in results[group]:
        question = sample["question"]
        match = re.fullmatch(r"CALC:(add|multiply)\((\d+),(\d+)\)", question)
        if match:
            operation, a, b = match[1], int(match[2]), int(match[3])
            truth = a + b if operation == "add" else a * b
            wanted = {"name": operation, "arguments": {"a": a, "b": b}}
        else:
            assert re.fullmatch(r"COPY:\d+", question)
            operation, truth, wanted = "copy", int(question.split(":")[1]), None
        assert truth == sample["expected"] and operation == sample["operation"]
        messages = [{"role": "system", "content": app._TOOLS_SYSTEM}, {"role": "user", "content": question}]
        status, answer, calls, generations, parameters_ok = "step_limit", None, [], [], []
        for event in sample["trace"]:
            generation = event["generation"]
            assert generation["input_ids"] == app._prompt_ids(messages, tokenizer)
            assert generation["input_tokens"] == len(generation["input_ids"])
            assert generation["candidate_count"] == 1 and generation["temperature"] == 0.0
            assert generation["max_new_tokens"] == 64 and generation["seconds"] >= 0
            raw = generation["samples"][0]
            ids = raw["generated_ids"]
            assert ids[-1] == tokenizer.eos_id and raw["eos"] is True
            assert not [token for token in ids[:-1] if token < len(tokenizer.state()["specials"])]
            assert raw["invalid_special_tokens"] == []
            assert len(ids) == raw["generated_tokens"] == generation["generated_tokens"]
            text = tokenizer.decode(ids[:-1])
            assert text == raw["generated"]
            generations.append({"generated": text, "eos": True, "invalid_special_tokens": [], "seconds": generation["seconds"]})
            all_events.append(generations[-1])
            try:
                action = json.loads(text)
            except json.JSONDecodeError:
                assert "parsed" not in event and event.get("executed") is False
                status = "invalid_request"
                break
            assert isinstance(action, dict) and event["parsed"] == action
            if "done" in action:
                assert set(action) == {"done", "answer"} and action["done"] is True
                status, answer = "done", action["answer"]
                assert event.get("executed") is False
                break
            actual = call_tool(action)
            assert event["executed"] is True and event["tool_result"] == actual
            assert event["finite_result"] is True
            assert event["request_matches_question"] == (action == wanted)
            calls.append(actual)
            parameters_ok.append(action == wanted)
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": f"TOOL_RESULT:{actual}"}]
        correct_answer = status == "done" and type(answer) in (int, float) and answer == truth
        required = not calls if operation == "copy" else bool(calls and any(parameters_ok) and answer == calls[-1])
        correct = correct_answer and required
        assert (status, answer, correct, correct_answer, required, len(calls)) == (
            sample["status"], sample["answer"], sample["correct"], sample["answer_correct"],
            sample["required_tool_behavior"], sample["actual_tool_calls"])
        assert len(sample["trace"]) <= sample["max_steps"]
        assert sample["seconds"] >= sum(g["seconds"] for g in generations)
        audited[group].append({"question": question, "operation": operation, "status": status, "answer": answer,
                               "truth": truth, "correct": correct, "tool_results": calls,
                               "parameters_match": parameters_ok, "generations": generations,
                               "first_json_object": isinstance(json.loads(generations[0]["generated"]), dict),
                               "max_steps": sample["max_steps"]})

normal = audited["samples"]
cap = audited["one_step_cap_samples"]
arithmetic = [row for row in normal if row["operation"] != "copy"]
copies = [row for row in normal if row["operation"] == "copy"]
summary = {
    "normal_tasks": len(normal), "arithmetic_tasks": len(arithmetic), "copy_tasks": len(copies),
    "correct_tasks": sum(row["correct"] for row in normal),
    "normal_statuses": dict(Counter(row["status"] for row in normal)),
    "first_json_objects": sum(row["first_json_object"] for row in normal),
    "normal_tool_calls": sum(len(row["tool_results"]) for row in normal),
    "correct_call_parameters": sum(sum(row["parameters_match"]) for row in normal),
    "arithmetic_generations": sum(len(row["generations"]) for row in arithmetic),
    "copy_generations": sum(len(row["generations"]) for row in copies),
    "cap_tasks": len(cap), "cap_statuses": dict(Counter(row["status"] for row in cap)),
    "cap_tool_calls": sum(len(row["tool_results"]) for row in cap),
    "cap_generations": sum(len(row["generations"]) for row in cap),
    "all_generations": len(all_events), "eos_generations": sum(event["eos"] for event in all_events),
    "invalid_special_tokens": sum(len(event["invalid_special_tokens"]) for event in all_events),
    "all_input_ids_reconstructed": True,
}
assert (summary["normal_tasks"], summary["arithmetic_tasks"], summary["copy_tasks"], summary["correct_tasks"]) == (20, 18, 2, 19)
assert summary["normal_statuses"] == {"done": 19, "invalid_request": 1}
assert summary["first_json_objects"] == 20
assert summary["normal_tool_calls"] == summary["correct_call_parameters"] == 18
assert (summary["arithmetic_generations"], summary["copy_generations"], summary["cap_generations"]) == (36, 2, 18)
assert summary["all_generations"] == summary["eos_generations"] == 56
assert summary["cap_tasks"] == summary["cap_tool_calls"] == 18 and summary["cap_statuses"] == {"step_limit": 18}
assert {row["question"] for row in cap} == {row["question"] for row in arithmetic}
assert all(row["max_steps"] == 3 for row in normal) and all(row["max_steps"] == 1 for row in cap)
assert all(len(row["tool_results"]) == 1 for row in arithmetic + cap)
assert all(len(row["tool_results"]) == 0 for row in copies)
assert results["accuracy"] == {"numerator": 19, "denominator": 20, "rate": 19 / 20}
assert results["status_counts"] == summary["normal_statuses"]

inputs = {path: digest((ROOT / path).read_bytes()) for path in (
    "tiny_perceptron/retrieval.py", "scripts/course_experiments/applications.py", "tiny_perceptron/data.py",
    "docs/course-experiments/results/tools.json", "docs/course-review-20261019/freeze-01/sources/B.4.md")}
historical = subprocess.run(["git", "show", saved["revision"] + ":scripts/course_experiments/applications.py"],
                            cwd=ROOT, capture_output=True, text=True)
verification = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-4/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "input_sha256": inputs, "four_runs": cases, "limits": limits, "wrong_answer_runs": wrong_cases,
    "validation_gates": validation, "exception_boundaries": exception_cases,
    "excerpt_ast_identical": True, "excerpt_rebinds_imported_name": True,
    "excerpt_dependencies_loaded_before_call": True, "protocol_replay_not_model_quality": replayed,
    "experiment_environment_as_recorded": {key: saved[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
    "recorded_revision_application_lookup": {"exit_code": historical.returncode, "stderr": historical.stderr.strip()},
    "summary": summary, "recomputed_episodes": audited,
    "limits_of_verification": "重算保存紀錄並重播協議；未重新訓練或取得原 GPU 模型。紀錄標示的 revision 沒有該實驗實作路徑，不能證明歷史執行使用的精確程式 blob。重新生成支線由保存的逐步輸出、獨立計時與現行 run_tools 的獨立呼叫支持。",
    "result": "所有本地核對與逐步紀錄重算通過。",
}

receipts = [
    fetch("https://docs.python.org/3.13/library/stdtypes.html", [("id=\"dict.get\"", "If default is not given, it defaults to None", "缺鍵時回預設值；普通字典 get 不因缺鍵拋出 KeyError。")]),
    fetch("https://docs.python.org/3.13/library/functions.html", [
        ("id=\"enumerate\"", "from start which defaults to 0", "enumerate 依序產生計數與項目；預設從 0 開始。"),
        ("id=\"func-str\"", "Return a str version of object", "str 將錯誤物件轉為字串表示。")]),
    fetch("https://docs.python.org/3.13/reference/compound_stmts.html", [
        ("id=\"except-clause\"", "When an exception occurs in the try suite", "例外處理只涵蓋 try suite；列出的型別及其子類會匹配，未匹配則向外傳。"),
        ("id=\"function-definitions\"", "binds the function name", "def 執行時建立函式物件並綁定名稱；本文的定義可取代同作用域匯入的名稱，函式體至呼叫時才執行。")]),
    fetch("https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation", [
        ("id=\"transformers.GenerationConfig.eos_token_id\"", "The id of the end-of-sequence token.", "EOS 是序列結束 token；本專案 _sample 在模型選到它時停止當次生成，整題 done 由外層協議判斷。")]),
]
save("verification.json", verification)
save("source-receipts.json", {"sources": receipts, "result": "以 curl 讀取原始官方文件；只保存短摘錄與原始回應 SHA-256，不保存全文。"})
print(json.dumps(summary, ensure_ascii=False))
print("核對完成：工具流程、例外邊界、程式節錄、逐題成績、生成輸入、EOS 與官方文件。")
