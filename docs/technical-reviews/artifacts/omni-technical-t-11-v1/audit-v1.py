"""T.11 獨立核對：只跑教材短 CPU 參考與歷史 JSON，不訓練正式模型。"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-t-11-v1")
OUT = Path(__file__).with_name("execution-summary-v1.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(data)


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def run_cli(name, arguments):
    argv = [sys.executable, *arguments]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
    raw = CACHE / (name + "-raw-v1.txt")
    write_once(raw, result.stdout + result.stderr)
    assert result.returncode == 0, (name, result.stderr)
    return {"argv": argv, "exit_code": result.returncode, "raw": ref(raw),
            "key_output": result.stdout.splitlines()[:3]}


started = datetime.now(UTC).isoformat()
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "T.11")
assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
assert page["figures_sha256"] == {}
canonical = json.loads((ROOT / freeze["canonical_source_freeze"]["path"]).read_text())
linked = {}
for lesson in ["19.21", "19.22", "19.23", "19.24", "20.16"]:
    entry = next(p for p in canonical["pages"] if p["page_id"] == lesson)
    path = ROOT / entry["snapshot"]
    assert sha(path) == entry["source_sha256"]
    text = path.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)
    code = "\n\n".join(blocks)
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    old = Path.cwd()
    working = CACHE / (lesson + "-execution-v1")
    working.mkdir()
    try:
        os.chdir(working)
        with contextlib.redirect_stdout(output):
            exec(compile(code, str(path), "exec"), namespace)
    finally:
        os.chdir(old)
    raw = CACHE / (lesson + "-stdout-v1.txt")
    write_once(raw, output.getvalue())
    linked[lesson] = {"source": ref(path), "python_blocks": len(blocks),
                      "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                      "raw": ref(raw), "key_output": output.getvalue().splitlines()}

assert "(3, 3) (3, 2, 25, 25)" in (CACHE / "19.21-stdout-v1.txt").read_text()
assert "[True, True, True, True, True]" in (CACHE / "19.21-stdout-v1.txt").read_text()
assert "1.371999" in (CACHE / "19.22-stdout-v1.txt").read_text()
assert "示意記帳分母 gold／整鏈 1 1" in (CACHE / "19.22-stdout-v1.txt").read_text()
assert "M保持原值 True" in (CACHE / "19.23-stdout-v1.txt").read_text()
assert "等分數人工校準T 1.0" in (CACHE / "19.23-stdout-v1.txt").read_text()
assert "真路由整條成功： 1 / 4" in (CACHE / "19.24-stdout-v1.txt").read_text()
assert "accepted risk 0.4" in (CACHE / "20.16-stdout-v1.txt").read_text()

# 依實際模型輸入的 byte 協議獨立重建，不讀已重建的 sidecar 結果。
from tiny_perceptron.retrieval import call_tool

tools_path = ROOT / "docs/course-experiments/results/tools.json"
tools_before = sha(tools_path)
assert tools_before == "3befd7fcb9a1a48d850761ab6d0917ac42388dc3bbfe9f011f0b98c3752d7140"
tools = json.loads(tools_path.read_text())
roles = {"system": 7, "user": 3, "assistant": 4}
records = []
executed = 0
for group in ["samples", "one_step_cap_samples"]:
    for index, episode in enumerate(tools["results"][group]):
        first = next(e["generation"] for e in episode["trace"] if "generation" in e)
        history = copy.deepcopy(first["messages"][:2])
        assert history[1]["content"] == episode["question"]
        for event_index, event in enumerate(episode["trace"]):
            if "generation" not in event:
                continue
            generation = event["generation"]
            ids = [1]
            for message in history:
                ids += [roles[message["role"]]] + [b + 8 for b in message["content"].encode()] + [2]
            ids += [4]
            assert ids == generation["input_ids"]
            assert len(ids) == generation["input_tokens"]
            records.append({"pointer": f"/results/{group}/{index}/trace/{event_index}/generation",
                            "ids_equal": True, "input_tokens": len(ids),
                            "messages_differ": generation["messages"] != history})
            if event.get("executed") and event.get("finite_result"):
                request = json.loads(generation["samples"][0]["generated"])
                assert call_tool(request) == event["tool_result"]
                executed += 1
                history += [
                    {"role": "assistant", "content": generation["samples"][0]["generated"]},
                    {"role": "user", "content": f"TOOL_RESULT:{event['tool_result']}"},
                ]
assert len(records) == 56 and sum(r["messages_differ"] for r in records) == 36
assert sha(tools_path) == tools_before
trace_raw = CACHE / "independent-tool-reconstruction-v1.json"
write_once(trace_raw, json.dumps(records, ensure_ascii=False, indent=2) + "\n")

historical = {}
for name in ["rag", "reasoning"]:
    p = ROOT / f"docs/course-experiments/results/{name}.json"
    r = json.loads(p.read_text())
    historical[name] = {"source": ref(p), "revision": r["revision"], "device": r["device"],
                        "seed": r["seed"], "step_scale": r["step_scale"],
                        "split": r["results"]["split"]}
moe_path = ROOT / "docs/selftrained/results/public-raw/moe/test/metrics.json"
moe = json.loads(moe_path.read_text())
assert moe["evaluation_complete"] is True and moe["count"] == moe["expected_count"] == 3734
assert moe["per_task_final_reply"]["tool_call"]["tool_roundtrip"]["numerator"] == 0
historical["moe_final_test"] = {"source": ref(moe_path), "n": moe["count"],
    "teacher_forcing_used_for_generation": moe["teacher_forcing_used_for_generation"],
    "tool_roundtrip": moe["per_task_final_reply"]["tool_call"]["tool_roundtrip"],
    "scope": "讀取已保存的歷史結果；沒有重跑模型、GPU或訓練。"}
report_path = ROOT / "docs/lesson-products/results/final-report-v1.json"
product_report = json.loads(report_path.read_text())
historical["products_final_test"] = {"source": ref(report_path),
    "b_boxes_5": product_report["b"]["5"]["boxes"],
    "b_sum_5": product_report["b"]["5"]["sum"],
    "c_choice": product_report["c"]["choice"]["after"],
    "gates": product_report["gates"], "scope": product_report["scope"]}
for name in ["b", "c"]:
    receipt = json.loads((CACHE / f"{name}-training-receipt.json").read_text())
    assert receipt["random_initialization"] is True and receipt["resume_source"] is None
    assert receipt["weights_changed"] is True
    historical[name + "_training"] = {"source": ref(CACHE / f"{name}-training-receipt.json"),
        **{k: receipt[k] for k in ["seed", "batch_size", "last_step", "selected_step", "random_initialization", "environment", "data_manifest_sha256"]}}

cli = [run_cli("selftrained-local-help", ["scripts/selftrained/train_local_stage.py", "--help"]),
       run_cli("selftrained-chat-help", ["scripts/selftrained/chat.py", "--help"]),
       run_cli("products-help", ["-m", "tiny_perceptron.lesson_products", "--help"]),
       run_cli("products-local-help", ["-m", "scripts.lesson_products.train_local", "--help"]),
       run_cli("natural-help", ["scripts/natural_assistant.py", "--help"])]

result = {"started_at": started, "completed_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-11-v1/audit-v1.py",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "threads": "1"},
    "t11_source": ref(ROOT / page["snapshot"]), "figures_sha256": {},
    "hand_example": {"old_bytes": 80000, "new_bytes": 60000, "old_correct": 4,
        "new_correct": 3, "n": 5, "storage_smaller": 60000 < 80000,
        "accuracy_preserved": 3 >= 4, "origin": "教材手寫假設；本次只核算，非模型實測"},
    "linked_inline_cpu": linked,
    "tool_trace": {"source": ref(tools_path), "episodes": 38, "generations": 56,
        "ids_equal": 56, "messages_mismatches": 36, "finite_executor_results_rechecked": executed,
        "details": ref(trace_raw), "model_invocations": 0},
    "historical": historical, "cli_help": cli,
    "scope": "未重跑歷史GPU、未測成熟模型、未啟動正式訓練／Modal／發布。19.22只有教材指定一次CPU更新；其他內嵌碼只作機制或人工數學核對。"}
write_once(OUT, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": str(OUT.relative_to(ROOT)), "sha256": sha(OUT),
                  "inline_lessons": list(linked), "tool_inputs_verified": len(records),
                  "messages_mismatches": 36, "cli_help_ok": len(cli)}, ensure_ascii=False))
