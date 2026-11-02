"""19.24 v6: exact source/inline/table/trace-scope recheck against own raw projection."""

import contextlib
import hashlib
import io
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.24"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v6.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


original = read(REVIEW / "technical-original-v1.json")
assert digest(REVIEW / "technical-original-v1.json") == "1cefb37887387e48343ee83d472a2a25e4ad1c64b08407782bfff9ff66bfb65f"
assert digest(FREEZE) == "57f17dde51dac7f9b190e0b02dac68ad8f4f8214f368af1c8c8f8f2aa6ba18df"
freeze = read(FREEZE)
page = next(row for row in freeze["pages"] if row["page_id"] == "19.24")
assert page["source_sha256"] == "dd918e3808998845b615e8b53f32c75e3d16e872f06b7a6057643e70b294c88e"
source = ROOT / page["snapshot"]
assert digest(source) == page["source_sha256"] and page["figures_sha256"] == {}
body = source.read_text()
working = (ROOT / page["source"]).read_text()
slice_now = re.search(r"^## 19\.24 .*?(?=^## |\Z)", working, re.M | re.S).group()
assert slice_now == body
old = (ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.24.md").read_text()
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert blocks == re.findall(r"```python\n(.*?)```", old, re.S) and len(blocks) == 2
captured = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(captured):
    for block in blocks:
        exec(compile(block, str(source), "exec"), namespace)
assert (namespace["gold_ok"], namespace["route_ok"], namespace["whole_ok"]) == (3, 2, 1)
notebook = ROOT / page["notebook"]
assert digest(notebook) == page["notebook_sha256"]
cells = read(notebook)["cells"]
assert len(cells) == 7
assert ["".join(cells[index]["source"]) for index in (3, 5)] == [block.rstrip("\n") for block in blocks]
source_bindings = {}
for source_ref in original["sources"]:
    if source_ref["kind"] == "repository_code":
        actual = digest(ROOT / source_ref["path"])
        assert actual == source_ref["sha256"], source_ref["path"]
        source_bindings[source_ref["path"]] = actual
projection = HERE / "measurement_projection_v1.json"
assert digest(projection) == "4a7e20ba975181a8d02dd7119e637b4fa9af2d52f575a7bf47388f9191a088c3"
data = read(projection)
counts = {route: Counter() for route in ("A", "B", "C", "none")}
for row in data["auto_rows"]:
    counts[row["gold_route"]].update(n=1, route_correct=row["route_correct"],
        gold_chain=row["gold_route_end_to_end_correct"], real_chain=row["real_route_end_to_end_correct"])
auto_rows = {}
for line in body.splitlines():
    if re.fullmatch(r"\| (?:A|B|C|none|合計) \|.*", line):
        cells_now = [part.strip() for part in line.strip("|").split("|")]
        auto_rows[cells_now[0]] = list(map(int, cells_now[1:]))
for route in counts:
    assert auto_rows[route] == [counts[route][key] for key in ("n", "route_correct", "gold_chain", "real_chain")]
assert auto_rows["合計"] == [100, 89, 66, 55]
old_results = read(HERE / "replay_result_v1.json")
labels = {"服飾辨識": "vision_clothing", "位置關係": "vision_relation", "OCR讀字": "ocr",
    "聲音意圖問答": "voice_qa", "聲音主題後續": "voice_topic_continuation", "文字問答": "text",
    "工具呼叫": "tool_call", "工具結果答覆": "tool_reply", "工具概念": "tool_concept",
    "工具缺必要資訊": "tool_missing", "工具不可用": "tool_unavailable", "工具不支援": "tool_unsupported"}
old_table_checked = {}
for line in body.splitlines():
    if line.startswith("| "):
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[0] in labels:
            task = old_results["old_A_table"][labels[parts[0]]]
            assert parts[1:] == [f"{task['old_correct']}／{task['observations']}", f"{task['wrapper_correct']}／{task['observations']}"]
            old_table_checked[parts[0]] = parts[1:]
assert len(old_table_checked) == 12
a_defs = [row for row in data["auto_definitions"] if row["gold_route"] == "A"]
composition = Counter(row.get("old_A_task", row["payload_kind"]) for row in a_defs)
assert composition == {"tool_call": 15, "vision_clothing": 1, "vision_relation": 1, "ocr": 1, "voice_qa": 1, "video_audio_definition": 1}
new_composition_quote = "A的20族包含15個計算器往返請求、4個舊媒體請求（服飾辨識、位置關係、OCR讀字、語音問答各1個），以及1個新影音加聲音輸出請求"
new_failures_quote = "四個舊媒體與一個新影音請求直接交A時都完成，但C把這五個請求送到none"
assert new_composition_quote in body and new_failures_quote in body
assert "5個影音加聲音輸出" not in body and "五個影音請求" not in body
defs = {row["family"]: row for row in data["auto_definitions"]}
media = [row for row in data["auto_rows"] if row["gold_route"] == "A" and not row["family"].startswith("auto-A-calculator")]
assert len(media) == 5 and all(row["gold_route_end_to_end_correct"] and row["actual_route"] == "none" and not row["real_route_end_to_end_correct"] for row in media)
assert sum("sound_analysis" in row["gold_route_result"] for row in media) == 1
missing = Counter()
for row in data["auto_rows"]:
    for key in ("gold_route_result", "real_route_result"):
        result = row.get(key, {})
        if row["gold_route"] == "B":
            assert set(result) == {"correct"}
            missing["B_answer"] += 1
        elif row["gold_route"] == "C":
            assert set(result["actual"]) == {"choice", "info", "rubric"}
            assert "choice_accepted" in result and "choice_decision" in result
            assert "logits" not in result and "choice_info" not in result and "information" not in result
            missing["C_service_logits_and_choice_info_prediction"] += 1
assert missing == {"B_answer": 60, "C_service_logits_and_choice_info_prediction": 60}
scope_quote = "B的Auto列只存對錯，未存答案；C存三型代表答案與接受狀態，未存各服務分數及對應Choice的資訊充分性預測，適配中間結果也未保存。"
assert scope_quote in body and "B／C的每一步輸出則無法由這份Auto紀錄重建" in body
assert "完整逐題輸出" not in body and "原始逐列結果、已保存的工具trace" in body
assert "前面的逐步紀錄清單仍是完整診斷應保存的內容。" in body
out = HERE / "current_recheck_result_v1.json"
assert not out.exists()
result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_p5_technical_19_24",
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "source": {"path": page["snapshot"], "sha256": digest(source), "bytes": source.stat().st_size, "lines": len(body.splitlines()), "full_current_working_slice_equals_freeze": True},
    "original_report_sha256": digest(REVIEW / "technical-original-v1.json"),
    "portable_projection_sha256": digest(projection), "unchanged_repository_sources_checked": source_bindings,
    "exact_two_inline_code_blocks_unchanged": True, "inline_stdout": captured.getvalue(),
    "inline_block_sha256": [hashlib.sha256(block.encode()).hexdigest() for block in blocks],
    "notebook": {"path": page["notebook"], "sha256": digest(notebook), "cells_read": 7, "two_teaching_cells_equal_current_body": True},
    "current_Auto_table": auto_rows, "current_old_A_table": old_table_checked, "A_actual_composition": composition,
    "I01_current_source_quote": new_composition_quote, "I01_current_failure_quote": new_failures_quote,
    "I01_original_data_five_media_success_and_none_route": [{"family": row["family"], "payload_kind": defs[row["family"]]["payload_kind"], "gold_success": row["gold_route_end_to_end_correct"], "actual_route": row["actual_route"]} for row in media],
    "I02_original_missing_trace_counts": missing, "I02_current_source_quote": scope_quote,
    "I02_current_scope_explicit": True, "author_results_summary_used_as_authority": False,
    "new_necessary_issues": 0, "new_optional_issues": 0, "NN_inference_calls": 0, "weights_loaded": 0, "training_updates": 0,
    "environment": {"python": platform.python_version(), "device": "CPU", "libraries": "stdlib only; torch/model inference not imported"},
    "limits": "Full v6 body/7 notebook cells read. No new NN, training, audio waveform analysis, browser, Colab, Windows or cross-hardware resume checks. Original missing Auto outputs remain missing and are now explicitly disclosed."}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": str(out.relative_to(ROOT)), "bytes": out.stat().st_size, "sha256": digest(out), "I01": "resolved", "I02": "resolved", "NN_inference_calls": 0}, ensure_ascii=False))
