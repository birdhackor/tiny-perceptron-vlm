"""Bounded original-owner recheck: four historical 19.24 issue occurrences only."""

import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PACKET = ROOT / "docs/course-revision-20261101-c-v3/original-owner-callbacks-v21/omni_p5_technical_19_24.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def pointer(value, path):
    for item in path.lstrip("/").split("/") if path else []:
        item = item.replace("~1", "/").replace("~0", "~")
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


packet = read(PACKET)
assert sha(PACKET) == "11876db8d04f5ee7adeef3393f92750150fb46400ac686566e0c495cd44ec04b"
assert packet["intended_original_owner"] == "/root/omni_p5_technical_19_24"
assert [row["index"] for row in packet["rows"]] == [433, 434, 435, 436]
occurrences = []
for row in packet["rows"]:
    old_path = ROOT / row["report"]["path"]
    assert sha(old_path) == row["report"]["sha256"]
    old = read(old_path)
    issue = pointer(old, row["json_pointer"])
    assert issue == row["original_issue"]
    assert old["reviewer_task"] == row["original_issue_owner"] == "/root/omni_p5_technical_19_24"
    assert issue["necessity"] == "necessary"
    fixes = []
    for fixed in row["original_verification"]:
        fix_path = ROOT / fixed["report"]["path"]
        assert sha(fix_path) == fixed["report"]["sha256"]
        fix = pointer(read(fix_path), fixed["json_pointer"])
        assert fix["reviewer_task"] == "/root/omni_p5_technical_19_24" and fix["verdict"] == "pass"
        closure = next(item for item in fix["original_issue_resolutions"] if item["json_pointer"] == row["json_pointer"])
        assert closure["status"] == "resolved" and closure["necessity_preserved"] == "necessary"
        fixes.append({"report": fixed["report"], "json_pointer": fixed["json_pointer"], "resolution_pointer": "/original_issue_resolutions/" + str(fix["original_issue_resolutions"].index(closure))})
    occurrences.append({"index": row["index"], "issue_id": row["issue_id"], "report": row["report"],
        "json_pointer": row["json_pointer"], "report_created_at": old["created_at"], "report_verdict": old["verdict"],
        "issue_status_in_referenced_report": issue["status"], "necessity": issue["necessity"], "fix_refs_checked": fixes})
freeze_path = ROOT / packet["current_source_freeze"]["path"]
assert sha(freeze_path) == "d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea"
freeze = read(freeze_path)
page = next(item for item in freeze["pages"] if item["page_id"] == "19.24")
snapshot = ROOT / page["snapshot"]
assert sha(snapshot) == page["source_sha256"] == "3118c0054adf229741788ab742f8a55e1dd67a6149570e3c489f9e21605d0bbf"
assert page["figures_sha256"] == {}
body = snapshot.read_text()
working = (ROOT / page["source"]).read_text()
section = re.search(r"^## 19\.24 .*?(?=^## |\Z)", working, re.M | re.S).group()
assert section == body
old_current = read(ROOT / packet["rows"][0]["report"]["path"])
unchanged_artifacts = []
for artifact in old_current["artifacts"]:
    path = ROOT / artifact["path"]
    assert sha(path) == artifact["sha256"]
    unchanged_artifacts.append({"id": artifact["id"], **ref(path)})
projection_ref = next(item for item in old_current["artifacts"] if item["id"] == "a-projection")
projection = read(ROOT / projection_ref["path"])
definitions = {row["family"]: row for row in projection["auto_definitions"]}
a_definitions = [row for row in definitions.values() if row["gold_route"] == "A"]
composition = Counter(row.get("old_A_task", row["payload_kind"]) for row in a_definitions)
assert composition == {"tool_call": 15, "vision_clothing": 1, "vision_relation": 1, "ocr": 1, "voice_qa": 1, "video_audio_definition": 1}
a_rows = [row for row in projection["auto_rows"] if row["gold_route"] == "A"]
media = [row for row in a_rows if not row["family"].startswith("auto-A-calculator")]
assert len(a_rows) == 20 and len(media) == 5
assert all(row["gold_route_end_to_end_correct"] and row["actual_route"] == "none" and not row["real_route_end_to_end_correct"] for row in media)
assert sum("sound_analysis" in row["gold_route_result"] for row in media) == 1
tools = [row for row in a_rows if row not in media]
assert len(tools) == 15 and all(row["route_correct"] and not row["gold_route_result"]["tool_trace"]["executed"] and not row["gold_route_end_to_end_correct"] for row in tools)
missing = Counter()
for row in projection["auto_rows"]:
    for kind in ("gold_route_result", "real_route_result"):
        if row["gold_route"] == "B":
            assert set(row[kind]) == {"correct"}
            missing["B_actual_answer"] += 1
        elif row["gold_route"] == "C":
            result = row[kind]
            assert set(result["actual"]) == {"choice", "info", "rubric"}
            assert {"choice_accepted", "choice_decision"} <= set(result)
            assert not {"logits", "information", "choice_info"} & set(result)
            missing["C_service_logits_and_corresponding_Choice_info"] += 1
assert missing == {"B_actual_answer": 60, "C_service_logits_and_corresponding_Choice_info": 60}
totals = {key: sum(row[field] for row in projection["auto_rows"]) for key, field in (
    ("route", "route_correct"), ("gold", "gold_route_end_to_end_correct"), ("real", "real_route_end_to_end_correct"))}
assert totals == {"route": 89, "gold": 66, "real": 55}
assert "| A | 20 | 15 | 5 | 0 |" in body and "| 合計 | 100 | 89 | 66 | 55 |" in body
quotes = {
    "I01_composition": "A的20族包含15個計算器往返請求、4個舊媒體請求（服飾辨識、位置關係、OCR讀字、語音問答各1個），以及1個新影音加聲音輸出請求",
    "I01_failure": "四個舊媒體與一個新影音請求直接交A時都完成，但C把這五個請求送到none",
    "I02_trace_limits": "B的Auto列只存對錯，未存答案；C存三型代表答案與接受狀態，未存各服務分數及對應Choice的資訊充分性預測，適配中間結果也未保存。",
    "I02_reconstruction_limit": "因此89／66／55可以從逐列旗標重算，B／C的每一步輸出則無法由這份Auto紀錄重建。",
    "I02_normative_list": "前面的逐步紀錄清單仍是完整診斷應保存的內容。",
    "I02_link_label": "原始逐列結果、已保存的工具trace與下載版本",
    "historical_boundary": "本節先保留C-v2神經路由的歷史量測。",
    "no_score_transfer": "C-v3沒有新A／B／C路由頭，也不能沿用本節89／66／55的成績。",
    "historical_failure_boundary": "該批C-v2未通過自己的門檻，含它的神經路由Auto也只作研究候選；後續C-v3的裝置服務另由下一節接入，不把這批路由成績改報為新版結果。",
}
located_quotes = {}
for name, quote in quotes.items():
    assert quote in body, name
    located_quotes[name] = {"line": body[:body.index(quote)].count("\n") + 1, "quote": quote}
assert "5個影音加聲音輸出" not in body and "五個影音請求" not in body and "完整逐題輸出" not in body
small_sources = []
for sid in ("s-evaluation", "s-datafreeze", "s-media", "s-definition", "s-metrics"):
    original_source = next(item for item in old_current["sources"] if item["id"] == sid)
    path = ROOT / original_source["path"]
    assert sha(path) == original_source["sha256"]
    small_sources.append({"id": sid, **ref(path)})
result = {"schema": "bounded-original-owner-callback-execution-v1", "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_19_24", "packet": ref(PACKET), "source_freeze": ref(freeze_path),
    "page": {"page_id": "19.24", "source_sha256": page["source_sha256"], "figures_sha256": {}, "snapshot": ref(snapshot), "working_slice_equals_freeze": True},
    "occurrences_preserved": occurrences, "logical_duplicate_groups_not_collapsed": {"I01": [433, 435], "I02": [434, 436]},
    "unchanged_reviewer_owned_artifacts_reused_without_copy": unchanged_artifacts,
    "unchanged_old_measurement_sources": small_sources, "I01_A_composition": composition,
    "I01_media_gold_success_count": 5, "I01_new_AV_sound_count": 1, "I01_actual_successful_calculator_executions": 0,
    "I02_missing_fields_count": missing, "old_C_v2_Auto_flag_totals": totals, "source21_located_quotes": located_quotes,
    "I01_still_fixed": True, "I02_still_fixed": True, "NN_calls": 0, "weights_loaded": 0, "training_calls": 0,
    "environment": {"python": platform.python_version(), "libraries": "stdlib only", "device": "CPU"},
    "scope_limit": "Only the four dispatched original occurrences and related source21 paragraphs/small old score/assembly code. No fresh whole-page review; no review of19.25/C-v3 capability; no full old NN replay, audio analysis, browser, training/final/Modal/GPU/public write or Git operation."}
out = HERE / "execution-result-v1.json"
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": ref(out), "indices": [row["index"] for row in occurrences], "I01": "still_fixed", "I02": "still_fixed", "NN_calls": 0}, ensure_ascii=False))
