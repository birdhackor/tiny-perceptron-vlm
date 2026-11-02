"""Package already saved first-read notes; never read or rewrite lesson text."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[3]
base = Path(__file__).resolve().parent
task = "/root/c_v3_reader_bridges"
freeze_path = root / "docs/course-revision-20261101-c-v3/source-freeze-v19.json"
freeze = json.loads(freeze_path.read_bytes())
sessions = {
    "9.18": "9a844d303cca4c8eb41ea048ddc12238",
    "19.13": "a5d150430e88462db146e0f219cd15c6",
    "19.16": "0d22692557fb41de959393787a93c6f2",
    "19.17": "ae55b731663d4802ab7f015c4d76ac5e",
    "19.19": "ccdbdfca32cf4a48a1aca9795a9aa5c7",
}


def ref(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


record_pages = []
page_notes = {}
for pid, sid in sessions.items():
    page = next(row for row in freeze["pages"] if row["page_id"] == pid)
    folder = root / "docs/course-revision-20261016/reader-sessions" / sid
    opening = json.loads((folder / "opening.json").read_bytes())
    state = json.loads((folder / "state.json").read_bytes())
    paths = sorted(folder.glob("note-*.json"))
    notes = [json.loads(path.read_bytes()) for path in paths]
    assert state["next_index"] == len(notes)
    assert opening["reviewer_task"] == task and opening["source_sha256"] == page["source_sha256"]
    assert hashlib.sha256((root / page["snapshot"]).read_bytes()).hexdigest() == page["source_sha256"]
    last = opening["started_at"]
    for i, note in enumerate(notes):
        assert note["unit_index"] == i and note["page_id"] == pid and note["reviewer_task"] == task
        assert note["saved_at"] >= last
        last = note["saved_at"]
    page_notes[pid] = notes
    record_pages.append({
        "page_id": pid,
        "session": sid,
        "opening": ref(folder / "opening.json"),
        "source": ref(root / page["snapshot"]),
        "figures": {},
        "notes": [ref(path) for path in paths],
        "started_at": opening["started_at"],
        "completed_at": last,
        "units": len(notes),
    })

now = datetime.now(UTC).isoformat()
records = {
    "schema_version": 1,
    "reviewer_task": task,
    "source_freeze": ref(freeze_path),
    "reading_plan": ref(base / "reading-plan.json"),
    "sessions": record_pages,
    "total_units": sum(row["units"] for row in record_pages),
    "created_at": now,
}
records_path = base / "reader-records.json"
write_new(records_path, records)

program_details = {
    "9.18": "人工分數示範，先校準NLL挑T、policy挑tau，固定後評final；逐筆可分預測與真值、coverage與risk；X/Y跨題次序反例及零接受None均已手推，未執行。",
    "19.13": "此頁為用途導覽與最小輸入表，沒有可執行程式區塊；不要求為導覽硬加程式。",
    "19.16": "逐字固定查表、CLS/SEP/PAD、四列到一題的slices回填、候選softmax與CLS前向均能解釋；隨機例只查結構不變性，不代表成品準確。未執行。",
    "19.17": "rule_depth與policy可手推，manual表/硬上限/requested/actual與前置門可區分；min例只計預算，實際B工作及成本另報。未執行。",
    "19.19": "五行終端命令能對各產品，release/下載JSON/三推論JSON可分身份與任務輸出；課文作者CPU收據與本讀者實際執行分開。未執行。",
}
variation_unit = {"9.18": 6, "19.13": 0, "19.16": 8, "19.17": 5, "19.19": 11}
reports = []
for position, (pid, sid) in enumerate(sessions.items()):
    page = next(row for row in freeze["pages"] if row["page_id"] == pid)
    notes = page_notes[pid]
    final = next(check for check in notes[-1]["concept_checks"] if check["concept"].startswith("頁末整體"))
    issues = [issue for note in notes for issue in note.get("issues", [])]
    for issue in issues:
        issue.update(
            unit_index=issue["source_unit"],
            location=f"{pid}/unit-{issue['source_unit']:03}",
            reader_understanding=issue["already_taught"],
            missing_explanation=issue["missing_or_burden"],
            resolved_at=("unit-005；主要疑點已澄清，僅保留用字可選建議" if issue["id"] == "BR-19.17-01" else "尚未修文；可選不阻主線"),
            status="optional_pending_coordinator",
        )
    variation = notes[variation_unit[pid]]["small_change"]
    report = {
        "schema_version": 1,
        "review_stage": "reader",
        "reviewer_context": "fresh_at_sequential_first_read",
        "page_id": pid,
        "lesson_id": pid,
        "source": page["source"] + "#" + pid,
        "source_snapshot": ref(root / page["snapshot"]),
        "reviewer_task": task,
        "source_sha256": page["source_sha256"],
        "figure_sha256": page["figures_sha256"],
        "figures_sha256": page["figures_sha256"],
        "verdict": "pass",
        "verdict_scope": "正文理解已足夠；不包含尚未做的實頁視覺、程式執行或技術原紀錄核對。可選問題不代表必要缺口。",
        "reader_summary": notes[-1]["understanding"],
        "learning_question": page["title"],
        "four_questions": {
            "identity": {"answer": final["identity"], "evidence": final["source_quote"], "sufficiency": "已足夠"},
            "mechanism": {"answer": final["mechanism"], "evidence": final["source_quote"], "sufficiency": "已足夠"},
            "need": {"answer": final["need"], "evidence": final["source_quote"], "sufficiency": "已足夠"},
            "usage": {"answer": final["usage"], "evidence": final["source_quote"], "sufficiency": "已足夠"},
            "example_scope": {"answer": final["example_scope"], "evidence": final["source_quote"], "sufficiency": "已足夠"},
        },
        "checks": {
            "background_and_links": {"status": "pass", "details": "依派工設定按順序已讀必要主線；本輪未開額外前置/選讀/19.23以後來源，也未以連結存在代替補讀。精確範圍在40份原始notes。"},
            "terms": {"status": "pass", "details": "首次需要的概念分別存四問，第二問機制/需要各有原句與判定；待後段展示項保留原判，後文另記。19.19 RNG有一個可選名稱補充，非必要阻礙。" if pid == "19.19" else "首次需要的概念分別存四問，第二問機制/需要各有原句與判定；待後段展示項保留原判，後文另記。"},
            "examples": {"status": "pass", "details": final["example_scope"]},
            "program_explanation": {"status": "not_applicable" if pid == "19.13" else "pass", "details": program_details[pid]},
            "exercise": {"status": "pass", "details": "原練習與讀者小變化均保存題答/理由/規則來源，區分讀者新增材料與來源既有需要；未執行練習。"},
        },
        "variation": {"change": variation["question"], "prediction": variation["answer"], "reason": variation["reason"], "note": record_pages[position]["notes"][variation_unit[pid]]},
        "missing_visuals": notes[-1]["missing_visuals"],
        "visual_checks": {"status": "not_verified", "figures": {}, "required_missing_figure": False, "details": "此凍結頁無引用圖；已逐段檢查是否缺必要圖。尚未查看實際桌機/手機頁面，不宣稱版面通過。"},
        "issues": issues,
        "required_issue_ids": [],
        "optional_issue_ids": [item["id"] for item in issues],
        "trace_file": records_path.relative_to(root).as_posix(),
        "trace_json_pointer": "/sessions/" + str(position),
        "session": record_pages[position],
        "first_use_concept_records": [
            {"note": record_pages[position]["notes"][i], "concept_json_pointer": "/concept_checks/" + str(j), "concept": check["concept"], "original_verdict": check["verdict"]}
            for i, note in enumerate(notes)
            for j, check in enumerate(note["concept_checks"])
        ],
        "cross_unit_burden": [
            {"unit_index": i, "observation": note["cross_unit_burden"], "later_clarification": note["later_clarification"]}
            for i, note in enumerate(notes)
        ],
        "source_execution": "not_run",
        "created_at": now,
    }
    reports.append(report)

report = {
    "schema_version": 1,
    "report_kind": "immutable_reader_original",
    "review_stage": "reader",
    "reviewer_task": task,
    "author_task": "/root",
    "reader_profile": "數學不錯的高中生；依派工按順序已讀至各頁所需主線前文。本輪不重新驗證全部前文介紹時序。",
    "review_route": list(sessions),
    "source_freeze": ref(freeze_path),
    "reading_plan": ref(base / "reading-plan.json"),
    "records": ref(records_path),
    "started_at": record_pages[0]["started_at"],
    "first_read_completed_at": record_pages[-1]["completed_at"],
    "created_at": now,
    "independent_ai_first_read": True,
    "human_research": False,
    "context_access": [],
    "optional_or_external_sources_opened_during_first_read": [],
    "post_route_exposure_disclosure": {
        "occurred": True,
        "after_all_first_read_notes_saved": True,
        "last_first_read_saved_at": record_pages[-1]["completed_at"],
        "file": ref(root / "scripts/omni_revision/core_report.py"),
        "range": "lines 1–280",
        "reason": "首讀完成後尋找native reader report格式時誤選同名technical工具；其中有範圍外C/整合評測細節。",
        "impact": "40份原始checkpoint已先保存，未修改；報告僅封裝既存首讀判斷，不引用此檔為證。後續不宣稱仍未見任何範圍外技術內容。",
        "parent_notified": True,
    },
    "pages": reports,
    "summary": {"pages": 5, "units": 40, "required_issues": 0, "optional_issues": 3, "text_verdict": "pass", "visual_status": "not_yet_verified", "execution_status": "not_run"},
}
report_path = base / "reader-original.json"
write_new(report_path, report)
intake = {
    "schema_version": 1,
    "reviewer_task": task,
    "report": ref(report_path),
    "records": ref(records_path),
    "created_at": now,
    "pages": [{"page_id": page["page_id"], "json_pointer": "/pages/" + str(i), "source_sha256": page["source_sha256"], "figure_sha256": page["figure_sha256"], "verdict": page["verdict"], "session": record_pages[i]} for i, page in enumerate(reports)],
}
intake_path = base / "reader-original-intake.json"
write_new(intake_path, intake)
print(json.dumps({"original": ref(report_path), "intake": ref(intake_path), "records": ref(records_path), "summary": report["summary"]}, ensure_ascii=False, indent=2))
