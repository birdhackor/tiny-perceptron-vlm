"""封存本角色已完成的原始判斷；不產生 basis/current/closure。"""

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REVIEWER = "/root/c_v3_reader_entries"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def ref(path):
    path = path if isinstance(path, Path) else ROOT / path
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}

def read(path):
    return json.loads(path.read_bytes())

def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

freeze_path = ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v19.json"
freeze = read(freeze_path)
metadata = {p["page_id"]: p for p in freeze["pages"]}
sessions = {
    "readme": "424567101a70489793de5a134f43f023",
    "course": "8ec95c28a5544772b92762d786868fd0",
    "training": "c884bb58731c4832ae21cfbefeba51a3",
    "asset-storage": "1fde18b814194f598da59d4d89d5da3e",
    "curriculum": "86a52948681a46ba81a0ec3e07755100",
    "validation": "03f3529698c6412592dee7c86439910e",
    "index": "c2de0686eb0742f9afbdc53f0a4cca9f",
}
summaries = {
    "readme": "從接字表的限制串起全課，讀書、重跑、訓練、成果與維護路線可辨；通路成功與模型能力分清。入口縮寫、舊產品限定前置、維護模式有可選改善，字編號圖未實際視覺驗證。",
    "course": "入門以貓看狗編號還原，並以相同末字不同前文解釋路線；主線、兩次C插入、延伸與按題跳讀清楚。各方法內部不是導覽的即時先備。",
    "training": "依各節操作入口理解資料、家族分割、dry-run、真正更新、驗證/測試、模式/檔案/配方差別與結果判準。未讀所有前文章或折疊固定配方，故不宣稱已懂全部架構原理或已執行；C-v3預設評級理由仍需所連規則。",
    "asset-storage": "程式、資料、權重與完整訓練狀態的分工可懂；Git/LFS/HF/Release與指紋核對能回應下載/分享/重做需要，不把一致檔案當正確模型。",
    "curriculum": "設計取捨與來源限制、機制/品質/資源分工可理解；必要導航問題：明訂第19章只順讀到19.24，卻在同頁把19.25列新增主線及現行推薦。流程圖只有源碼標籤/路徑核對，未實看渲染。",
    "validation": "分層證據、分母、家族/文字新穎性、校準/選版/final時點與硬體範圍可辨；AI審閱/閱讀時間與真人/執行耗時分清。未開舊審核報告或重算結果。",
    "index": "短敘事說明階段需求，直接第一節、暖身、按題路線可用。332小節與其他入口331口徑看似不一致，保留可選核對；實際按鈕/手機選單未驗。",
}

spec = importlib.util.spec_from_file_location("original_reader_audit", ROOT / "docs/review-tools/incremental_revision.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
pages = []
all_issues = []
for pid, sid in sessions.items():
    p = metadata[pid]
    folder = ROOT / "docs/course-revision-20261016/reader-sessions" / sid
    opening = read(folder / "opening.json")
    state = read(folder / "state.json")
    notes = sorted(folder.glob("note-*.json"))
    trace = {"opening": ref(folder / "opening.json"), "source": ref(p["snapshot"]), "figures": {name: freeze["figure_snapshots"][name] for name in p["figures_sha256"]}, "notes": [ref(n) for n in notes]}
    audited_pid, last_at = audit.reader_session(ROOT, trace, REVIEWER)
    assert audited_pid == pid and state["next_index"] == len(notes)
    issues = []
    for i, note_path in enumerate(notes):
        note = read(note_path)
        for issue in note.get("issues", []):
            entry = {**issue, "page_id": pid, "unit_index": i, "source_sha256": p["source_sha256"], "unit_sha256": note["unit_sha256"], "discovery": "independent_ai_first_read", "evidence": ref(note_path), "evidence_pointer": "/issues/" + str(note["issues"].index(issue))}
            issues.append(entry)
            all_issues.append(entry)
    pages.append({
        "page_id": pid,
        "source": p["source"],
        "selector": p["selector"],
        "source_sha256": p["source_sha256"],
        "figures_sha256": p["figures_sha256"],
        "reviewer_task": REVIEWER,
        "verdict": "需補說明：必要導航終點排除19.25；圖與版面未驗證" if pid == "curriculum" else "已足夠（文字與本頁定位範圍；執行及實際圖／版面未驗證）",
        "reader_summary": summaries[pid],
        "unit_count": len(notes),
        "completed_at": last_at.isoformat(),
        "native_reader_trace": trace,
        "sessions": [trace],
        "issues": issues,
        "four_questions": {"status": "recorded_per_concept_and_at_page_end", "question_2": "mechanism 與 need 各自有答案、來源及足夠性，未合併放行", "evidence": trace["notes"]},
        "checks": {
            "actual_incremental_reading": {"status": "pass", "reason": "原生start/next逐段開放，每段手寫觀察在下一段前保存；reader_session已核順序及來源。"},
            "concepts_methods_and_small_changes": {"status": "recorded", "reason": "按實際當段目的判導覽、操作或機制；每段保存小變化與規則/需要來源，不把自設題算作者原例。"},
            "prerequisite_scope": {"status": "limited_to_actual_route", "reason": "只用先前已讀七頁與文中精確前置定位；未補讀的模型原理/折疊/舊審閱不當已教證據。"},
            "necessary_visuals": {"status": "not_verified" if p["figures_sha256"] else "text_sufficient_at_this_scope", "reason": "含圖頁尚未成功視覺查看；無圖頁逐段說明文字/表是否足夠，未因零圖跳過圖需求。"},
            "actual_desktop_mobile_layout": {"status": "not_verified", "reason": "原判封存前未看整頁桌機/手機；後續實看必須另存時點，不能改寫此初判。"},
            "execution": {"status": "not_verified", "reason": "未下載模型/資料、未執行教材或訓練命令；閱讀報告不等程式或能力驗證。"},
        },
    })

auxiliary = []
for pid in ("readme-en", "omni-products", "decision-product"):
    folder = OUT / "auxiliary" / pid
    state = read(folder / "state.json")
    notes = sorted(folder.glob("note-*.json"))
    assert state["next_index"] == len(notes)
    raw = (ROOT / state["snapshot"]).read_bytes()
    assert sha(raw) == state["source_sha256"]
    issues = []
    for i, note_path in enumerate(notes):
        note = read(note_path)
        for issue in note.get("issues", []):
            entry = {**issue, "page_id": pid, "unit_index": i, "source_sha256": state["source_sha256"], "discovery": "independent_ai_first_read_after_canonical_route", "evidence": ref(note_path)}
            issues.append(entry)
            all_issues.append(entry)
    auxiliary.append({"page_id": pid, "source": state["source"], "source_sha256": state["source_sha256"], "snapshot": ref(state["snapshot"]), "canonical": False, "native_370_coverage": False, "unit_count": len(notes), "opening": ref(folder / "opening.json"), "notes": [ref(n) for n in notes], "verdict": "已足夠（操作與導覽範圍；評級尺度前文／執行／實際版面未驗證）" if pid == "decision-product" else "已足夠（文字操作／導覽範圍；執行／視聽／實際版面未驗證）", "issues": issues})

report = {
    "schema_version": 1,
    "kind": "reader-original",
    "role": "reader",
    "reviewer_task": REVIEWER,
    "created_at": datetime.now(UTC).isoformat(),
    "source_freeze": ref(freeze_path),
    "reading_plan": ref(OUT / "reading-plan.json"),
    "reader_profile": "數學基礎不錯的高中生；不預設熟悉ML縮寫。這是獨立AI首讀，不是真人學生測試。",
    "route": list(sessions),
    "source_policy": "canonical只讀freeze v19快照，由原生reader start/next逐段；非canonical三頁各先凍結原文，再自行逐段，明確排除native370頁覆蓋。",
    "independence": {"author_role": False, "technical_role": False, "project_context_read": False, "author_plans_read": False, "old_reviews_or_peer_answers_read": False, "actual_prior_context": "按指定路線先讀七個canonical頁，再讀三附頁；相似內容以已讀前文使用，不假稱每個概念在附頁仍是首次。"},
    "protocol_limitations": ["首讀前已讀SKILL與review-protocol，未另開calibration.md校準例；不宣稱該附錄已先讀。", "原判封存前未成功實看兩張圖或桌機/手機整頁；圖源碼核對不算視覺驗證。", "必要前文與折疊未全補讀；方法原理只在本頁定位相稱範圍判，未知明記。", "curriculum unit2有部分簡體輸入；保留原始紀錄，不事後改成另一份首讀。"],
    "counts": {"canonical_pages": len(pages), "canonical_units": sum(p["unit_count"] for p in pages), "noncanonical_pages": len(auxiliary), "noncanonical_units": sum(p["unit_count"] for p in auxiliary)},
    "pages": pages,
    "supplementary_pages": auxiliary,
    "issues": all_issues,
    "necessary_issue_ids": [x["id"] for x in all_issues if x.get("necessity") == "necessary"],
    "followup_policy": "保留全部原始判斷與後文釐清；此檔不含current、basis或closure。收到真實收件後，再由協調者另派修後或補查。",
}
path = OUT / "reader-original-v1.json"
write_new(path, report)
print(json.dumps({"report": ref(path), "counts": report["counts"], "necessary_issue_ids": report["necessary_issue_ids"], "issue_count": len(all_issues), "native_trace_audit": "7/7 passed", "created_at": report["created_at"]}, ensure_ascii=False, indent=2))
