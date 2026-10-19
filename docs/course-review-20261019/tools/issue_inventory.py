"""局部問題清單，不是發布閘門：列出草稿涵蓋的報告裡每一個要處置的問題，對照處置草稿的缺漏與逐筆條件。

用法：issue_inventory.py DRAFT --mode intake|sealed [--stage reader|technical|all]

草稿同 preflight.py，另可帶 issue_reports、additional_issue_inventories、dispositions、dispositions_collection、
tool_audit、dependency_audit（格式同 manifest）。應處置集合照 incremental_revision.check_dispositions 的規則算：
草稿各頁、各角色（依 --stage）proof 引用的報告（report、original_report、basis、checkpoint_closure、section_reports、
original_reports）、reader session 的全部 note、tool_audit 與 dependency_audit 的報告，以及 issue_reports 另列的
報告；用 incremental_revision.reported_issues 找出其中每個 issues／new_issues 底下的問題，每個「報告 SHA＋
pointer」各算一筆，再加上 additional_issue_inventories 的每一項。issue_reports 漏列必要報告時閘門會先擋下；
這裡照樣把那些報告的問題算進去，清單才完整。

輸出：issues 是應處置集合；missing_issue_reports 是 issue_reports 漏列的必要報告；undisposed 是有問題、沒處置；
unmatched_dispositions 是有處置、對不到問題；dispositions.failing 是逐筆條件不合的處置。任何一項不空就 exit 1。

驗證：應處置集合與實際處置以「報告 SHA＋pointer」逐筆對照；issue_reports 是否涵蓋必要報告；每筆處置的四個理由欄、
final_necessity、action、necessary 只能 fixed、fixed 要有 verification，且複查者是原提出者（提出者是作者或沒有
owner 時例外）、頁面複查是目前版本的 pass、工具複查列了現行 SHA 的檔，並涵蓋 affected_pages／affected_files；
處置表的 pending_count 與收件時間。intake 模式把收件時間與 seal 的關係列在 pending_seal。
不涵蓋：處置內容是否合理、修正是否真的做了；草稿沒列的頁與報告；閘門的其他檢查（見 preflight.py）。
"""

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTE = "局部問題清單，不是發布閘門：只依草稿列出的頁、報告與處置計算；全部通過也不代表閘門會通過。"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


preflight = load("preflight_for_issue_inventory", Path(__file__).resolve().with_name("preflight.py"))
review = preflight.review
CAUGHT = preflight.CAUGHT
message = preflight.message


def required_reports(draft, selected, errors):
    """搬自 check_dispositions：草稿各頁、各角色 proof 引用的報告，reader session 的 note，以及工具與依賴審閱。"""
    required, note_reports = {}, {}
    for page in draft.get("pages", []):
        for role in selected:
            try:
                proof = page.get(role, {})
                if role == "reader":
                    for key in ("basis", "checkpoint_closure"):
                        if key in proof:
                            ref = proof[key]["report"]
                            required[ref["sha256"]] = ref
                    for session in proof.get("sessions", []):
                        for ref in session["notes"]:
                            note_reports[ref["sha256"]] = ref
                for key in ("report", "original_report"):
                    if key in proof:
                        required[proof[key]["sha256"]] = proof[key]
                for key in ("section_reports", "original_reports"):
                    for ref in proof.get(key, []):
                        required[ref["sha256"]] = ref
            except CAUGHT as error:
                pid = page.get("page_id") if isinstance(page, dict) else repr(page)
                errors.append(f"{pid}/{role} 的 proof 格式不對：{message(error)}")
    try:
        if "technical" in selected and "tool_audit" in draft:
            ref = draft["tool_audit"]["report"]
            required[ref["sha256"]] = ref
        if "continuity" in selected and "dependency_audit" in draft:
            ref = draft["dependency_audit"]
            required[ref["sha256"]] = ref
    except CAUGHT as error:
        errors.append("tool_audit／dependency_audit 格式不對：" + message(error))
    return required, note_reports


def entry(ref, pointer, issue):
    found = {"report": ref, "json_pointer": pointer}
    if isinstance(issue, dict):
        found["id"] = issue.get("id", issue.get("issue_id"))
    return found


def expected_issues(root, draft, reports, errors):
    """搬自 check_dispositions：報告裡 reported_issues 找到的每一處，加上 additional_issue_inventories 的每一項。"""
    expected = {}
    for digest, ref in reports.items():
        try:
            report = review.document(root, ref)
        except CAUGHT as error:
            errors.append(f"讀不到報告 {ref.get('path')}: {message(error)}")
            continue
        for pointer in review.reported_issues(report):
            expected[(digest, pointer)] = entry(ref, pointer, review.node(report, pointer))
    for extra in draft.get("additional_issue_inventories", []):
        try:
            items = review.node(review.document(root, extra["report"]), extra["json_pointer"])
            review.require(isinstance(items, list), "additional issues must be a list")
        except CAUGHT as error:
            errors.append(f"additional_issue_inventories {extra.get('report', {}).get('path')}: {message(error)}")
            continue
        for i, item in enumerate(items):
            pointer = extra["json_pointer"] + "/" + str(i)
            expected[(extra["report"]["sha256"], pointer)] = entry(extra["report"], pointer, item)
    return expected


def collection(draft, dispositions, sealed, mode, errors, pending):
    """搬自 check_dispositions：處置的收件時間不早於處置表、不晚於現在，並落在收件 seal 的期間。"""
    collected = draft.get("dispositions_collection")
    if collected is None and mode == "intake":
        pending.append(
            "dispositions_collection 尚未寫：寫入後核對 dispositions.at ≤ received_at ≤ 現在，"
            "以及 received_at 落在收件 seal 的期間"
        )
        return
    try:
        collected = collected or {}
        received = review.at(collected["received_at"])
        review.require(
            review.at(dispositions["at"]) <= received <= datetime.now(UTC), "dispositions not actually collected yet"
        )
        phase = sealed[collected["wave"]].get(collected["role"])
        if phase:
            review.require(
                review.at(phase["started_at"]) <= received <= review.at(phase["completed_at"]),
                "dispositions outside closure phase",
            )
        else:
            # 閘門在那份 seal 不存在時不核這一項：intake 是還沒封，sealed 是草稿沒提供、這個 stage 也不要求那份 seal。
            pending.append(f"處置的 received_at 是否落在 {collected['wave']}/{collected['role']} seal 期間")
    except CAUGHT as error:
        errors.append("dispositions_collection: " + message(error))


def check_decision(root, draft, decision, live, tasks):
    """搬自 check_dispositions 對每一筆處置的條件。閘門遇到第一個錯就停；這裡列出這一筆的全部錯誤。"""
    errors = []

    def need(condition, text):
        if not condition:
            errors.append(text)

    try:
        review.node(review.document(root, decision["report"]), decision["json_pointer"])
    except CAUGHT as error:
        errors.append("處置指的問題讀不到：" + message(error))
    for field in ("source_support", "learning_benefit", "change_burden", "reason"):
        need(
            isinstance(decision.get(field), str) and decision[field].strip(), "incomplete author adjudication: " + field
        )
    need(decision.get("final_necessity") in {"necessary", "optional"}, "issue necessity not adjudicated")
    need(decision.get("action") in {"fixed", "keep", "defer", "reject"}, "issue still pending")
    need(
        decision.get("final_necessity") != "necessary" or decision.get("action") == "fixed", "necessary gap unresolved"
    )
    fixed = decision.get("action") == "fixed"
    if fixed:
        need(decision.get("verification"), "fix requires actual verification references")
    try:
        issue_owner = review.owner(review.document(root, decision["report"]))
    except CAUGHT:
        return errors  # 提出問題的報告讀不到，上面已記錯；無法核對複查者
    checked_pages, checked_files = set(), set()
    # 寫成 null 的清單當成空清單，錯記在這一筆，不中斷整份清單。
    for verification in decision.get("verification") or []:
        where = verification.get("report", {}).get("path") if isinstance(verification, dict) else verification
        try:
            report = review.document(root, verification["report"])
            task = review.owner(report)
            review.require(
                task in tasks and (task == issue_owner or issue_owner in draft["author_tasks"] or issue_owner is None),
                "fix was not rechecked by original reviewer",
            )
            if verification.get("kind") == "tool_audit":
                review.require(
                    report.get("verdict") == "pass" and decision["issue_id"] in report.get("resolved_issue_ids", []),
                    "tool fix lacks actual independent closure",
                )
                for checked in report.get("checked_files", []):
                    if checked.get("sha256") == review.sha(review.local(root, checked["path"]).read_bytes()):
                        checked_files.add(checked["path"])
                continue
            value = review.node(report, verification["json_pointer"])
            pid = value.get("page_id", value.get("lesson_id"))
            review.require(pid in live and value.get("verdict") == "pass", "fix lacks reviewer pass on affected page")
            review.require(value.get("source_sha256") == live[pid]["source_sha256"], "fix recheck is stale")
            review.require(
                value.get("figures_sha256", value.get("figure_sha256")) == live[pid]["figures_sha256"],
                "fix figure recheck stale",
            )
            checked_pages.add(pid)
        except CAUGHT as error:
            errors.append(f"verification {where}: {message(error)}")
    if fixed:
        affected = set(decision.get("affected_pages") or [])
        files = set(decision.get("affected_files") or [])
        need(
            (affected or files) and affected <= checked_pages and files <= checked_files,
            "fix does not cover affected pages/files",
        )
    return errors


def describe(decision):
    if not isinstance(decision, dict):
        return {"disposition": decision}
    return {
        "report": decision.get("report"),
        "json_pointer": decision.get("json_pointer"),
        "issue_id": decision.get("issue_id"),
    }


def inventory(root, draft, live, mode, stage):
    root = Path(root).resolve()
    selected = preflight.selected_roles(stage)
    errors, pending = [], []
    task_roles, _ = preflight.assignments(root, draft, errors)
    waves = draft.get("review_waves", [])
    sealed = preflight.seals(root, waves, selected, mode, errors) if waves else {}
    required, note_reports = required_reports(draft, selected, errors)
    listed = {}
    for ref in draft.get("issue_reports", []):
        try:
            listed[ref["sha256"]] = ref
        except CAUGHT as error:
            errors.append("issue_reports 的一筆格式不對：" + message(error))
    missing = [ref for digest, ref in required.items() if digest not in listed]
    expected = expected_issues(root, draft, required | listed | note_reports, errors)
    actual, unmatched, failing = set(), [], []
    checked = 0
    dispositions = None
    if "dispositions" not in draft:
        errors.append("草稿沒有 dispositions：每個問題都還沒有處置")
    else:
        try:
            dispositions = review.document(root, draft["dispositions"])
        except CAUGHT as error:
            errors.append("dispositions: " + message(error))
    if dispositions is not None:
        # 搬自 check_dispositions：處置表不能還有待決的項目。
        if dispositions.get("pending_count") != 0:
            errors.append("unresolved author dispositions")
        collection(draft, dispositions, sealed, mode, errors, pending)
        for decision in dispositions.get("items", []):
            checked += 1
            if not isinstance(decision, dict):
                failing.append({**describe(decision), "errors": ["處置不是 JSON object"]})
                continue
            problems = []
            try:
                key = (decision["report"]["sha256"], decision["json_pointer"])
            except CAUGHT as error:
                key = None
                problems.append("處置缺少 report 或 json_pointer：" + message(error))
            if key is not None:
                if key in actual:
                    problems.append("duplicate issue disposition")
                actual.add(key)
                if key not in expected:
                    unmatched.append(describe(decision))
            problems += check_decision(root, draft, decision, live, task_roles)
            if problems:
                failing.append({**describe(decision), "errors": problems})
    result = {
        "tool": "issue_inventory",
        "note": NOTE,
        "mode": mode,
        "stage": stage,
        "gate_module": preflight.gate_identity(),
        "tool_errors": preflight.drift(),
        "errors": errors,
        "pending_seal": pending,
        "required_reports": list(required.values()),
        "missing_issue_reports": missing,
        "issues": list(expected.values()),
        "undisposed": [value for key, value in expected.items() if key not in actual],
        "unmatched_dispositions": unmatched,
        "dispositions": {"checked": checked, "failing": failing},
    }
    names = ("tool_errors", "errors", "missing_issue_reports", "undisposed", "unmatched_dispositions")
    result["failing"] = [name for name in names if result[name]] + (["dispositions"] if failing else [])
    return result


def run(draft_path, mode, stage, root=ROOT, live=None):
    """讀草稿、算目前清單、印出結果 JSON，回傳 exit code。"""
    draft = json.loads(Path(draft_path).read_bytes())
    if live is None:
        try:
            live = preflight.current_pages(root)
        except CAUGHT as error:
            failure = {"tool": "issue_inventory", "note": NOTE, "tool_errors": ["算不出目前清單：" + message(error)]}
            print(json.dumps(failure, ensure_ascii=False, indent=1))
            return 1
    result = inventory(root, draft, live, mode, stage)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 1 if result["failing"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("draft", type=Path, help="草稿 manifest（JSON）")
    parser.add_argument("--mode", required=True, choices=("intake", "sealed"))
    parser.add_argument("--stage", default="all", choices=("reader", "technical", "all"))
    args = parser.parse_args()
    raise SystemExit(run(args.draft, args.mode, args.stage))


if __name__ == "__main__":
    main()
