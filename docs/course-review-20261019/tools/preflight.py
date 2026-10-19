"""局部 preflight，不是發布閘門：草稿 manifest 列出的每一頁、每個角色，跑 audit_manifest 頁面迴圈的檢查。

用法：preflight.py DRAFT --mode intake|sealed [--stage reader|technical|all]

草稿是最終 manifest 的子集：source_freeze、assignments、review_waves、author_tasks，加上要檢查的 pages[]
（格式同 manifest 的 pages[]）。草稿沒列的頁不檢查。--stage 同閘門：reader 只核讀者，technical 加上技術，
all 三個角色都核。

- intake：stage 還開著、seal 還沒寫。依賴 seal 的項目（報告是否列在 seal、session 與 closure 的時間是否落在
  seal 內、派工時間窗、stage 先後）在那份 seal 寫進草稿以前列在 pending_seal，不算錯；草稿已有的 seal 照常核對。
- sealed：草稿的 review_waves 已有寫好的 seal，全部檢查都跑。報告的收件 seal 草稿沒提供、這個 stage 也不要求時
  （例如 --stage reader、草稿只有 reader seal，而報告記在 technical 階段收件），閘門不核那一項，這裡同樣列在
  pending_seal。

判準不另寫：incremental_revision 的函式能直接呼叫就呼叫；閘門迴圈裡內聯的 require 照原樣搬來，旁邊註明搬自
哪一段，錯誤訊息保留閘門的英文原文，方便和閘門的輸出對照。intake 模式的 reader_closure 拆開執行，見
intake_reader_closure。搬來的段落以 TRANSCRIBED 綁住閘門原始碼；閘門改了，tool_errors 會要求重新對照。

輸出 JSON：shared 是各頁共用的派工表、seal 與派工時間窗；pages[頁]["page"] 是頁層級的凍結核對，其餘鍵是
角色；coordinator_checks 另列三項協調者要查、閘門不查或只查一部分的事，不和閘門的檢查混在一起；failing 列出
有錯的位置。
只要有任何錯誤就 exit 1；pending_seal 不算錯。

驗證：草稿各頁的頁層級凍結核對（目前清單 6 欄等於凍結檔、snapshot 是目前正文），各角色的 inherit 條件、
section-composite 的小節涵蓋與 _validate、review 證據的收件、owner、派工範圍、版本、verdict、圖與原始判斷、
reader 的 session 與 reader_closure、technical 的 _validate；派工表、seal 先後與派工時間窗。協調者另核：reader 的
basis 版本有一個列在 sessions、讀完全部單元的 session 讀過；每個 session 的閱讀計畫存在、SHA 相符、main_pages
含該頁、凍結檔與 session 相同；technical 報告的 figure_sha256 恰好等於凍結檔這一頁的圖。
不涵蓋：閘門在頁面迴圈以外的檢查（重放舊輪次、baseline 身分、全書頁集合、implementation 與 criteria 雜湊、
Notebook 與索引重建、凍結圖副本、工具審閱、處置與依賴審閱；處置見 issue_inventory.py）；草稿沒列的頁；
審閱內容是否正確、派工是否真的發生、seal 是否如實。
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
GATE = "docs/review-tools/incremental_revision.py"
NOTE = (
    "局部 preflight，不是發布閘門：只核對草稿列出的頁與角色，以及草稿的派工表、seal 與派工時間窗；"
    "閘門在頁面迴圈以外的檢查不在這裡跑，全部通過也不代表閘門會通過。"
)
# 閘門接住的例外，再加上 AttributeError：草稿某一頁的格式錯（例如該是 dict 的地方是字串）不擋住其他頁。
CAUGHT = (ValueError, KeyError, TypeError, OSError, StopIteration, RuntimeError, IndexError, AttributeError)
FIELDS = ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256")
# 本工具與 issue_inventory.py 搬寫內聯檢查時，閘門對應原始碼（gate_segments）的 SHA-256。
TRANSCRIBED = {
    "audit_manifest": "6d524e6e9b1242bcaf8341dc1b8de44c23fdd9d35956e155c0515000c1f8d479",
    "reader_closure": "3107611701489ad4dfe4a6d415c968499a44770633e4d43897dbb90d8dc514ed",
    "check_dispositions": "f84efefb01bd4a11253292ce9ce2596b0505939720d0e76cd2c3c9ee4f859088",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


review = load("incremental_revision_for_preflight", ROOT / GATE)
ROLES = review.ROLES


def message(error):
    """閘門 require 的訊息原樣保留；其他例外加上類別名，像 KeyError 只帶鍵名的訊息才看得懂。"""
    return str(error) if type(error) is ValueError else f"{type(error).__name__}: {error}"


def section():
    return {"errors": [], "pending_seal": []}


def gate_segments():
    """搬寫所依據的閘門原始碼：audit_manifest 從作者排除名單到頁面迴圈結束，以及兩個整個函式。

    終點取迴圈後的第一個敘述，迴圈尾端新加的檢查也會改變這一段。
    """
    source = inspect.getsource(review.audit_manifest)
    start = source.rfind("\n", 0, source.index('authors = set(manifest.get("author_tasks", []))')) + 1
    end = source.index('if stage in {"technical", "all"}:', source.index("records.append(", start))
    return {
        "audit_manifest": source[start:end],
        "reader_closure": inspect.getsource(review.reader_closure),
        "check_dispositions": inspect.getsource(review.check_dispositions),
    }


def drift():
    """閘門被改過、搬來的檢查可能已過期時，回傳要重新對照的段落。"""
    try:
        segments = gate_segments()
    except (OSError, ValueError) as error:
        return ["在 audit_manifest 找不到搬寫段落的起點或終點，閘門結構已變：" + message(error)]
    errors = []
    for name, text in segments.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != TRANSCRIBED[name]:
            errors.append(
                f"閘門 {name} 的原始碼和搬寫時不同（現行 SHA-256 {digest}）：先對照 preflight.py 與 issue_inventory.py "
                f"裡註明搬自 {name} 的段落並更新，再把 preflight.TRANSCRIBED[{name!r}] 換成現行值。"
            )
    return errors


def gate_identity():
    return {"path": GATE, "sha256": hashlib.sha256((ROOT / GATE).read_bytes()).hexdigest()}


def selected_roles(stage):
    # 同 audit_manifest：reader 只核讀者，technical 加上技術，all 三個角色。
    return {"reader": ROLES[:1], "technical": ROLES[:2], "all": ROLES}[stage]


def assignments(root, draft, errors):
    """搬自 audit_manifest 的派工表迴圈：/root/ 底下的 task、不是作者、角色合法、同一 task 只派一次、fresh。

    閘門遇到第一筆不合格就停；這裡記下錯誤並略過那一筆，其餘照常收進 task_roles。
    """
    authors = set(draft.get("author_tasks", []))
    task_roles, task_assignments = {}, {}
    try:
        tasks = review.document(root, draft["assignments"])["tasks"]
    except CAUGHT as error:
        errors.append("assignments: " + message(error))
        return task_roles, task_assignments
    for assignment in tasks:
        try:
            task, role = assignment["task"], assignment["role"]
            review.require(task.startswith("/root/") and task not in authors, "invalid independent task")
            review.require(role in ROLES and task not in task_roles, "task reused across assignments")
            review.require(assignment.get("context") == "fresh", "review task context is not fresh")
        except CAUGHT as error:
            name = assignment.get("task") if isinstance(assignment, dict) else repr(assignment)
            errors.append(f"派工 {name}: {message(error)}")
            continue
        task_roles[task] = role
        task_assignments[task] = assignment
    return task_roles, task_assignments


def seals(root, waves, selected, mode, errors):
    """incremental_revision.review_seals：讀出 seal 並核對 stage 先後。

    sealed 模式照閘門傳入 stage 的角色，缺 seal 就是錯；intake 模式只讀草稿已寫的 seal。讀不過時記錯，再逐份
    讀出還能讀的 seal，讓各頁的檢查仍能進行。
    """
    try:
        return review.review_seals(root, waves, selected if mode == "sealed" else ())
    except CAUGHT as error:
        errors.append("review_seals: " + message(error))
    sealed = {}
    for wave in waves:
        try:
            found = sealed.setdefault(wave["id"], {})
            references = list(wave["seals"].items())
        except CAUGHT as error:
            errors.append(f"review_waves 的一筆 wave 格式不對：{message(error)}")
            continue
        for role, reference in references:
            try:
                found[role] = review.document(root, reference)
            except CAUGHT as error:
                errors.append(f"seal {wave['id']}/{role}: {message(error)}")
    return sealed


def technical_evidence(root, draft, errors):
    """incremental_revision.technical_reports：technical 報告的 lesson_id 不重複，並記下每個 task 擁有的頁。

    整份草稿一起讀不過時記錯，再逐頁讀出還能讀的報告，讓其他頁的 _validate 仍能核對 task 是否被重用。
    """
    try:
        return review.technical_reports(draft, root)
    except CAUGHT as error:
        errors.append("technical_reports: " + message(error))
    reports, owners = {}, {}
    for page in draft.get("pages", []):
        try:
            found, found_owners = review.technical_reports({"pages": [page]}, root)
        except CAUGHT:
            continue  # 這一頁的 technical 檢查會再讀一次報告，錯誤記在那一頁
        reports.update(found)
        for task, keys in found_owners.items():
            owners.setdefault(task, []).extend(keys)
    return reports, owners


def setup(root, draft, mode, stage):
    """頁面迴圈之前 audit_manifest 先算好的東西：派工表、technical 報告、seal 與派工時間窗。"""
    shared = section()
    selected = selected_roles(stage)
    # 搬自 audit_manifest：作者排除名單要含 /root。
    if "/root" not in set(draft.get("author_tasks", [])):
        shared["errors"].append("missing actual author exclusions")
    task_roles, task_assignments = assignments(root, draft, shared["errors"])
    reports, owners = technical_evidence(root, draft, shared["errors"])
    waves = draft.get("review_waves", [])
    sealed = {}
    if not waves:
        shared["errors"].append("missing actual review chronology")
    else:
        sealed = seals(root, waves, selected, mode, shared["errors"])
    if mode == "intake":
        for wave_id, found in sealed.items():
            for role in selected:
                if role not in found:
                    shared["pending_seal"].append(f"{wave_id}/{role} seal 尚未寫入：寫入後核對 stage 先後")
    for task, assignment in task_assignments.items():
        role = assignment["role"]
        if role not in selected:
            continue
        try:
            review.require(assignment["wave"] in sealed, f"wave {assignment['wave']} 不在草稿的 review_waves")
            phase = sealed[assignment["wave"]].get(role)
            window = assignment["dispatch_window"]
            if phase is None and mode == "intake":
                # seal 寫好以前，只能先核對時間窗本身沒有倒置。
                review.require(
                    review.at(window["not_before"]) <= review.at(window["not_after"]),
                    "task dispatch window outside stage interval",
                )
                shared["pending_seal"].append(f"派工時間窗 {task}：是否落在 {assignment['wave']}/{role} seal 期間")
                continue
            review.require(phase is not None, f"{assignment['wave']}/{role} seal 沒有提供")
            # 搬自 audit_manifest：派工時間窗要落在同 wave、同角色 seal 的期間內。
            review.require(
                review.at(phase["started_at"])
                <= review.at(window["not_before"])
                <= review.at(window["not_after"])
                <= review.at(phase["completed_at"]),
                "task dispatch window outside stage interval",
            )
        except CAUGHT as error:
            shared["errors"].append(f"派工 {task}: {message(error)}")
    return SimpleNamespace(
        mode=mode,
        selected=selected,
        shared=shared,
        task_roles=task_roles,
        task_assignments=task_assignments,
        technical_reports=reports,
        technical_owners=owners,
        reader_tasks={task for task, role in task_roles.items() if role == "reader"},
        sealed=sealed,
        cache={},
    )


def tool(ctx, name, path):
    """只在需要時載入閘門用到的其他工具，都從本工具所在的 repo 載入。"""
    if name not in ctx.cache:
        ctx.cache[name] = load(name + "_for_preflight", ROOT / path)
    return ctx.cache[name]


def baseline_pages(root, ctx):
    """閘門的 old 來自 manifest 的 baseline_freeze，而閘門要求它和下面這份 source_freeze 位元組相同。

    草稿通常不帶 baseline_freeze，所以直接讀 docs/course-publish-20261008/review-manifest.json 的 source_freeze。
    """
    if "baseline pages" not in ctx.cache:
        policy = json.loads((root / "docs/course-publish-20261008/review-manifest.json").read_bytes())
        pages = review.document(root, policy["source_freeze"])["pages"]
        ctx.cache["baseline pages"] = {p["page_id"]: p for p in pages}
    return ctx.cache["baseline pages"]


def inherit(root, ctx, pid, proof, page):
    """搬自 audit_manifest 的 inherit 分支：正文與圖和 baseline 相同，Notebook 等於 baseline 的 live_sha256。"""
    old = baseline_pages(root, ctx)
    inherited_identity = pid in old and all(
        page.get(f) == old[pid].get(f) for f in ("source", "selector", "source_sha256", "figures_sha256")
    )
    review.require(inherited_identity, "changed source cannot inherit old verdict: " + pid)
    if page.get("notebook"):
        old_notebook = old[pid].get("notebook", {})
        review.require(old_notebook.get("live_sha256") == page["notebook_sha256"], "changed Notebook cannot inherit")
    review.require(proof.get("baseline_page_id") == pid, "inherited page mismatch")
    review.require(
        isinstance(proof.get("dependency_note"), str) and proof["dependency_note"].strip(),
        "inheritance needs scope/route disposition",
    )


def section_composite(root, ctx, pid, proof, page, frozen_page, result):
    """搬自 audit_manifest 的 section-composite 分支：只限 whole-file 頁，導言不變，變動小節恰好等於報告集合，
    每節由派給該節的 technical task 審閱並通過 _validate，收件用 incremental_revision.require_collected。

    閘門遇到第一節有錯就停；這裡逐條列出每一節的 _validate 錯誤。
    """
    old = baseline_pages(root, ctx)
    review.require(pid in old and page["selector"] == "whole-file", "section composite needs original whole page")
    reading_time = tool(ctx, "reading_time", "scripts/reading_time.py")
    # 閘門用 read_text() 而不指定編碼；照搬，結果才和閘門一致。
    old_text = review.local(root, old[pid]["snapshot"]).read_text()
    live_text = review.local(root, frozen_page["snapshot"]).read_text()
    old_sections = reading_time.lesson_slices(old_text)
    live_sections = reading_time.lesson_slices(live_text)
    review.require(set(old_sections) <= set(live_sections), "original numbered section removed")
    review.require(
        reading_time.chapter_introduction(old_text) == reading_time.chapter_introduction(live_text),
        "changed page introduction needs separate factual review",
    )
    changed = {key for key, body in live_sections.items() if body != old_sections.get(key)}
    supplied = {review.document(root, ref)["lesson_id"] for ref in proof["section_reports"]}
    review.require(changed == supplied, "section factual reviews do not cover exact changed sections")
    validator = tool(ctx, "check_technical_reviews", "scripts/check_technical_reviews.py")
    for key in sorted(changed):
        report = ctx.technical_reports[key]
        review.require(ctx.task_roles.get(review.owner(report)) == "technical", "wrong section reviewer role")
        assignment = ctx.task_assignments[review.owner(report)]
        review.require(
            assignment["wave"] == proof["wave"] and assignment.get("lesson_id") == key,
            "section outside actual assignment",
        )
        errors = validator._validate(
            root,
            review.local(root, page["source"]),
            key,
            live_sections[key],
            report,
            ctx.technical_owners,
            ctx.reader_tasks,
        )
        result["errors"].extend(f"_validate {key}: {error}" for error in errors)
    if review.collection_seal(proof, "technical", ctx.sealed) is None and ctx.mode == "intake":
        result["pending_seal"].append(
            f"小節報告是否列在 {proof['wave']}/{proof.get('collection_role', 'technical')} seal 的 report_sha256s"
        )
    else:
        review.require_collected(proof, "technical", ctx.sealed, proof["section_reports"])


def intake_reader_closure(root, proof, current, task, pid, sealed, pending):
    """收件模式的 incremental_revision.reader_closure：子檢查照原樣搬來，依賴 seal 的在 seal 寫好前改記 pending。

    延到 seal 寫好才核對的子檢查（草稿已提供那份 seal 時照常核對）：
    - 原始判斷列在 reader seal 的 report_sha256s；
    - 每個 session 的 opening 與最後 checkpoint 落在所屬收件 seal 的期間；
    - closure 的建立時間不早於 reader seal 開始；收件角色不是 reader 時，也不早於 reader seal 結束；
    - closure 的 received_at 落在收件 seal 的期間，closure 報告列在那份 seal。
    其餘子檢查不依賴 seal，照常執行：basis 與目前判斷的版本、圖與連結，ai_estimate，每個 session 完整
    （reader_session），至少一個 session 讀的是原始判斷的版本，closure 的版本、圖與 basis_report 連結，closure
    不早於各次觀察與判斷、created_at ≤ received_at ≤ 現在，以及逐項收束恰好涵蓋非「已足夠」的概念。
    """
    reader_seal = sealed[proof["wave"]].get("reader")
    original = review.document(root, proof["original_report"])
    original_page = review.page_member(original, proof["original_json_pointer"], pid, task)
    if reader_seal is None:
        pending.append(
            f"原始判斷 {proof['original_report']['path']} 是否列在 {proof['wave']}/reader seal 的 report_sha256s"
        )
    else:
        review.require(
            proof["original_report"]["sha256"] in reader_seal["report_sha256s"], "original report not sealed"
        )
    review.require(current.get("basis_report"), "current reader judgment lacks actual basis")
    basis = review.document(root, proof["basis"]["report"])
    basis_page = review.page_member(basis, proof["basis"]["json_pointer"], pid, task)
    review.require(
        basis_page["source_sha256"] == current["source_sha256"], "current judgment lacks read revised source"
    )
    basis_declared = current["basis_report"]
    declared_path = basis_declared.get("path") if isinstance(basis_declared, dict) else basis_declared
    review.require(declared_path == proof["basis"]["report"]["path"], "current basis link mismatch")
    if isinstance(basis_declared, dict) and "sha256" in basis_declared:
        review.require(basis_declared["sha256"] == proof["basis"]["report"]["sha256"], "declared basis digest mismatch")
    verdict = basis_page.get("verdict", "")
    review.require(
        isinstance(verdict, str)
        and (
            verdict == "pass"
            or (
                "已足夠" in verdict
                and not any(word in verdict for word in ("需補說明", "revise", "待修", "必要缺口未解"))
            )
        ),
        "basis still requests revision or is not a string",
    )
    review.require(
        basis_page.get("figures_sha256", basis_page.get("figure_sha256")) == current["figures_sha256"],
        "basis figure version mismatch",
    )
    estimate = current.get("ai_estimate", {})
    lo = estimate.get("minutes_min", estimate.get("min_minutes"))
    hi = estimate.get("minutes_max", estimate.get("max_minutes"))
    review.require(
        type(lo) is int and type(hi) is int and 0 < lo <= hi and estimate.get("reason"), "missing reader estimate"
    )
    observed = {}
    matched_original = False
    # 閘門從 reader seal 開始時間起算最後觀察時間；seal 未寫時只取各 session 的最後 checkpoint，下限另記 pending。
    latest_observation = review.at(reader_seal["started_at"]) if reader_seal else None
    for reference in proof["sessions"]:
        opening = review.document(root, reference["opening"])
        if opening["source_sha256"] == original_page["source_sha256"]:
            matched_original = True
        _, completed = review.reader_session(root, reference, task)
        latest_observation = completed if latest_observation is None else max(latest_observation, completed)
        wave, role = reference.get("wave", proof["wave"]), reference.get("collection_role", "reader")
        phase = sealed[wave].get(role)
        if phase is None:
            pending.append(
                f"session {opening.get('session', reference['opening']['path'])}："
                f"opening 與最後 checkpoint 是否落在 {wave}/{role} seal 期間"
            )
        else:
            review.require(
                review.at(phase["started_at"])
                <= review.at(opening["started_at"])
                <= completed
                <= review.at(phase["completed_at"]),
                "reader checkpoint outside actual phase",
            )
        for ref in reference["notes"]:
            for i, concept in enumerate(review.document(root, ref)["concept_checks"]):
                if concept["verdict"] != "已足夠":
                    observed[(ref["sha256"], i)] = concept["verdict"]
    review.require(matched_original, "original judgment has no matching first-read source")
    if not observed and "checkpoint_closure" not in proof:
        return
    closure_ref = proof["checkpoint_closure"]
    closure_report = review.document(root, closure_ref["report"])
    closure = review.page_member(closure_report, closure_ref["json_pointer"], pid, task)
    review.require(
        closure.get("verdict") == "pass" and closure["source_sha256"] == current["source_sha256"],
        "reader closure stale/open",
    )
    review.require(closure.get("figures_sha256") == current["figures_sha256"], "reader closure figures stale")
    review.require(
        closure_report.get("basis_report") == proof["report"], "checkpoint closure not linked to own current judgment"
    )
    created = review.at(closure_report["created_at"])
    review.require(created >= latest_observation, "closure predates its observations")
    if reader_seal is None:
        pending.append(f"closure 建立時間是否不早於 {proof['wave']}/reader seal 開始")
        if closure_ref["collection_role"] != "reader":
            pending.append(
                f"closure 在 {closure_ref['collection_role']} 階段收件：建立時間是否不早於 {proof['wave']}/reader seal 結束"
            )
    elif closure_ref["collection_role"] != "reader":
        review.require(
            created >= review.at(reader_seal["completed_at"]), "later closure predates sealed current judgment"
        )
    for data in (basis, review.document(root, proof["report"])):
        for key in ("created_at", "confirmed_at"):
            if isinstance(data.get(key), str) and "T" in data[key]:
                review.require(created >= review.at(data[key]), "closure predates its basis judgment")
    received = review.at(closure_ref["received_at"])
    lower = review.at(reader_seal["started_at"]) if reader_seal else created
    review.require(lower <= created <= received <= datetime.now(UTC), "reader closure chronology invalid")
    closing_seal = sealed[closure_ref["wave"]].get(closure_ref["collection_role"])
    if closing_seal:
        review.require(
            review.at(closing_seal["started_at"]) <= received <= review.at(closing_seal["completed_at"]),
            "closure outside followup phase",
        )
        review.require(
            closure_ref["report"]["sha256"] in closing_seal["report_sha256s"], "closure not collected before phase end"
        )
    else:
        pending.append(
            f"closure 的 received_at 是否落在 {closure_ref['wave']}/{closure_ref['collection_role']} seal 期間，"
            f"報告 {closure_ref['report']['path']} 是否列在那份 seal"
        )
    resolved = {}
    for item in closure.get("checkpoint_resolutions", []):
        key = (item["note"]["sha256"], item["concept_index"])
        review.require(key not in resolved, "duplicate checkpoint closure")
        review.artifact(root, item["note"])
        review.require(
            item.get("final_status") in {"resolved", "nonblocking", "deferred"}, "reader necessary gap still open"
        )
        review.require(item.get("reason") and item.get("evidence"), "reader closure lacks actual reason/evidence")
        resolved[key] = item["original_verdict"]
    review.require(observed == resolved, "original non-pass concept lacks reader final closure")


def review_evidence(root, ctx, pid, proof, role, page, frozen_page, result):
    """搬自 audit_manifest 頁面迴圈的 review 分支，reader 與 technical 的專屬部分照閘門呼叫模組函式。"""
    review.require(proof.get("mode") == "review", "missing current role evidence: " + pid + "/" + role)
    report = review.document(root, proof["report"])
    phase = review.collection_seal(proof, role, ctx.sealed)
    if phase:
        review.require(
            proof["report"]["sha256"] in phase["report_sha256s"], "report was not collected before stage closure"
        )
    else:
        # 閘門在那份 seal 不存在時不核這一項：intake 是還沒封，sealed 是草稿沒提供、這個 stage 也不要求那份 seal。
        result["pending_seal"].append(
            f"報告 {proof['report']['path']} 是否列在 {proof['wave']}/{proof.get('collection_role', role)} seal "
            "的 report_sha256s"
        )
    task = review.owner(report)
    review.require(ctx.task_roles.get(task) == role, "undeclared or wrong-role reviewer")
    assignment = ctx.task_assignments[task]
    review.require(assignment["wave"] == proof["wave"], "review wave outside actual assignment")
    review.require(
        pid in assignment.get("pages", []) if role != "technical" else assignment.get("lesson_id") == pid,
        "page outside actual assignment",
    )
    current = review.node(report, proof.get("json_pointer", ""))
    review.require(current.get("page_id", current.get("lesson_id")) == pid, "report page membership mismatch")
    review.require(current.get("source_sha256") == page["source_sha256"], "report source is stale: " + pid)
    review.require(current.get("verdict") in {"pass", "已足夠"}, "reviewer requests revision: " + pid)
    figures = current.get("figures_sha256", current.get("figure_sha256"))
    review.require(figures == page["figures_sha256"], "review figure identity incomplete: " + pid)
    review.require(proof.get("original_report"), "original judgment must be retained")
    original_report = review.document(root, proof["original_report"])
    review.page_member(original_report, proof.get("original_json_pointer", ""), pid, task)
    if role == "reader":
        sessions = proof.get("sessions", [])
        review.require(sessions, "no actual sequential first-read trace")
        for reference in sessions:
            review.require(
                review.document(root, reference["opening"])["page_id"] == pid, "reader session belongs to another page"
            )
        if ctx.mode == "sealed":
            review.reader_closure(root, proof, current, task, pid, ctx.sealed[proof["wave"]]["reader"], ctx.sealed)
        else:
            intake_reader_closure(root, proof, current, task, pid, ctx.sealed, result["pending_seal"])
    if role == "technical":
        # 閘門用 read_text() 而不指定編碼；照搬，結果才和閘門一致。
        body = review.local(root, frozen_page["snapshot"]).read_text()
        validator = tool(ctx, "check_technical_reviews", "scripts/check_technical_reviews.py")
        errors = validator._validate(
            root, review.local(root, page["source"]), pid, body, report, ctx.technical_owners, ctx.reader_tasks
        )
        # 閘門把全部錯誤合成一句；這裡逐條列出。
        result["errors"].extend("_validate: " + error for error in errors)


def check_page(root, ctx, pid, row, page, frozen_page):
    result = {"page": section()}
    try:
        # 搬自 audit_manifest 頁面迴圈開頭：目前清單的 6 欄要等於凍結檔。閘門只報頁名，這裡另列不同的欄位。
        stale = [field for field in FIELDS if page.get(field) != frozen_page.get(field)]
        review.require(
            not stale,
            "current source/figure/notebook freeze stale: " + pid + "（不同的欄位：" + "、".join(stale) + "）",
        )
        # 搬自同一段：凍結檔記的 snapshot 要是目前正文的位元組。
        review.require(
            review.sha(review.artifact(root, {"path": frozen_page["snapshot"], "sha256": page["source_sha256"]}))
            == page["source_sha256"],
            "missing source snapshot",
        )
    except CAUGHT as error:
        result["page"]["errors"].append(message(error))
    for role in ctx.selected:
        found = section()
        proof = row.get(role, {})
        try:
            if proof.get("mode") == "inherit":
                inherit(root, ctx, pid, proof, page)
            elif role == "technical" and proof.get("mode") == "section-composite":
                section_composite(root, ctx, pid, proof, page, frozen_page, found)
            else:
                review_evidence(root, ctx, pid, proof, role, page, frozen_page, found)
        except CAUGHT as error:
            found["errors"].append(message(error))
        result[role] = found
    return result


def basis_read(root, proof):
    """協調者核對：目前判斷的 basis 頁，要有一個列在 sessions、讀完全部單元的 session 讀的是同一版正文與圖。

    閘門核 basis 是目前版本、每個 session 都讀完、至少一個 session 讀的是原始判斷的版本，但不核有沒有 session
    讀過 basis 那一版。「讀完全部單元」用 incremental_revision.reader_session 判斷：note 數等於單元數，逐單元相符。
    """
    task = review.owner(review.document(root, proof["report"]))
    basis = review.node(review.document(root, proof["basis"]["report"]), proof["basis"]["json_pointer"])
    version = (basis.get("source_sha256"), basis.get("figures_sha256", basis.get("figure_sha256")))
    rejected = []
    for reference in proof.get("sessions", []):
        try:
            opening = review.document(root, reference["opening"])
            review.reader_session(root, reference, task)
        except CAUGHT as error:
            rejected.append(f"session {reference.get('opening', {}).get('path')} 不算讀完：{message(error)}")
            continue
        if (opening.get("source_sha256"), opening.get("figures_sha256")) == version:
            return []
    where = proof["basis"]["report"]["path"] + "#" + proof["basis"]["json_pointer"]
    return [
        f"basis {where} 的版本（source_sha256 {str(version[0])[:12]}…）沒有任何列在 sessions、讀完全部單元的 "
        "session 讀過：最後的判斷沒有讀過新版本的紀錄"
    ] + rejected


def session_plans(root, proof, pid):
    """協調者核對：每個列在 sessions 的 session，state.json 指向的閱讀計畫存在、SHA 相符、main_pages 含這一頁。

    state.json 每讀一個單元就被改寫，沒有 SHA 綁定；所以另核它的 reading_plan 與有綁定的 opening.json 相同，
    並核閱讀計畫的凍結檔就是這個 session 讀的凍結檔。
    """
    errors = []
    for reference in proof.get("sessions", []):
        name = reference.get("opening", {}).get("path")
        try:
            opening = review.document(root, reference["opening"])
            state = json.loads((review.local(root, reference["opening"]["path"]).parent / "state.json").read_bytes())
            review.require(state.get("page_id") == pid, "state.json 屬於另一頁")
            review.require(state.get("reading_plan"), "state.json 沒有 reading_plan：這個 session 不是依閱讀計畫開的")
            review.require(
                state["reading_plan"] == opening.get("reading_plan"), "state.json 與 opening.json 的 reading_plan 不同"
            )
            plan = review.document(root, state["reading_plan"])
            review.require(pid in plan.get("main_pages", []), "閱讀計畫的 main_pages 不含這一頁")
            review.require(
                plan.get("source_freeze") == opening.get("source_freeze"), "閱讀計畫的凍結檔不是這個 session 讀的凍結檔"
            )
        except CAUGHT as error:
            errors.append(f"session {name}: {message(error)}")
    return errors


def technical_figures(root, proof, frozen_page):
    """協調者核對：technical 報告的 figure_sha256 恰好等於凍結檔這一頁的圖，不多列先備圖、也不漏列。

    閘門的頁面迴圈先取 figures_sha256、沒有才取 figure_sha256。報告只有 figure_sha256 時，閘門已核它等於目前
    清單的圖，這一項和閘門重疊；兩個鍵都有時，figure_sha256 只經過 _validate，而 _validate 允許多列圖。這裡
    固定核 figure_sha256，並對照凍結檔。
    """
    current = review.node(review.document(root, proof["report"]), proof.get("json_pointer", ""))
    if current.get("figure_sha256") != frozen_page["figures_sha256"]:
        return [
            "figure_sha256 不等於凍結檔這一頁的圖："
            + json.dumps(current.get("figure_sha256"), ensure_ascii=False)
            + " ≠ "
            + json.dumps(frozen_page["figures_sha256"], ensure_ascii=False)
        ]
    return []


def coordinator_checks(root, ctx, pid, row, frozen_page):
    result = {}
    checks = []
    reader = row.get("reader", {})
    if "reader" in ctx.selected and reader.get("mode") == "review":
        checks += [
            ("basis_read_by_complete_session", lambda: basis_read(root, reader)),
            ("session_reading_plans", lambda: session_plans(root, reader, pid)),
        ]
    technical = row.get("technical", {})
    if "technical" in ctx.selected and technical.get("mode") == "review":
        checks.append(("technical_figure_inventory", lambda: technical_figures(root, technical, frozen_page)))
    for name, check in checks:
        try:
            result[name] = {"errors": check()}
        except CAUGHT as error:
            result[name] = {"errors": ["無法核對：" + message(error)]}
    return result


def failing(result):
    found = ["tool"] if result["tool_errors"] else []
    found += ["shared"] if result["shared"]["errors"] else []
    for pid, roles in result["pages"].items():
        found += [f"{pid}/{role}" for role, value in roles.items() if value["errors"]]
    for pid, checks in result["coordinator_checks"].items():
        found += [f"coordinator {pid}/{name}" for name, value in checks.items() if value["errors"]]
    return found


def check_draft(root, draft, live, mode, stage):
    """草稿裡每一頁、每個角色各自收集錯誤；一頁、一個角色的失敗不影響其他頁。"""
    root = Path(root).resolve()
    ctx = setup(root, draft, mode, stage)
    frozen = {}
    try:
        frozen = {p["page_id"]: p for p in review.document(root, draft["source_freeze"])["pages"]}
    except CAUGHT as error:
        ctx.shared["errors"].append("source_freeze: " + message(error))
    pages, coordinator = {}, {}
    for row in draft.get("pages", []):
        pid = row.get("page_id")
        if pid in pages:
            ctx.shared["errors"].append(f"草稿 pages 重複列出 {pid}")
            continue
        missing = [name for name, found in (("目前清單", live), ("凍結檔", frozen)) if pid not in found]
        if missing:
            pages[pid] = {"page": {"errors": [f"{pid}: {'與'.join(missing)}沒有這一頁"], "pending_seal": []}}
            continue
        pages[pid] = check_page(root, ctx, pid, row, live[pid], frozen[pid])
        coordinator[pid] = coordinator_checks(root, ctx, pid, row, frozen[pid])
    result = {
        "tool": "preflight",
        "note": NOTE,
        "mode": mode,
        "stage": stage,
        "gate_module": gate_identity(),
        "tool_errors": drift(),
        "shared": ctx.shared,
        "pages": pages,
        "coordinator_checks": coordinator,
    }
    result["failing"] = failing(result)
    return result


def current_pages(root):
    """閘門的 live：composite_review.current_inventory 從目前的 course/ 算出各頁的版本。"""
    inventory = load("composite_review_for_preflight", ROOT / "docs/review-tools/composite_review.py")
    return {p["page_id"]: p for p in inventory.current_inventory(root)["pages"]}


def run(draft_path, mode, stage, root=ROOT, live=None):
    """讀草稿、算目前清單、印出結果 JSON，回傳 exit code。"""
    draft = json.loads(Path(draft_path).read_bytes())
    if live is None:
        try:
            live = current_pages(root)
        except CAUGHT as error:
            failure = {"tool": "preflight", "note": NOTE, "tool_errors": ["算不出目前清單：" + message(error)]}
            print(json.dumps(failure, ensure_ascii=False, indent=1))
            return 1
    result = check_draft(root, draft, live, mode, stage)
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
