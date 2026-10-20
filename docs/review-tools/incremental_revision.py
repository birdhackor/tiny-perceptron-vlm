"""沿用經重播的舊證據，核對新增／受影響頁的實際獨立審閱。

只驗證保存的來源、身分聲明、逐段紀錄與證據契約；不產生審閱答案。
"""

import hashlib
import importlib.util
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = "tutorial_incremental_v2"
ROLES = ("reader", "technical", "continuity")
SCOPE = (
    "Preserved baseline replay plus actual bounded independent reviews of changed sources and affected routes. "
    "Not a new full-course blind review, human test, proof of comprehension, scientific truth, physical isolation, "
    "image viewing, kernel execution, training success or deployment."
)
PATCHES_KIND = "author patches without AI review"
PATCHES_STATEMENT = "每一筆都是作者直接要求的小修改，沒有經過 AI 審閱；閘門只核對改動確實就是這裡寫的那一筆。"
PATCH_FIELDS = ("id", "path", "old", "new", "requested_by", "requested_on", "reason")
PATCH_LIMIT = 80
# old／new 不能出現的記號：換行會跨行，``` 與 ~~~ 是程式區塊的圍欄，![ 與 ]( 是圖與連結的語法。
PATCH_FORBIDDEN = {"\n": "換行", "\r": "換行", "```": "「```」", "~~~": "「~~~」", "![": "「![」", "](": "「](」"}
# 程式圍欄所在的行：行首（空白與引用記號 >）、三個以上的反引號或波浪號、同一行其餘的字。
FENCE = re.compile(r"([> \t]*)(`{3,}|~{3,})(.*)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def local(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "expected relative artifact path")
    path = (root / name).resolve()
    require(path.is_relative_to(root), "artifact escapes repository")
    return path


def artifact(root, reference):
    require(isinstance(reference, dict), "missing artifact reference")
    path = local(root, reference.get("path"))
    raw = path.read_bytes()
    require(sha(raw) == reference.get("sha256"), "artifact changed: " + reference["path"])
    return raw


def document(root, reference):
    return json.loads(artifact(root, reference))


def node(data, pointer):
    require(isinstance(pointer, str) and (not pointer or pointer.startswith("/")), "invalid JSON pointer")
    for token in pointer.split("/")[1:] if pointer else []:
        key = token.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def at(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp requires timezone")
    return result


def owner(report):
    return report.get("reviewer_task", report.get("reviewer"))


def reader_session(root, reference, expected_owner):
    opening = document(root, reference["opening"])
    require(opening.get("reviewer_task") == expected_owner, "reader trace owner mismatch")
    page_raw = artifact(root, reference["source"])
    require(sha(page_raw) == opening.get("source_sha256"), "reader trace source mismatch")
    figures = reference.get("figures", {})
    require(set(figures) == set(opening.get("figures_sha256", {})), "reader figure provenance incomplete")
    for name, digest in opening.get("figures_sha256", {}).items():
        require(sha(artifact(root, figures[name])) == digest, "reader original figure changed")
    tool = module("revision_reader_units", root / "docs/course-revision-20261016/tools/reader.py")
    chunks = tool.units(page_raw.decode())
    notes = reference.get("notes", [])
    require(len(notes) == len(chunks), "missing sequential checkpoint")
    last = at(opening["started_at"])
    for i, (chunk, ref) in enumerate(zip(chunks, notes)):
        note = document(root, ref)
        require(note.get("unit_index") == i and note.get("page_id") == opening["page_id"], "wrong checkpoint unit")
        require(note.get("reviewer_task") == expected_owner, "checkpoint owner mismatch")
        require(note.get("unit_sha256") == sha(chunk.encode()), "checkpoint visible text mismatch")
        saved = at(note["saved_at"])
        require(saved >= last, "checkpoint chronology reversed")
        last = saved
        for field in (
            "understanding",
            "materials_and_labels",
            "expected_change",
            "confusion_and_quote",
            "missing_visuals",
        ):
            require(isinstance(note.get(field), str) and note[field].strip(), "empty reader observation")
        checks = note.get("concept_checks")
        require(isinstance(checks, list) and checks, "missing concept-level four questions")
        for check in checks:
            for field in (
                "concept",
                "identity",
                "mechanism",
                "need",
                "usage",
                "example_scope",
                "source_quote",
                "verdict",
            ):
                require(isinstance(check.get(field), str) and check[field].strip(), "incomplete concept observation")
    return opening["page_id"], last


def technical_reports(manifest, root):
    """讀取實際逐節報告；讓舊validator核對來源、算例及獨立身分。"""
    reports = {}
    owners = {}
    for page in manifest["pages"]:
        proof = page.get("technical", {})
        references = (
            proof.get("section_reports", [])
            if proof.get("mode") == "section-composite"
            else ([proof["report"]] if proof.get("mode") == "review" else [])
        )
        for reference in references:
            report = document(root, reference)
            key = report["lesson_id"]
            require(key not in reports, "duplicate technical section")
            reports[key] = report
            owners.setdefault(owner(report), []).append(key)
    return reports, owners


def page_member(report, pointer, pid, task):
    require(owner(report) == task, "original/basis reviewer mismatch")
    value = node(report, pointer)
    require(value.get("page_id", value.get("lesson_id")) == pid, "original/basis page mismatch")
    require(value.get("source_sha256"), "original/basis lacks source identity")
    return value


def collection_seal(proof, role, sealed):
    collection_role = proof.get("collection_role", role)
    require(collection_role in ROLES and ROLES.index(collection_role) >= ROLES.index(role), "invalid followup phase")
    return sealed[proof["wave"]].get(collection_role)


def review_seals(root, waves, selected):
    """後補證據仍須核對實際收件階段，不回填已結束的早期簽封。"""
    sealed = {}
    preceding_wave_end = None
    for wave in waves:
        require(wave["id"] not in sealed, "duplicate review wave")
        sealed[wave["id"]] = {}
        previous_end = preceding_wave_end
        for role in ROLES:
            if role not in selected and role not in wave["seals"]:
                continue
            seal = document(root, wave["seals"][role])
            require(seal.get("role") == role and seal.get("status") == "completed", "stage is incomplete")
            require(at(seal["started_at"]) <= at(seal["completed_at"]), "stage duration reversed")
            if previous_end:
                require(at(seal["started_at"]) >= previous_end, "review stages overlapped")
            previous_end = at(seal["completed_at"])
            sealed[wave["id"]][role] = seal
        preceding_wave_end = previous_end
    return sealed


def require_collected(proof, role, sealed, references):
    phase = collection_seal(proof, role, sealed)
    require(phase is not None, "missing actual collection stage")
    require(
        all(ref["sha256"] in phase["report_sha256s"] for ref in references),
        "reports were not collected before stage closure",
    )


def reader_closure(root, proof, current, task, pid, reader_seal, sealed):
    original = document(root, proof["original_report"])
    original_page = page_member(original, proof["original_json_pointer"], pid, task)
    require(proof["original_report"]["sha256"] in reader_seal["report_sha256s"], "original report not sealed")
    require(current.get("basis_report"), "current reader judgment lacks actual basis")
    basis = document(root, proof["basis"]["report"])
    basis_page = page_member(basis, proof["basis"]["json_pointer"], pid, task)
    require(basis_page["source_sha256"] == current["source_sha256"], "current judgment lacks read revised source")
    basis_declared = current["basis_report"]
    declared_path = basis_declared.get("path") if isinstance(basis_declared, dict) else basis_declared
    require(declared_path == proof["basis"]["report"]["path"], "current basis link mismatch")
    if isinstance(basis_declared, dict) and "sha256" in basis_declared:
        require(basis_declared["sha256"] == proof["basis"]["report"]["sha256"], "declared basis digest mismatch")
    verdict = basis_page.get("verdict", "")
    require(
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
    require(
        basis_page.get("figures_sha256", basis_page.get("figure_sha256")) == current["figures_sha256"],
        "basis figure version mismatch",
    )
    estimate = current.get("ai_estimate", {})
    lo = estimate.get("minutes_min", estimate.get("min_minutes"))
    hi = estimate.get("minutes_max", estimate.get("max_minutes"))
    require(type(lo) is int and type(hi) is int and 0 < lo <= hi and estimate.get("reason"), "missing reader estimate")
    observed = {}
    matched_original = False
    latest_observation = at(reader_seal["started_at"])
    for reference in proof["sessions"]:
        opening = document(root, reference["opening"])
        if opening["source_sha256"] == original_page["source_sha256"]:
            matched_original = True
        _, completed = reader_session(root, reference, task)
        latest_observation = max(latest_observation, completed)
        phase = sealed[reference.get("wave", proof["wave"])][reference.get("collection_role", "reader")]
        require(
            at(phase["started_at"]) <= at(opening["started_at"]) <= completed <= at(phase["completed_at"]),
            "reader checkpoint outside actual phase",
        )
        for ref in reference["notes"]:
            for i, concept in enumerate(document(root, ref)["concept_checks"]):
                if concept["verdict"] != "已足夠":
                    observed[(ref["sha256"], i)] = concept["verdict"]
    require(matched_original, "original judgment has no matching first-read source")
    if not observed and "checkpoint_closure" not in proof:
        return
    closure_ref = proof["checkpoint_closure"]
    closure_report = document(root, closure_ref["report"])
    closure = page_member(closure_report, closure_ref["json_pointer"], pid, task)
    require(
        closure.get("verdict") == "pass" and closure["source_sha256"] == current["source_sha256"],
        "reader closure stale/open",
    )
    require(closure.get("figures_sha256") == current["figures_sha256"], "reader closure figures stale")
    require(
        closure_report.get("basis_report") == proof["report"], "checkpoint closure not linked to own current judgment"
    )
    created = at(closure_report["created_at"])
    require(created >= latest_observation, "closure predates its observations")
    if closure_ref["collection_role"] != "reader":
        require(created >= at(reader_seal["completed_at"]), "later closure predates sealed current judgment")
    for data in (basis, document(root, proof["report"])):
        for key in ("created_at", "confirmed_at"):
            if isinstance(data.get(key), str) and "T" in data[key]:
                require(created >= at(data[key]), "closure predates its basis judgment")
    received = at(closure_ref["received_at"])
    require(
        at(reader_seal["started_at"]) <= created <= received <= datetime.now(UTC), "reader closure chronology invalid"
    )
    closing_seal = sealed[closure_ref["wave"]].get(closure_ref["collection_role"])
    if closing_seal:
        require(
            at(closing_seal["started_at"]) <= received <= at(closing_seal["completed_at"]),
            "closure outside followup phase",
        )
        require(
            closure_ref["report"]["sha256"] in closing_seal["report_sha256s"], "closure not collected before phase end"
        )
    resolved = {}
    for item in closure.get("checkpoint_resolutions", []):
        key = (item["note"]["sha256"], item["concept_index"])
        require(key not in resolved, "duplicate checkpoint closure")
        artifact(root, item["note"])
        require(item.get("final_status") in {"resolved", "nonblocking", "deferred"}, "reader necessary gap still open")
        require(item.get("reason") and item.get("evidence"), "reader closure lacks actual reason/evidence")
        resolved[key] = item["original_verdict"]
    require(observed == resolved, "original non-pass concept lacks reader final closure")


def reported_issues(value, pointer=""):
    """列出報告真正保存的問題位置，不靠作者的完整性布林值。"""
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key in {"issues", "new_issues"} and isinstance(item, list):
                result.extend(location + "/" + str(i) for i, issue in enumerate(item) if isinstance(issue, dict))
            else:
                result.extend(reported_issues(item, location))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            result.extend(reported_issues(item, pointer + "/" + str(i)))
    return result


def check_dispositions(root, manifest, selected, live, tasks, sealed, earlier=None):
    """核對每個問題的處置；earlier 是各頁倒退作者補丁得到的較早版本（page_versions），回傳修正的複查靠了哪些補丁。"""
    earlier = earlier or {}
    relied = {}
    dispositions = document(root, manifest["dispositions"])
    require(dispositions.get("pending_count") == 0, "unresolved author dispositions")
    collected = manifest.get("dispositions_collection", {})
    received = at(collected["received_at"])
    require(at(dispositions["at"]) <= received <= datetime.now(UTC), "dispositions not actually collected yet")
    phase = sealed[collected["wave"]].get(collected["role"])
    if phase:
        require(at(phase["started_at"]) <= received <= at(phase["completed_at"]), "dispositions outside closure phase")
    required = {}
    note_reports = {}
    for page in manifest["pages"]:
        for role in selected:
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
    if "technical" in selected and "tool_audit" in manifest:
        ref = manifest["tool_audit"]["report"]
        required[ref["sha256"]] = ref
    if "continuity" in selected and "dependency_audit" in manifest:
        ref = manifest["dependency_audit"]
        required[ref["sha256"]] = ref
    listed = {ref["sha256"]: ref for ref in manifest.get("issue_reports", [])}
    require(set(required) <= set(listed), "issue inventory omits a current or original report")
    expected = set()
    for digest, ref in (listed | note_reports).items():
        report = document(root, ref)
        for pointer in reported_issues(report):
            expected.add((digest, pointer))
    for extra in manifest.get("additional_issue_inventories", []):
        report = document(root, extra["report"])
        items = node(report, extra["json_pointer"])
        require(isinstance(items, list), "additional issues must be a list")
        for i in range(len(items)):
            expected.add((extra["report"]["sha256"], extra["json_pointer"] + "/" + str(i)))
    actual = set()
    for decision in dispositions.get("items", []):
        key = (decision["report"]["sha256"], decision["json_pointer"])
        require(key not in actual, "duplicate issue disposition")
        actual.add(key)
        node(document(root, decision["report"]), decision["json_pointer"])
        for field in ("source_support", "learning_benefit", "change_burden", "reason"):
            require(isinstance(decision.get(field), str) and decision[field].strip(), "incomplete author adjudication")
        require(decision.get("final_necessity") in {"necessary", "optional"}, "issue necessity not adjudicated")
        require(decision.get("action") in {"fixed", "keep", "defer", "reject"}, "issue still pending")
        require(decision["final_necessity"] != "necessary" or decision["action"] == "fixed", "necessary gap unresolved")
        if decision["action"] == "fixed":
            require(decision.get("verification"), "fix requires actual verification references")
        checked_pages = set()
        checked_files = set()
        issue_report = document(root, decision["report"])
        issue_owner = owner(issue_report)
        for verification in decision.get("verification", []):
            report = document(root, verification["report"])
            task = owner(report)
            require(
                task in tasks
                and (task == issue_owner or issue_owner in manifest["author_tasks"] or issue_owner is None),
                "fix was not rechecked by original reviewer",
            )
            if verification.get("kind") == "tool_audit":
                require(
                    report.get("verdict") == "pass" and decision["issue_id"] in report.get("resolved_issue_ids", []),
                    "tool fix lacks actual independent closure",
                )
                for checked in report.get("checked_files", []):
                    if checked.get("sha256") == sha(local(root, checked["path"]).read_bytes()):
                        checked_files.add(checked["path"])
                continue
            value = node(report, verification["json_pointer"])
            pid = value.get("page_id", value.get("lesson_id"))
            require(pid in live and value.get("verdict") == "pass", "fix lacks reviewer pass on affected page")
            used = patched_version(value.get("source_sha256"), live[pid]["source_sha256"], earlier.get(pid, {}))
            require(used is not None, "fix recheck is stale")
            relied.setdefault(pid, set()).update(used)
            require(
                value.get("figures_sha256", value.get("figure_sha256")) == live[pid]["figures_sha256"],
                "fix figure recheck stale",
            )
            checked_pages.add(pid)
        if decision["action"] == "fixed":
            affected = set(decision.get("affected_pages", []))
            files = set(decision.get("affected_files", []))
            require(
                (affected or files) and affected <= checked_pages and files <= checked_files,
                "fix does not cover affected pages/files",
            )
    require(actual == expected, "reported issues and dispositions differ")
    return relied


def baseline_supplements(root, manifest):
    """讀候選manifest指定的補充還原清單，驗外層格式後交給重放；欄位缺席時不補。"""
    # 驗證：清單讀得到、位元組等於manifest記的SHA、是JSON物件，schema_version為1，files是list，每筆恰好三個字串欄位。
    # 不涵蓋：每筆的路徑、來源、重複與舊身分，由baseline_replay在建立暫存目錄前核對。
    if "baseline_supplements" not in manifest:
        return []
    reference = manifest["baseline_supplements"]
    try:
        raw = artifact(root, reference)
    except OSError as error:
        raise ValueError(f"補充還原清單讀不到（{type(error).__name__}）：{reference['path']}") from None
    name = reference["path"]
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError("補充還原清單不是合法的 JSON：" + name) from None
    require(isinstance(data, dict), "補充還原清單不是 JSON 物件：" + name)
    version = data.get("schema_version")
    require(type(version) is int and version == 1, "補充還原清單的 schema_version 不是 1：" + name)
    files = data.get("files")
    require(isinstance(files, list), "補充還原清單的 files 不是 list：" + name)
    for i, entry in enumerate(files):
        require(
            isinstance(entry, dict)
            and set(entry) == {"live_path", "path", "sha256"}
            and all(isinstance(value, str) for value in entry.values()),
            f"補充還原清單 {name} 的 files[{i}] 必須是只含 live_path、path、sha256 三個字串欄位的物件",
        )
    return files


def is_calendar_day(value):
    """value 是不是 YYYY-MM-DD 寫法、而且真的存在的日期。"""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def author_patches(root, manifest, live):
    """讀 manifest 的 author_patches 指向的作者補丁清單，逐筆核對格式；欄位缺席時回傳空清單。

    live 是目前清單的各頁（page_id 對 page）；manifest 涵蓋的頁和它相同，每筆的 path 必須是其中一頁的 Markdown source。
    """
    # 驗證：清單的位元組等於 manifest 記的 SHA，是 JSON 物件，schema_version 為 1，kind 與 statement 是規定的文字；
    # 每筆恰好 PATCH_FIELDS 七個欄位，id 不重複，path 是某一頁以 .md 結尾的 source（首頁的 source 是產生它的
    # Python 程式，整份都是程式碼），old 與 new 是彼此不同、各不超過 80 字、
    # 不含 PATCH_FORBIDDEN 記號的非空字串，requested_by 是 author，requested_on 是 YYYY-MM-DD，reason 非空。
    # 不涵蓋：補丁的語意；new 或 old 落在行內程式碼、數學式、連結的網址或文字、沒有圍欄的程式區塊或 HTML 裡時，
    # 上面的字面限制與 fenced 都擋不住。規約把這些交給要求補丁的作者負責。
    # 現行檔是否正好是審閱過的版本加上這些補丁，由 file_versions 與 page_versions 倒退核對。
    if "author_patches" not in manifest:
        return []
    sources = {page.get("source") for page in live.values()}
    reference = manifest["author_patches"]
    try:
        raw = artifact(root, reference)
    except OSError as error:
        raise ValueError(f"作者補丁清單讀不到（{type(error).__name__}）：{reference['path']}") from None
    name = reference["path"]
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError("作者補丁清單不是合法的 JSON：" + name) from None
    require(isinstance(data, dict), "作者補丁清單不是 JSON 物件：" + name)
    version = data.get("schema_version")
    require(type(version) is int and version == 1, "作者補丁清單的 schema_version 不是 1：" + name)
    require(data.get("kind") == PATCHES_KIND, f"作者補丁清單的 kind 不是 {PATCHES_KIND!r}：" + name)
    require(data.get("statement") == PATCHES_STATEMENT, "作者補丁清單的 statement 不是規定的聲明：" + name)
    patches = data.get("patches")
    require(isinstance(patches, list), "作者補丁清單的 patches 不是 list：" + name)
    seen = set()
    for i, patch in enumerate(patches):
        where = f"作者補丁清單 {name} 的 patches[{i}]"
        require(
            isinstance(patch, dict) and set(patch) == set(PATCH_FIELDS),
            where + " 必須是恰好含 " + "、".join(PATCH_FIELDS) + " 七個欄位的物件",
        )
        where += f"（id {patch['id']!r}）"
        require(isinstance(patch["id"], str) and patch["id"].strip(), where + "：id 必須是非空字串")
        require(patch["id"] not in seen, where + "：id 和前面的補丁重複")
        seen.add(patch["id"])
        require(
            isinstance(patch["path"], str) and patch["path"] in sources,
            where + "：path 不是 manifest 任何一頁的 source",
        )
        require(
            patch["path"].endswith(".md"),
            where + "：path 不是 Markdown（.md）原文；程式檔整份是程式碼，不能用作者補丁改",
        )
        for field in ("old", "new"):
            text = patch[field]
            require(isinstance(text, str) and text, where + f"：{field} 必須是非空字串")
            require(len(text) <= PATCH_LIMIT, where + f"：{field} 有 {len(text)} 個字元，超過 {PATCH_LIMIT}")
            for token, label in PATCH_FORBIDDEN.items():
                require(token not in text, where + f"：{field} 不能含{label}")
        require(patch["old"] != patch["new"], where + "：old 與 new 相同")
        require(patch["requested_by"] == "author", where + "：requested_by 必須是 'author'")
        require(is_calendar_day(patch["requested_on"]), where + "：requested_on 必須是 YYYY-MM-DD 格式的日期")
        require(isinstance(patch["reason"], str) and patch["reason"].strip(), where + "：reason 必須是非空字串")
    return patches


def positions(text, needle):
    """needle 在 text 裡每一處的起點，彼此重疊的也算。"""
    found, start = [], text.find(needle)
    while start != -1:
        found.append(start)
        start = text.find(needle, start + 1)
    return found


def closes_superfences(opener, line):
    """網站用的 pymdownx.superfences：行首與開頭那行完全相同，同一串記號，後面只接空白。"""
    return line[1] == opener[1] and line[2] == opener[2] and not line[3].strip()


def quote_and_indent(prefix):
    """行首的引用層數，以及最後一個 > 之後的縮排格數（> 後面那一格空白屬於引用記號）。"""
    depth = prefix.count(">")
    rest = prefix.rsplit(">", 1)[-1]
    if depth and rest.startswith(" "):
        rest = rest[1:]
    return depth, len(rest.expandtabs(4))


def closes_commonmark(opener, line):
    """Notebook 顯示 Markdown 用的 CommonMark：引用層數相同、縮排不超過三格，同種記號、不短於開頭，後面只接空白。"""
    depth, indent = quote_and_indent(line[1])
    return (
        depth == quote_and_indent(opener[1])[0]
        and indent <= 3
        and line[2][0] == opener[2][0]
        and len(line[2]) >= len(opener[2])
        and not line[3].strip()
    )


def inside_fence(text, position, closes):
    """照 closes 這條收尾規則，position 是否在圍欄區塊裡；圍欄那兩行也算，沒有收尾的一路算到文字結尾。"""
    opener, start = None, 0
    for line in text.split("\n"):
        match = FENCE.match(line)
        if position <= start + len(line):
            return opener is not None or match is not None
        if opener is None:
            opener = match
        elif match and closes(opener, match):
            opener = None
        start += len(line) + 1
    return opener is not None


def fenced(text, position):
    """position 是否落在程式圍欄圍起來的區塊裡：網站與 Notebook 的認法任一個認為在區塊裡就算。"""
    # 驗證：區塊外符合 FENCE 的行開一個區塊；區塊裡照 closes_superfences（網站）或 closes_commonmark（Notebook）
    # 收尾，其他符合 FENCE 的行是區塊內容。兩種認法各跑一次，任一個在區塊裡就算。superfences 因選項寫錯或縮排
    # 不夠而放棄的圍欄，這裡仍當成程式區塊。
    # 不涵蓋：縮排四格的程式區塊與 HTML 的 <pre>；行中間出現 ```python 再換行時，scripts/build_course.py 會從
    # 那裡切出 Notebook 的程式格，這裡只認行首的圍欄。
    return any(inside_fence(text, position, closes) for closes in (closes_superfences, closes_commonmark))


def undo_patches(text, patches, where, absent_ok=False):
    """從 text 依清單倒序逐筆倒退補丁，回傳每倒退一筆得到的（較早版本的文字，到這一版為止倒退的補丁 id）。

    patches 是同一個檔的（清單位置，補丁），照清單順序。每一筆的 new 在當下的文字裡要恰好出現一次、不在程式圍欄
    圍起來的區塊裡，才換回 old。absent_ok 時 new 沒出現的那一筆略過：那一筆改的是同一個檔的別頁。
    """
    versions, undone = [], ()
    for i, patch in reversed(patches):
        label = f"作者補丁清單的 patches[{i}]（id {patch['id']!r}）"
        hits = positions(text, patch["new"])
        if absent_ok and not hits:
            continue
        require(len(hits) == 1, f"{label}：new 在倒退到這一筆時的 {where} 出現 {len(hits)} 次，必須恰好一次")
        require(not fenced(text, hits[0]), f"{label}：new 在 {where} 的那一處落在程式圍欄圍起來的區塊裡")
        text = text[: hits[0]] + patch["old"] + text[hits[0] + len(patch["new"]) :]
        undone += (patch["id"],)
        versions.append((text, undone))
    return versions


def earlier_digests(versions, current):
    """倒退得到的版本換成 UTF-8 位元組的 SHA-256；等於 current 的不算較早版本，同一個 SHA 記第一次得到時倒退的補丁。"""
    result = {}
    for text, undone in versions:
        digest = sha(text.encode("utf-8"))
        if digest != current:
            result.setdefault(digest, undone)
    return result


def file_versions(root, patches):
    """有補丁的每個檔，從現行位元組倒退它的全部補丁：{path: {較早版本的 SHA-256: 倒退的補丁 id}}。"""
    result = {}
    for path in dict.fromkeys(patch["path"] for patch in patches):
        raw = local(root, path).read_bytes()
        own = [(i, patch) for i, patch in enumerate(patches) if patch["path"] == path]
        result[path] = earlier_digests(undo_patches(raw.decode("utf-8"), own, path), sha(raw))
    return result


def page_versions(root, frozen_page, page, patches):
    """這一頁凍結快照的現行文字，倒退 new 出現在這一頁的補丁：{較早版本的 SHA-256: 倒退的補丁 id}。"""
    own = [(i, patch) for i, patch in enumerate(patches) if patch["path"] == page["source"]]
    if not own:
        return {}
    raw = artifact(root, {"path": frozen_page["snapshot"], "sha256": page["source_sha256"]})
    where = "頁 " + page["page_id"] + " 的凍結快照"
    return earlier_digests(undo_patches(raw.decode("utf-8"), own, where, absent_ok=True), page["source_sha256"])


def patched_version(recorded, current, versions):
    """recorded 是現行版本時回傳 ()；是倒退作者補丁得到的較早版本時回傳倒退的補丁 id；都不是時回傳 None。"""
    if recorded == current:
        return ()
    if isinstance(recorded, str) and recorded in versions:
        return versions[recorded]
    return None


def technical_patches(report, versions, files):
    """技術報告靠作者補丁才對上的補丁 id：報告的 source_sha256，以及 repository_code 來源記的檔案 SHA-256。"""
    used = []
    if isinstance(report.get("source_sha256"), str):
        used += versions.get(report["source_sha256"], ())
    sources = report.get("sources")
    for source in sources if isinstance(sources, list) else []:
        if isinstance(source, dict) and source.get("kind") == "repository_code":
            path, digest = source.get("path"), source.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                used += files.get(path, {}).get(digest, ())
    return used


def check_tool_audit(root, manifest, task_roles, sealed):
    """工具審閱要對著現行工具位元組通過，並在宣告的階段收件。"""
    audit_ref = manifest["tool_audit"]
    audit = document(root, audit_ref["report"])
    require(
        audit.get("verdict") == "pass" and task_roles.get(owner(audit)) == "technical",
        "incremental tools lack independent pass",
    )
    checked = {item["path"]: item["sha256"] for item in audit.get("checked_files", [])}
    for name in (
        "docs/review-tools/incremental_revision.py",
        "docs/review-tools/baseline_replay.py",
        "docs/review-tools/phase7_review.py",
        "docs/course-revision-20261016/tools/reader.py",
        "docs/course-revision-20261016/tools/freeze_sources.py",
        "tests/test_incremental_revision_review.py",
    ):
        require(checked.get(name) == sha(local(root, name).read_bytes()), "tool audit version stale: " + name)
    if "baseline_supplements" in manifest:
        # 驗證：checked_files列了候選manifest指定的補充還原清單，SHA等於它現在的位元組；清單一改，舊報告就過期。
        # 不涵蓋：清單指到的舊副本；它們由清單裡的sha256與重放時的舊身分核對綁住。
        name = manifest["baseline_supplements"]["path"]
        require(name in checked, "工具審閱的 checked_files 沒有列補充還原清單：" + name)
        require(
            checked[name] == sha(local(root, name).read_bytes()),
            "工具審閱核對的補充還原清單不是現行版本：" + name,
        )
    require_collected(audit_ref, "technical", sealed, [audit_ref["report"]])


def audit_manifest(root, path, stage="all"):
    root = Path(root).resolve()
    failures, records = [], []
    patches, relied = [], {}
    try:
        require(stage in {"reader", "technical", "all"}, "unknown review stage")
        manifest = json.loads(Path(path).read_bytes())
        require(manifest.get("review_policy") == POLICY and manifest.get("schema_version") == 2, "wrong policy")
        supplements = baseline_supplements(root, manifest)
        baseline = module("revision_baseline_replay", root / "docs/review-tools/baseline_replay.py").replay(
            root, supplements
        )
        require(not baseline["failures"], "preserved baseline failed: " + repr(baseline["failures"]))
        original = document(root, manifest["baseline_freeze"])
        original_policy = json.loads((root / "docs/course-publish-20261008/review-manifest.json").read_bytes())
        require(
            artifact(root, manifest["baseline_freeze"]) == artifact(root, original_policy["source_freeze"]),
            "baseline identity is not the replayed source freeze",
        )
        freeze = document(root, manifest["source_freeze"])
        inventory = module("revision_inventory", root / "docs/review-tools/composite_review.py").current_inventory(root)
        old = {p["page_id"]: p for p in original["pages"]}
        live = {p["page_id"]: p for p in inventory["pages"]}
        frozen = {p["page_id"]: p for p in freeze["pages"]}
        catalog = {p["page_id"]: p for p in manifest["pages"]}
        require(
            len(live)
            == len(inventory["pages"])
            == len(frozen)
            == len(freeze["pages"])
            == len(catalog)
            == len(manifest["pages"]),
            "duplicate or missing canonical page",
        )
        require(
            set(live) == set(frozen) == set(catalog) and set(old) <= set(live),
            "canonical scope lost or added without evidence",
        )
        # 作者補丁：核對清單、倒退每個補丁檔；各頁的較早版本在頁面迴圈裡由 page_versions 算，存進 earlier。
        patches = author_patches(root, manifest, live)
        files = file_versions(root, patches)
        earlier = {}
        original_inputs = document(root, original_policy["current_technical_inputs"])["files_sha256"]
        required_implementation = set(original_inputs) | {
            "docs/review-tools/incremental_revision.py",
            "docs/review-tools/baseline_replay.py",
            "docs/review-tools/phase7_review.py",
            "docs/review-tools/composite_review.py",
            "docs/review-tools/incremental_reader.py",
            "scripts/check_technical_reviews.py",
            "scripts/build_course.py",
            "scripts/reading_time.py",
            "scripts/export_course.py",
            "docs/course-revision-20261016/tools/reader.py",
            "docs/course-revision-20261016/tools/freeze_sources.py",
            "tests/test_incremental_revision_review.py",
        }
        require(
            required_implementation <= set(manifest.get("implementation", {})), "implementation inventory incomplete"
        )
        builder = module("revision_notebook_builder", root / "scripts/build_course.py")
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            builder.build(check=True)
        figure_set = {name: digest for page in live.values() for name, digest in page["figures_sha256"].items()}
        require(set(freeze.get("figure_snapshots", {})) == set(figure_set), "frozen figure inventory incomplete")
        for name, digest in figure_set.items():
            require(sha(artifact(root, freeze["figure_snapshots"][name])) == digest, "frozen figure source mismatch")
        for kind in ("criteria", "implementation"):
            entries = manifest.get(kind, {})
            require(entries, "missing " + kind + " freeze")
            if kind == "criteria":
                required = {".agents/skills/clear-tutorial/" + name for name in original["criteria"]}
                require(set(entries) == required, "incomplete tutorial criteria")
            for name, digest in entries.items():
                require(sha(local(root, name).read_bytes()) == digest, kind + " drift: " + name)
        authors = set(manifest.get("author_tasks", []))
        require("/root" in authors, "missing actual author exclusions")
        assignments = document(root, manifest["assignments"])
        task_roles = {}
        task_assignments = {}
        for assignment in assignments["tasks"]:
            task, role = assignment["task"], assignment["role"]
            require(task.startswith("/root/") and task not in authors, "invalid independent task")
            require(role in ROLES and task not in task_roles, "task reused across assignments")
            require(assignment.get("context") == "fresh", "review task context is not fresh")
            task_roles[task] = role
            task_assignments[task] = assignment
        reports, technical_owners = technical_reports(manifest, root)
        technical = module("revision_factual_validator", root / "scripts/check_technical_reviews.py")
        reader_tasks = {task for task, role in task_roles.items() if role == "reader"}
        selected = ROLES[:1] if stage == "reader" else ROLES[:2] if stage == "technical" else ROLES
        waves = manifest.get("review_waves", [])
        require(waves, "missing actual review chronology")
        sealed = review_seals(root, waves, selected)
        for assignment in assignments["tasks"]:
            if assignment["role"] not in selected:
                continue
            seal = sealed[assignment["wave"]][assignment["role"]]
            window = assignment["dispatch_window"]
            require(
                at(seal["started_at"])
                <= at(window["not_before"])
                <= at(window["not_after"])
                <= at(seal["completed_at"]),
                "task dispatch window outside stage interval",
            )
        for pid, page in live.items():
            for field in ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256"):
                require(
                    page.get(field) == frozen[pid].get(field), "current source/figure/notebook freeze stale: " + pid
                )
            require(
                sha(artifact(root, {"path": frozen[pid]["snapshot"], "sha256": page["source_sha256"]}))
                == page["source_sha256"],
                "missing source snapshot",
            )
            versions = earlier[pid] = page_versions(root, frozen[pid], page, patches)
            inherited_identity = pid in old and all(
                page.get(f) == old[pid].get(f) for f in ("source", "selector", "source_sha256", "figures_sha256")
            )
            row = catalog[pid]
            for role in selected:
                proof = row.get(role, {})
                if proof.get("mode") == "inherit":
                    require(inherited_identity, "changed source cannot inherit old verdict: " + pid)
                    if page.get("notebook"):
                        old_notebook = old[pid].get("notebook", {})
                        require(
                            old_notebook.get("live_sha256") == page["notebook_sha256"],
                            "changed Notebook cannot inherit",
                        )
                    require(proof.get("baseline_page_id") == pid, "inherited page mismatch")
                    require(
                        isinstance(proof.get("dependency_note"), str) and proof["dependency_note"].strip(),
                        "inheritance needs scope/route disposition",
                    )
                    continue
                if role == "technical" and proof.get("mode") == "section-composite":
                    require(
                        pid in old and page["selector"] == "whole-file", "section composite needs original whole page"
                    )
                    from scripts import reading_time

                    old_text = local(root, old[pid]["snapshot"]).read_text()
                    live_text = local(root, frozen[pid]["snapshot"]).read_text()
                    old_sections = reading_time.lesson_slices(old_text)
                    live_sections = reading_time.lesson_slices(live_text)
                    require(set(old_sections) <= set(live_sections), "original numbered section removed")
                    require(
                        reading_time.chapter_introduction(old_text) == reading_time.chapter_introduction(live_text),
                        "changed page introduction needs separate factual review",
                    )
                    changed = {key for key, body in live_sections.items() if body != old_sections.get(key)}
                    supplied = {document(root, ref)["lesson_id"] for ref in proof["section_reports"]}
                    require(changed == supplied, "section factual reviews do not cover exact changed sections")
                    for key in changed:
                        report = reports[key]
                        require(task_roles.get(owner(report)) == role, "wrong section reviewer role")
                        assignment = task_assignments[owner(report)]
                        require(
                            assignment["wave"] == proof["wave"] and assignment.get("lesson_id") == key,
                            "section outside actual assignment",
                        )
                        errors = technical._validate(
                            root,
                            local(root, page["source"]),
                            key,
                            live_sections[key],
                            report,
                            technical_owners,
                            reader_tasks,
                        )
                        require(not errors, key + ": " + repr(errors))
                    require_collected(proof, role, sealed, proof["section_reports"])
                    continue
                require(proof.get("mode") == "review", "missing current role evidence: " + pid + "/" + role)
                report = document(root, proof["report"])
                phase = collection_seal(proof, role, sealed)
                if phase:
                    require(
                        proof["report"]["sha256"] in phase["report_sha256s"],
                        "report was not collected before stage closure",
                    )
                task = owner(report)
                require(task_roles.get(task) == role, "undeclared or wrong-role reviewer")
                assignment = task_assignments[task]
                require(assignment["wave"] == proof["wave"], "review wave outside actual assignment")
                require(
                    pid in assignment.get("pages", []) if role != "technical" else assignment.get("lesson_id") == pid,
                    "page outside actual assignment",
                )
                current = node(report, proof.get("json_pointer", ""))
                require(current.get("page_id", current.get("lesson_id")) == pid, "report page membership mismatch")
                # 報告記的可以是凍結快照倒退作者補丁得到的較早版本；靠了哪些補丁記進 relied，最後標在 records。
                # 不涵蓋：inherit 頁、section-composite 的整份檔頁、圖與 Notebook 輸出都只認現行版本，不走這條路；
                # 補丁的語意由作者負責，閘門只核對字面。
                used = patched_version(current.get("source_sha256"), page["source_sha256"], versions)
                require(used is not None, "report source is stale: " + pid)
                relied.setdefault(pid, set()).update(used)
                require(current.get("verdict") in {"pass", "已足夠"}, "reviewer requests revision: " + pid)
                figures = current.get("figures_sha256", current.get("figure_sha256"))
                require(figures == page["figures_sha256"], "review figure identity incomplete: " + pid)
                require(proof.get("original_report"), "original judgment must be retained")
                original_report = document(root, proof["original_report"])
                page_member(original_report, proof.get("original_json_pointer", ""), pid, task)
                if role == "reader":
                    sessions = proof.get("sessions", [])
                    require(sessions, "no actual sequential first-read trace")
                    for reference in sessions:
                        require(
                            document(root, reference["opening"])["page_id"] == pid,
                            "reader session belongs to another page",
                        )
                    reader_closure(root, proof, current, task, pid, sealed[proof["wave"]]["reader"], sealed)
                if role == "technical":
                    body = local(root, frozen[pid]["snapshot"]).read_text()
                    errors = technical._validate(
                        root,
                        local(root, page["source"]),
                        pid,
                        body,
                        report,
                        technical_owners,
                        reader_tasks,
                        earlier_versions={"source_sha256": versions, "files": files},
                    )
                    require(not errors, pid + ": " + repr(errors))
                    relied[pid].update(technical_patches(report, versions, files))
            records.append({"page_id": pid, "source_sha256": page["source_sha256"]})
        if stage in {"technical", "all"}:
            check_tool_audit(root, manifest, task_roles, sealed)
        for pid, used in check_dispositions(root, manifest, selected, live, task_roles, sealed, earlier).items():
            relied.setdefault(pid, set()).update(used)
        if stage == "all":
            dependency = document(root, manifest["dependency_audit"])
            require(
                dependency.get("verdict") == "pass" and task_roles.get(owner(dependency)) == "continuity",
                "missing independent dependency audit",
            )
            require(set(dependency.get("inventory_page_ids", [])) == set(live), "dependency audit inventory incomplete")
    except (ValueError, KeyError, TypeError, OSError, StopIteration, RuntimeError, IndexError) as error:
        failures.append(str(error))
    # 靠作者補丁才對上的頁，列出靠的補丁 id（照清單順序）。
    for record in records:
        used = relied.get(record["page_id"])
        if used:
            record["author_patches"] = [patch["id"] for patch in patches if patch["id"] in used]
    return {
        "schema_version": 2,
        "review_policy": POLICY,
        "stage": stage,
        "status": "failed" if failures else "passed_metadata_checks",
        "records": records,
        "failures": failures,
        "scope_note": SCOPE,
    }


def check_active(root, active, stage="all"):
    try:
        require(active.get("review_policy") == POLICY and active.get("schema_version") == 2, "invalid active policy")
        artifact(Path(root).resolve(), {"path": active.get("manifest"), "sha256": active.get("manifest_sha256")})
        return audit_manifest(root, Path(root) / active["manifest"], stage)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"status": "failed", "failures": [str(error)], "scope_note": SCOPE}
