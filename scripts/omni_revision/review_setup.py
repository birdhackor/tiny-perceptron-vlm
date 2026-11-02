"""準備本輪審閱來源與待補證據；不產生判定，也不修改 active pointer。"""

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "docs/course-reviews-active.json"
POLICY = "tutorial_incremental_v2"
ROLES = ("reader", "technical", "continuity")
ACCEPTED_BASELINE = {
    "path": "docs/course-review-20261027-tokenizer/assembly-all-01/manifest.json",
    "sha256": "22280b0c680f4011f96f9d1281878de559ebaecae5561953f023f823724d5e43",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("輸出及來源必須在repo內")
    return path


def reference(path):
    path = local(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def read_ref(ref):
    path = local(ref["path"])
    raw = path.read_bytes()
    if sha(raw) != ref["sha256"]:
        raise ValueError("來源指紋不符：" + ref["path"])
    return json.loads(raw)


def write_new(path, value):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, local(relative))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def baseline():
    pointer = json.loads(ACTIVE.read_bytes())
    manifest = read_ref({"path": pointer["manifest"], "sha256": pointer["manifest_sha256"]})
    if manifest.get("review_policy") != POLICY:
        raise ValueError("本工具只準備現行incremental_v2，不能轉換舊判準")
    return manifest, read_ref(manifest["source_freeze"])


def compare(prior, live):
    old = {page["page_id"]: page for page in prior["pages"]}
    current = {page["page_id"]: page for page in live["pages"]}
    changed = {}
    for pid, page in current.items():
        before = old.get(pid)
        reasons = []
        if before is None:
            reasons.append("new_page")
        else:
            for field in ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256"):
                if before.get(field) != page.get(field):
                    reasons.append(field)
        if reasons:
            changed[pid] = reasons
    return {
        "prior_page_count": len(old),
        "current_page_count": len(current),
        "removed_page_ids": sorted(set(old) - set(current)),
        "changed": changed,
        "fully_unchanged_page_ids": [pid for pid in current if pid not in changed],
    }


def digest_references(value):
    """只找候選依賴；舊報告中的歷史快照不能由這個掃描判定需重審。"""
    if isinstance(value, dict):
        name, digest = value.get("path"), value.get("sha256")
        if isinstance(name, str) and isinstance(digest, str) and len(digest) == 64:
            yield name, digest
        for child in value.values():
            yield from digest_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from digest_references(child)


def technical_dependency_candidates(manifest):
    candidates = []
    for page in manifest["pages"]:
        proof = page.get("technical", {})
        reports = proof.get("section_reports", []) if proof.get("mode") == "section-composite" else []
        if proof.get("mode") == "review":
            reports = [proof["report"]]
        for ref in reports:
            report = read_ref(ref)
            for name, recorded in digest_references(report):
                try:
                    path = local(name)
                except ValueError:
                    continue
                if not path.is_file():
                    status = "unavailable_reference"
                    actual = None
                else:
                    actual = sha(path.read_bytes())
                    if actual == recorded:
                        continue
                    status = "digest_changed"
                candidates.append(
                    {
                        "page_id": page["page_id"],
                        "lesson_id": report.get("lesson_id"),
                        "report": ref,
                        "dependency": name,
                        "recorded_sha256": recorded,
                        "current_sha256": actual,
                        "status": status,
                    }
                )
    return candidates


def inventory():
    manifest, prior = baseline()
    live = module("omni_review_inventory", "docs/review-tools/composite_review.py").current_inventory(ROOT)
    result = compare(prior, live)
    result.update(
        schema_version=1,
        recorded_at=datetime.now(UTC).isoformat(),
        active_pointer=reference(ACTIVE),
        active_source_freeze=manifest["source_freeze"],
        technical_dependency_candidates=technical_dependency_candidates(manifest),
        scope_note=(
            "只盤點來源與候選依賴，不產生審閱。新增／改正文或圖需真實三輪；"
            "整檔依賴改變另核technical。歷史快照與既有豁免須由協調者查證，"
            "不能把掃描結果直接當新pass或自動擴成全書重讀。"
        ),
    )
    return result


def prepare(output, wave, authors):
    """沿用不可變舊證據；有改動的位置一律明列缺少新證據。"""
    output = local(output)
    if output.exists():
        raise ValueError("每次準備需用全新資料夾，不能覆寫舊輪次")
    manifest, prior = baseline()
    output.mkdir(parents=True)
    freeze_path = output / "source-freeze.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "docs/course-revision-20261016/tools/freeze_sources.py"),
            "--previous",
            manifest["source_freeze"]["path"],
            "--output",
            freeze_path.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = json.loads(freeze_path.read_bytes())
    diff = compare(prior, freeze)
    if diff["removed_page_ids"]:
        raise ValueError("canonical頁被移除，先修復入口／連結，不準備發布草稿")
    prior_rows = {row["page_id"]: row for row in manifest["pages"]}
    draft = copy.deepcopy(manifest)
    draft["source_freeze"] = reference(freeze_path)
    draft["author_tasks"] = sorted(set(manifest["author_tasks"]) | {"/root", *authors})
    draft["pages"] = []
    for page in freeze["pages"]:
        pid = page["page_id"]
        if pid not in diff["changed"]:
            draft["pages"].append(copy.deepcopy(prior_rows[pid]))
            continue
        draft["pages"].append({"page_id": pid, **{role: {"mode": "evidence_required", "wave": wave} for role in ROLES}})
    assignments = read_ref(manifest["assignments"])
    write_new(output / "assignments.json", assignments)
    draft["assignments"] = reference(output / "assignments.json")
    draft["review_waves"].append({"id": wave, "seals": {}})
    accepted = read_ref(ACCEPTED_BASELINE)
    numbered = numbered_scope(read_ref(accepted["source_freeze"]), freeze)
    draft["numbered_section_requirements"] = {
        "comparison_source_freeze": accepted["source_freeze"],
        "page_ids": list(numbered),
    }
    draft["preparation_note"] = "未完成草稿：保留舊證據，新頁／改頁缺少真實三輪，不得activate。"
    write_new(output / "manifest-draft.json", draft)
    write_new(output / "scope-diff.json", diff)
    return {
        "draft": str((output / "manifest-draft.json").relative_to(ROOT)),
        "source_freeze": reference(freeze_path),
        "changed_page_ids": list(diff["changed"]),
        "whole_files_needing_numbered_reviews": {pid: list(rows) for pid, rows in numbered.items()},
        "status": "incomplete_draft_without_new_reviews",
        "active_pointer_written": False,
    }


def reading_plan(freeze_path, output, reviewer, pages, context):
    if not re.fullmatch(r"/root/[a-zA-Z0-9_/-]+", reviewer) or reviewer in baseline()[0]["author_tasks"]:
        raise ValueError("需提供真正已派的獨立reader身分；此工具不派工")
    freeze = json.loads(local(freeze_path).read_bytes())
    if freeze.get("kind") == "auxiliary_numbered_technical_freeze_not_canonical":
        raise ValueError("輔助逐節freeze只供technical，不能代替reader canonical閱讀來源")
    ids = {page["page_id"] for page in freeze["pages"]}
    if not pages or len(pages) != len(set(pages)) or len(context) != len(set(context)):
        raise ValueError("main/context pages需唯一且正文非空")
    if not set(pages + context) <= ids or set(pages) & set(context):
        raise ValueError("頁ID不在凍結範圍，或正文與必要前文重疊")
    value = {
        "author_task": "/root",
        "reviewer_task": reviewer,
        "created_at": datetime.now(UTC).isoformat(),
        "source_freeze": reference(freeze_path),
        "main_pages": pages,
        "context_pages": context,
    }
    write_new(output, value)
    return {"reading_plan": reference(output), "status": "plan_only_not_reader_evidence"}


def frozen_body(page):
    raw = local(page["snapshot"]).read_bytes()
    if sha(raw) != page["source_sha256"]:
        raise ValueError("凍結正文已變：" + page["page_id"])
    return raw.decode("utf-8")


def numbered_scope(prior, freeze):
    """只替有變動的whole-file頁決定本輪逐節範圍，不用20261008作delta。"""
    reading = module("omni_reading_time", "scripts/reading_time.py")
    changed = compare(prior, freeze)["changed"]
    return {
        page["page_id"]: reading.lesson_slices(frozen_body(page))
        for page in freeze["pages"]
        if page["selector"] == "whole-file"
        and page["page_id"] in changed
        and set(changed[page["page_id"]]) & {"new_page", "source", "selector", "source_sha256", "figures_sha256"}
        and reading.lesson_slices(frozen_body(page))
    }


def freeze_numbered(freeze_path, output, pages):
    """建立舊technical自查可讀的輔助切片；它不是canonical source freeze。"""
    reading = module("omni_section_reading_time", "scripts/reading_time.py")
    canonical = json.loads(local(freeze_path).read_bytes())
    catalog = {page["page_id"]: page for page in canonical["pages"]}
    if not pages or len(pages) != len(set(pages)):
        raise ValueError("請明列唯一whole-file canonical頁ID")
    output = local(output)
    if output.exists() or output.with_suffix("").exists():
        raise ValueError("逐節凍結輸出不可覆寫")
    rows = []
    for pid in pages:
        page = catalog[pid]
        if page["selector"] != "whole-file":
            raise ValueError("只接受whole-file canonical頁：" + pid)
        sections = reading.lesson_slices(frozen_body(page))
        if not sections:
            raise ValueError("此整頁沒有編號節：" + pid)
        for sid, text in sections.items():
            snapshot = output.with_suffix("") / "sections" / (sid + ".md")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.open("x", encoding="utf-8") as stream:
                stream.write(text)
            figures = reading.figure_hashes(ROOT, local(page["source"]), text)
            if any(page["figures_sha256"].get(name) != digest for name, digest in figures.items()):
                raise ValueError("圖已偏離canonical freeze，先重新凍結")
            rows.append(
                {
                    "page_id": sid,
                    "canonical_page_id": pid,
                    "source": page["source"],
                    "selector": "section:" + sid,
                    "snapshot": snapshot.relative_to(ROOT).as_posix(),
                    "source_sha256": sha(text.encode()),
                    "figures_sha256": figures,
                }
            )
    if len({page["page_id"] for page in rows}) != len(rows):
        raise ValueError("編號節ID重複，不能派工")
    value = {
        "schema_version": 1,
        "kind": "auxiliary_numbered_technical_freeze_not_canonical",
        "canonical_source_freeze": reference(freeze_path),
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_page_ids": pages,
        "pages": rows,
    }
    write_new(output, value)
    return {
        "section_freeze": reference(output),
        "sections": [page["page_id"] for page in rows],
        "status": "sources_only_no_reviews",
    }


def original_document_versions(canonical, report):
    """只重建原判實際 freeze 保存的教材全文；不承認任意舊實作 SHA。"""
    requested = {
        item.get("path")
        for item in report.get("sources", [])
        if isinstance(item, dict) and item.get("kind") == "repository_code" and isinstance(item.get("path"), str)
    }
    versions = {}
    for source in requested:
        if not source.endswith(".md"):
            continue
        pages = [page for page in canonical["pages"] if page.get("source") == source]
        digests = set()
        for page in pages:
            if page.get("selector") == "whole-file":
                digests.add(sha(frozen_body(page).encode()))
        # 章檔以導讀＋依原 freeze 順序的全部小節精確重建，逐片都驗 SHA。
        # 不用現行章檔補片、不接受呼叫者提供的 earlier_files 清單。
        if source.startswith("course/chapters/") and source.endswith(".md"):
            introductions = [page for page in pages if page.get("selector") == "before-first-section"]
            sections = [page for page in pages if page.get("selector", "").startswith("section:")]
            if len(introductions) == 1 and sections and len(pages) == len(sections) + 1:
                introduction = frozen_body(introductions[0])
                parts = {page["page_id"]: frozen_body(page) for page in sections}
                if len(parts) != len(sections) or any(
                    page["selector"] != "section:" + page["page_id"] for page in sections
                ):
                    raise ValueError("原教材章檔小節重複或selector不符：" + source)
                body = introduction + "".join(parts.values())
                reading = module("omni_original_document_slices", "scripts/reading_time.py")
                if reading.chapter_introduction(body) != introduction or reading.lesson_slices(body) != parts:
                    raise ValueError("原教材章檔不能由freeze精確重建：" + source)
                digests.add(sha(body.encode()))
        if digests:
            versions[source] = tuple(sorted(digests))
    return versions


def original_repository_versions(report, snapshots, api):
    """舊程式只接受原判已登記指紋所對應的真保存bytes，不收任意舊SHA。"""
    if not isinstance(snapshots, list):
        raise ValueError("原程式快照需為清單")
    known = {
        (item.get("path"), item.get("sha256"))
        for item in report.get("sources", [])
        if isinstance(item, dict)
        and item.get("kind") == "repository_code"
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    }
    versions, seen = {}, set()
    for item in snapshots:
        if not isinstance(item, dict) or set(item) != {"source", "snapshot"}:
            raise ValueError("原程式快照需明列source及snapshot reference")
        path = item["source"]
        local(path)
        raw = api.artifact(ROOT, item["snapshot"])
        digest = sha(raw)
        identity = (path, digest)
        if identity not in known or identity in seen:
            raise ValueError("快照不是原判已登記的唯一程式版本：" + str(path))
        seen.add(identity)
        versions.setdefault(path, []).append(digest)
    return versions


def check_original(report, freeze_ref, lesson_id, parent_id, task, validator, api, original_source_snapshots=None):
    """核原判結構及它真正讀過的凍結版本；revise不被改寫成pass。"""
    freeze = read_ref(freeze_ref)
    page = next((row for row in freeze["pages"] if row["page_id"] == lesson_id), None)
    if page is None:
        raise ValueError("原判freeze缺當節／整頁：" + lesson_id)
    body = frozen_body(page)
    canonical = freeze
    if lesson_id != parent_id:
        if freeze.get("kind") != "auxiliary_numbered_technical_freeze_not_canonical":
            raise ValueError("逐節原判需其原auxiliary freeze")
        canonical = read_ref(freeze["canonical_source_freeze"])
        parent = next(row for row in canonical["pages"] if row["page_id"] == parent_id)
        reading = module("omni_original_sections", "scripts/reading_time.py")
        if (
            page.get("canonical_page_id") != parent_id
            or page["source"] != parent["source"]
            or reading.lesson_slices(frozen_body(parent)).get(lesson_id) != body
        ):
            raise ValueError("原判section不是其原canonical稿的精確切片")
    source, _, fragment = report.get("source", "").partition("#")
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("review_stage") != "technical"
        or report.get("reviewer_context") != "fresh"
        or report.get("reviewer_task") != task
        or report.get("lesson_id") != lesson_id
        or source != page["source"]
        or (fragment and fragment != lesson_id)
        or report.get("source_sha256") != page["source_sha256"]
        or report.get("figure_sha256") != page["figures_sha256"]
        or report.get("verdict") not in {"pass", "revise"}
    ):
        raise ValueError("原判schema／source／版本／身分不完整：" + lesson_id)
    references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
    references += re.findall(r'<img\b[^>]*\bsrc=[\'"]([^\'"]+)[\'"]', body)
    expected_figures = {}
    for name in references:
        if "://" in name or name.startswith("data:"):
            continue
        target = local(page["source"]).parent / name.split("#", 1)[0]
        rel = local(target).relative_to(ROOT).as_posix()
        expected_figures[rel] = canonical["figure_snapshots"][rel]["sha256"]
    if page["figures_sha256"] != expected_figures:
        raise ValueError("原判圖清單沒有完整對應原正文")
    for name, digest in page["figures_sha256"].items():
        ref = canonical["figure_snapshots"][name]
        if ref["sha256"] != digest:
            raise ValueError("原判圖不是其原freeze版本")
        api.artifact(ROOT, ref)
    errors = []
    claims = validator._items(report.get("claims"), errors, "claims")
    artifacts = validator._artifacts(ROOT, report.get("artifacts"), errors)
    # 原判可以尚未核實；已提供的來源、artifact與宣稱verified的主張，仍須有完整證據。
    # 原判引用的教材舊全文須能由原canonical freeze逐byte還原。
    # 現行報告仍由原_validate核現行來源；舊程式另需原判指紋及真保存bytes雙重綁定。
    earlier_files = original_document_versions(canonical, report)
    snapshots = [] if original_source_snapshots is None else original_source_snapshots
    for path, digests in original_repository_versions(report, snapshots, api).items():
        earlier_files[path] = tuple(sorted(set(earlier_files.get(path, ())) | set(digests)))
    source_errors = []
    sources = validator._sources(ROOT, report.get("sources"), artifacts, source_errors, earlier_files)
    verified_source_ids = {
        item.get("source_id")
        for claim in claims.values()
        if claim.get("status") == "verified"
        for item in claim.get("evidence", [])
        if isinstance(item, dict)
    }
    allowed_status_errors = set()
    for identifier, source in sources.items():
        if report.get("verdict") != "revise" or identifier in verified_source_ids:
            continue
        # 沒能查到原文的真原判可以明記false，但不能省略title、版本或實際失敗記錄。
        if source.get("verified") is False and validator._text(source.get("title")):
            allowed_status_errors.add(f"source {identifier}: 需要title與實際核對的verified=true")
        if source.get("checked_original") is False and validator._text(source.get("inspection_note")):
            allowed_status_errors.add(f"source {identifier}: 必須讀原始來源，不能只沿用來源庫摘要")
    errors.extend(error for error in source_errors if error not in allowed_status_errors)
    validator._claims(
        [claim for claim in claims.values() if claim.get("status") == "verified"], sources, artifacts, errors
    )
    groups = {"claims": claims, "sources": sources, "artifacts": artifacts}
    for claim in groups["claims"].values():
        if (
            claim.get("kind") not in validator.CLAIM_KINDS
            or claim.get("status") not in {"verified", "unresolved", "contradicted"}
            or any(not validator._text(claim.get(key)) for key in ("statement", "location", "scope"))
        ):
            errors.append("原判claim缺必要結構")
        if claim.get("status") != "verified":
            # 未執行／未找到證據可保留unresolved；不強迫假填executed或verified。
            evidence = claim.get("evidence")
            if not isinstance(evidence, list):
                errors.append("原判未決claim需明列evidence（無證據時使用空清單）")
                evidence = []
            for item in evidence:
                if (
                    not isinstance(item, dict)
                    or item.get("source_id") not in sources
                    or not validator._text(item.get("locator"))
                    or not validator._text(item.get("supports"))
                ):
                    errors.append("原判未決claim的來源引用／定位不完整")
            validator._references(claim.get("artifact_ids", []), artifacts, errors, "原判未決claim artifact_ids")
    if not groups["claims"]:
        applicability = report.get("applicability", {})
        if applicability.get("substantive_claims") is not False or not applicability.get("reason") or "```" in body:
            errors.append("原判缺實質主張或合法NA理由")
    if not isinstance(report.get("issues"), list) or any(not isinstance(row, dict) for row in report.get("issues", [])):
        errors.append("原判缺issues清單")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append("原判缺checks字典")
        checks = {}
    for name in validator.CHECKS:
        check = checks.get(name, {})
        if not isinstance(check, dict):
            errors.append("原判check需具名字典：" + name)
            continue
        if not validator._text(check.get("details")) or check.get("status") not in {
            "pass",
            "revise",
            "not_applicable",
            "unresolved",
        }:
            errors.append("原判check缺狀態／核對內容：" + name)
        validator._references(check.get("claim_ids", []), claims, errors, name + " claim_ids")
        if report.get("verdict") == "pass":
            na_allowed = (
                (name in {"factual_accuracy", "source_verification", "limitations"} and not claims)
                or (
                    name == "numeric_verification"
                    and not any(row.get("kind") in {"numeric", "empirical"} for row in claims.values())
                )
                or (name == "figure_consistency" and not expected_figures)
            )
            if check.get("status") != "pass" and not (check.get("status") == "not_applicable" and na_allowed):
                errors.append("原判宣稱pass但check未通過或不適用的NA：" + name)
    if report.get("verdict") == "pass" and (
        any(row.get("status") != "verified" for row in groups["claims"].values())
        or any(row.get("status") != "resolved" or not row.get("resolution") for row in report.get("issues", []))
    ):
        errors.append("原判宣稱pass但仍有未決主張／問題")
    if errors:
        raise ValueError(lesson_id + "原判結構：" + repr(errors))
    api.at(report["created_at"])


def check_receipt(api, sealed, ref, report, receipt, window):
    """每份報告核唯一實際收件與建立時序；callback使用自己的collection stage。"""
    wave, role = receipt["wave"], receipt["collection_role"]
    if role not in {"technical", "continuity"}:
        raise ValueError("技術報告不得提前收在reader階段")
    phase = sealed[wave][role]
    matches = [
        row
        for row in phase.get("collection_index", [])
        if row.get("path") == ref["path"] and row.get("sha256") == ref["sha256"]
    ]
    if len(matches) != 1 or matches[0].get("received_at") != receipt.get("received_at"):
        raise ValueError("缺唯一collection_index實際收件：" + ref["path"])
    created, received = api.at(report["created_at"]), api.at(receipt["received_at"])
    if ref["sha256"] not in phase["report_sha256s"] or not (
        api.at(phase["started_at"])
        <= api.at(window["not_before"])
        <= api.at(window["not_after"])
        <= api.at(phase["completed_at"])
        and api.at(window["not_before"]) <= created <= received <= api.at(phase["completed_at"]) <= datetime.now(UTC)
    ):
        raise ValueError("報告建立／派工／收件不在真實階段：" + ref["path"])
    return created, received


def check_report_pair(api, sealed, entry, original, current, assignment):
    initial, final = entry["original_collection"], entry["current_collection"]
    if initial["wave"] != assignment["wave"] or initial["collection_role"] != "technical":
        raise ValueError("原判收件wave不符其真正首次派工")
    created0, received0 = check_receipt(
        api, sealed, entry["original_report"], original, initial, assignment["dispatch_window"]
    )
    window = entry.get("current_dispatch_window")
    if window is None:
        if final["wave"] != assignment["wave"] or final["collection_role"] != initial["collection_role"]:
            raise ValueError("callback缺真正後續派工時間窗")
        window = assignment["dispatch_window"]
    created1, received1 = check_receipt(api, sealed, entry["report"], current, final, window)
    if created1 < created0 or received1 < received0:
        raise ValueError("原判到current的建立／收件時序倒置")


def section_issue_index(bundle, author, api):
    """只索引真報告內的問題位置，不替作者決定必要性或處置。"""
    freeze = read_ref(bundle["section_freeze"])
    parents = {row["page_id"]: row["canonical_page_id"] for row in freeze["pages"]}
    reports, issues = {}, {}
    for entry in bundle["sections"]:
        sid = entry["lesson_id"]
        for key in ("original_report", "report"):
            ref = entry[key]
            reports[(ref["path"], ref["sha256"])] = ref
            report = read_ref(ref)
            for pointer in api.reported_issues(report):
                issues.setdefault(
                    (ref["sha256"], pointer),
                    {
                        "report": ref,
                        "json_pointer": pointer,
                        "lesson_id": sid,
                        "canonical_page_id": parents[sid],
                    },
                )
    return {
        "schema_version": 1,
        "kind": "numbered_section_issue_inventory",
        "author_task": author,
        "canonical_source_freeze": bundle["canonical_source_freeze"],
        "section_freeze": bundle["section_freeze"],
        "reports": list(reports.values()),
        "issues": list(issues.values()),
    }


def check_section_issues(manifest, bundle, entries, reports, catalog, api, *, intake):
    index_ref = bundle["issue_inventory"]
    index = read_ref(index_ref)
    author = index.get("author_task")
    if author not in manifest["author_tasks"]:
        raise ValueError("逐節問題索引缺真實作者／協調者身分")
    expected = section_issue_index(bundle, author, api)
    if index != expected:
        raise ValueError("逐節original/current問題索引不完整或改了真報告pointer")
    if index_ref not in manifest.get("issue_reports", []):
        raise ValueError("逐節問題索引未列入主issue_reports")
    inventory_ref = {"report": index_ref, "json_pointer": "/issues"}
    if inventory_ref not in manifest.get("additional_issue_inventories", []):
        raise ValueError("逐節問題沒有接入主additional_issue_inventories")
    if intake:
        return
    decisions = read_ref(manifest["dispositions"])["items"]
    for position, issue in enumerate(index["issues"]):
        matches = [
            row
            for row in decisions
            if row.get("report") == index_ref and row.get("json_pointer") == "/issues/" + str(position)
        ]
        if len(matches) != 1:
            raise ValueError("逐節問題缺主代理唯一完整disposition")
        decision, sid, pid = matches[0], issue["lesson_id"], issue["canonical_page_id"]
        for field in ("source_support", "learning_benefit", "change_burden", "reason"):
            if not isinstance(decision.get(field), str) or not decision[field].strip():
                raise ValueError("逐節問題缺主代理理解收益／處置理由：" + field)
        if (
            decision.get("final_necessity") not in {"necessary", "optional"}
            or decision.get("action") not in {"fixed", "keep", "defer", "reject"}
            or (decision["final_necessity"] == "necessary" and decision["action"] != "fixed")
            or pid not in decision.get("affected_pages", [])
        ):
            raise ValueError("必要逐節問題未修，或處置範圍／必要性不符")
        entry, current = entries[sid], reports[sid]
        if issue["report"]["sha256"] == entry["report"]["sha256"]:
            pointer = issue["json_pointer"]
        else:
            closure = current.get("original_issue_resolutions", [])
            positions = [i for i, row in enumerate(closure) if row["json_pointer"] == issue["json_pointer"]]
            if len(positions) != 1:
                raise ValueError("主代理處置沒有同reviewer原問題實核")
            pointer = "/original_issue_resolutions/" + str(positions[0])
        if decision.get("section_verification") != {"report": entry["report"], "json_pointer": pointer}:
            raise ValueError("主代理處置未綁當前同reviewer逐節實核")
        if decision["action"] == "fixed" and not any(
            row.get("report") == catalog[pid]["technical"]["report"] and row.get("json_pointer", "") == ""
            for row in decision.get("verification", [])
        ):
            raise ValueError("逐節修正缺canonical整頁collector驗證")


def check_numbered(manifest, *, intake=False, manifest_path=None):
    """額外收緊本輪whole-file逐節契約，仍交舊_validate核每份真報告。"""
    accepted = read_ref(ACCEPTED_BASELINE)
    if manifest_path is not None and reference(manifest_path) == ACCEPTED_BASELINE and manifest == accepted:
        return {"status": "authenticated_accepted_baseline", "sections": 0}
    requirement = manifest.get("numbered_section_requirements")
    if not isinstance(requirement, dict):
        raise ValueError("新candidate缺numbered_section_requirements，不能冒稱既有baseline")
    if requirement.get("comparison_source_freeze") != accepted["source_freeze"]:
        raise ValueError("comparison_source_freeze必須等於本輪固定已認證baseline，不能自比current")
    canonical = read_ref(manifest["source_freeze"])
    expected = numbered_scope(read_ref(requirement["comparison_source_freeze"]), canonical)
    if set(requirement["page_ids"]) != set(expected):
        raise ValueError("numbered-section範圍與本輪whole-file delta不同")
    if not expected:
        return {"status": "no_changed_numbered_whole_files", "sections": 0}
    if "numbered_section_reviews" not in manifest:
        raise ValueError("缺少whole-file逐節fresh technical sidecar")
    bundle_ref = manifest["numbered_section_reviews"]
    bundle = read_ref(bundle_ref)
    if bundle.get("canonical_source_freeze") != manifest["source_freeze"]:
        raise ValueError("section sidecar沒有綁定本輪canonical freeze")
    section_freeze = read_ref(bundle["section_freeze"])
    if section_freeze.get("canonical_source_freeze") != manifest["source_freeze"]:
        raise ValueError("section freeze沒有綁定同一canonical freeze")
    if section_freeze.get("kind") != "auxiliary_numbered_technical_freeze_not_canonical" or set(
        section_freeze.get("canonical_page_ids", [])
    ) != set(expected):
        raise ValueError("輔助逐節freeze種類／整頁範圍不符")
    frozen = {page["page_id"]: page for page in section_freeze["pages"]}
    bodies = {sid: (pid, body) for pid, sections in expected.items() for sid, body in sections.items()}
    entries = {entry["lesson_id"]: entry for entry in bundle["sections"]}
    if len(frozen) != len(section_freeze["pages"]) or len(entries) != len(bundle["sections"]):
        raise ValueError("重複的逐節來源／報告")
    if set(frozen) != set(bodies) or set(entries) != set(bodies):
        raise ValueError("本輪修改的whole-file各編號節需完整fresh技術證據")
    api = module("omni_section_incremental", "docs/review-tools/incremental_revision.py")
    validator = module("omni_section_technical", "scripts/check_technical_reviews.py")
    assignments = read_ref(manifest["assignments"])["tasks"]
    tasks = {row["task"]: row for row in assignments}
    if len(tasks) != len(assignments):
        raise ValueError("派工task重複")
    reader_tasks = {task for task, row in tasks.items() if row["role"] == "reader"}
    _, owners = api.technical_reports(manifest, ROOT)
    reports = {sid: read_ref(entry["report"]) for sid, entry in entries.items()}
    for sid, report in reports.items():
        owners.setdefault(report.get("reviewer_task"), []).append(sid)
    sealed = {} if intake else api.review_seals(ROOT, manifest["review_waves"], ("reader", "technical"))
    catalog = {page["page_id"]: page for page in manifest["pages"]}
    canonical_pages = {page["page_id"]: page for page in canonical["pages"]}
    authors = set(manifest["author_tasks"])
    collectors = {}
    if not intake:
        for pid in expected:
            proof = catalog[pid]["technical"]
            if proof.get("mode") != "review":
                raise ValueError("整頁需另有canonical review collector：" + pid)
            report = read_ref(proof["report"])
            task = report.get("reviewer_task")
            assigned = tasks.get(task, {})
            if (
                task in authors
                or assigned.get("role") != "technical"
                or assigned.get("context") != "fresh"
                or assigned.get("lesson_id") != pid
                or assigned.get("wave") != proof.get("wave")
            ):
                raise ValueError("整頁collector缺獨立fresh technical派工：" + pid)
            page = canonical_pages[pid]
            errors = validator._validate(
                ROOT, local(page["source"]), pid, frozen_body(page), report, owners, reader_tasks
            )
            if report.get("figure_sha256") != page["figures_sha256"] or errors:
                raise ValueError("整頁collector仍未完整通過：" + pid + ": " + repr(errors))
            original = read_ref(proof["original_report"])
            check_original(
                original,
                proof["original_source_freeze"],
                pid,
                pid,
                task,
                validator,
                api,
                original_source_snapshots=proof.get("original_source_snapshots", []),
            )
            check_report_pair(api, sealed, proof, original, report, assigned)
            if not any(
                item.get("path") == bundle_ref["path"] and item.get("sha256") == bundle_ref["sha256"]
                for item in report.get("artifacts", [])
            ):
                raise ValueError("整頁collector未以artifact綁定逐節sidecar")
            collectors[pid] = task
    for sid, (pid, body) in bodies.items():
        page, entry, report = frozen[sid], entries[sid], reports[sid]
        if (
            page.get("canonical_page_id") != pid
            or frozen_body(page) != body
            or page.get("source") != canonical_pages[pid]["source"]
            or page.get("selector") != "section:" + sid
        ):
            raise ValueError("逐節快照並非本輪整頁的精確切片：" + sid)
        expected_figures = module("omni_section_figures", "scripts/reading_time.py").figure_hashes(
            ROOT, local(page["source"]), body
        )
        if page["figures_sha256"] != expected_figures or report.get("figure_sha256") != expected_figures:
            raise ValueError("逐節圖清單不符：" + sid)
        task = report.get("reviewer_task")
        assigned = tasks.get(task, {})
        if (
            task in authors
            or assigned.get("role") != "technical"
            or assigned.get("context") != "fresh"
            or assigned.get("lesson_id") != sid
        ):
            raise ValueError("缺實際每節fresh technical派工／作者隔離：" + sid)
        original = read_ref(entry["original_report"])
        check_original(
            original,
            entry["original_section_freeze"],
            sid,
            pid,
            task,
            validator,
            api,
            original_source_snapshots=entry.get("original_source_snapshots", []),
        )
        if api.at(original["created_at"]) > api.at(report["created_at"]):
            raise ValueError("原判到current的建立時序倒置")
        issues = set(api.reported_issues(original))
        closure = report.get("original_issue_resolutions", [])
        if len(closure) != len(issues) or issues != {row.get("json_pointer") for row in closure}:
            raise ValueError("原始逐節問題缺少同reviewer的逐項收束：" + sid)
        if any(not isinstance(row.get("resolution"), str) or not row["resolution"].strip() for row in closure):
            raise ValueError("逐節問題收束缺理由：" + sid)
        errors = validator._validate(ROOT, local(page["source"]), sid, body, report, owners, reader_tasks)
        if errors:
            raise ValueError(sid + ": " + repr(errors))
        proof = catalog[pid]["technical"]
        if not intake and proof.get("mode") != "review":
            raise ValueError("整頁需另有canonical review collector，不能冒稱section-composite：" + pid)
        if not intake:
            check_report_pair(api, sealed, entry, original, report, assigned)
            if task == collectors[pid]:
                raise ValueError("逐節與整頁collector不能共用task")
    check_section_issues(manifest, bundle, entries, reports, catalog, api, intake=intake)
    if not intake:
        live = {row["page_id"]: row for row in canonical["pages"]}
        patches = api.author_patches(ROOT, manifest, live)
        earlier = {pid: api.page_versions(ROOT, page, live[pid], patches) for pid, page in live.items()}
        api.check_dispositions(
            ROOT,
            manifest,
            ("reader", "technical"),
            live,
            {task: row["role"] for task, row in tasks.items()},
            sealed,
            earlier,
        )
    return {
        "status": "schema_checked_collection_pending" if intake else "passed_metadata_checks",
        "sections": len(bodies),
        "canonical_pages": list(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inventory")
    inspect.add_argument("--output", type=Path)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--wave", required=True)
    prep.add_argument("--author", action="append", default=[])
    gate = commands.add_parser("gate")
    gate.add_argument("--manifest", type=Path, required=True)
    gate.add_argument("--stage", choices=("reader", "technical", "all"), default="all")
    gate.add_argument("--output", type=Path)
    plan = commands.add_parser("reading-plan")
    plan.add_argument("--freeze", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--reviewer", required=True)
    plan.add_argument("--page", action="append", required=True)
    plan.add_argument("--context-page", action="append", default=[])
    sections = commands.add_parser("freeze-numbered")
    sections.add_argument("--freeze", type=Path, required=True)
    sections.add_argument("--output", type=Path, required=True)
    sections.add_argument("--page", action="append", required=True)
    section_check = commands.add_parser("check-numbered")
    section_check.add_argument("--manifest", type=Path, required=True)
    section_check.add_argument("--intake", action="store_true")
    section_check.add_argument("--output", type=Path)
    issue_index = commands.add_parser("index-numbered-issues")
    issue_index.add_argument("--bundle", type=Path, required=True)
    issue_index.add_argument("--output", type=Path, required=True)
    issue_index.add_argument("--author", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory()
        if args.output:
            write_new(args.output, result)
    elif args.command == "prepare":
        result = prepare(args.output, args.wave, args.author)
    elif args.command == "gate":
        result = module("omni_incremental_gate", "docs/review-tools/incremental_revision.py").audit_manifest(
            ROOT, local(args.manifest), args.stage
        )
        if args.stage in {"technical", "all"}:
            try:
                result["numbered_section_check"] = check_numbered(
                    json.loads(local(args.manifest).read_bytes()), manifest_path=args.manifest
                )
            except (ValueError, KeyError, TypeError, OSError) as error:
                result["failures"].append("numbered_section_check: " + str(error))
                result["status"] = "failed"
        if args.output:
            write_new(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False))
        raise SystemExit(1 if result["failures"] else 0)
    elif args.command == "freeze-numbered":
        result = freeze_numbered(args.freeze, args.output, args.page)
    elif args.command == "check-numbered":
        try:
            result = check_numbered(
                json.loads(local(args.manifest).read_bytes()), intake=args.intake, manifest_path=args.manifest
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            result = {"status": "failed", "failures": [str(error)]}
        if args.output:
            write_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(1 if result["status"] == "failed" else 0)
    elif args.command == "index-numbered-issues":
        api = module("omni_numbered_issues", "docs/review-tools/incremental_revision.py")
        bundle = json.loads(local(args.bundle).read_bytes())
        result = section_issue_index(bundle, args.author, api)
        write_new(args.output, result)
        result = {
            "issue_inventory": reference(args.output),
            "reports": len(result["reports"]),
            "issues": len(result["issues"]),
            "status": "index_only_no_dispositions_or_reviews",
        }
    else:
        result = reading_plan(args.freeze, args.output, args.reviewer, args.page, args.context_page)
    if args.command == "inventory":
        result = {
            key: value
            for key, value in result.items()
            if key not in {"technical_dependency_candidates", "fully_unchanged_page_ids"}
        } | {"fully_unchanged_count": len(result["fully_unchanged_page_ids"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
