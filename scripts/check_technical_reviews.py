"""離線核對逐節事實審閱的範圍、證據與版本；不代替審閱者判斷真偽。"""

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
# 與reader checker使用相同來源及編號規則；可直接python scripts/...執行。
FRONT_MATTER = ("first-steps.md", "README.md", "training.md", "glossary.md")
HEADINGS = re.compile(r"^## ([A-Z\d]+\.\d+) (.+)$", re.M)
CHECKS = ("factual_accuracy", "numeric_verification", "figure_consistency", "source_verification", "limitations")
SOURCE_KINDS = {"paper", "official_docs", "official_source", "repository_code", "derivation", "execution"}
CLAIM_KINDS = {"concept", "numeric", "software", "empirical"}
HASH = re.compile(r"^[0-9a-f]{64}$")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _kind(value):
    return value if isinstance(value, str) else ""


def _task(value, child=True):
    if not _text(value) or (value == "/root" and child):
        return False
    return value == "/root" or (
        value.startswith("/root/") and all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in value.split("/")[2:])
    )


def sections(source):
    """與reader checker相同：含##標題，直到下一個##之前的UTF-8原文。"""
    raw = source.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## .+$", raw, re.M))
    for position, heading in enumerate(headings):
        match = HEADINGS.fullmatch(heading[0])
        if not match:
            raise ValueError(f"未編號小節：{source}: {heading[0]}")
        end = headings[position + 1].start() if position + 1 < len(headings) else len(raw)
        yield match[1], raw[heading.start() : end]


def _hash_file(root, path, digest, errors, label):
    if not _text(path) or not isinstance(digest, str) or not HASH.fullmatch(digest):
        errors.append(f"{label}: 需要相對路徑與SHA-256")
        return
    target = (root / path).resolve()
    if Path(path).is_absolute() or not target.is_relative_to(root.resolve()) or not target.is_file():
        errors.append(f"{label}: 本地證據檔不存在或超出repo：{path}")
    elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        errors.append(f"{label}: 證據版本已改變：{path}")


def _items(value, errors, label):
    if not isinstance(value, list):
        errors.append(f"{label}: 必須是清單")
        return {}
    result = {}
    for item in value:
        if not isinstance(item, dict) or not _text(item.get("id")):
            errors.append(f"{label}: 每項需要唯一id")
            continue
        identifier = item["id"]
        if identifier in result:
            errors.append(f"{label}: id重複：{identifier}")
        result[identifier] = item
    return result


def _artifacts(root, value, errors):
    result = _items(value, errors, "artifacts")
    for identifier, artifact in result.items():
        label = f"artifact {identifier}"
        _hash_file(root, artifact.get("path"), artifact.get("sha256"), errors, label)
        if _text(artifact.get("path")) and not (root / artifact["path"]).resolve().is_relative_to(
            (root / "docs/technical-reviews/artifacts").resolve()
        ):
            errors.append(f"{label}: 正式證據須在docs/technical-reviews/artifacts，不能只引用暫存輸入")
        if not _text(artifact.get("description")):
            errors.append(f"{label}: 缺少證據用途")
        if _kind(artifact.get("kind")) not in {"execution", "derivation", "figure_render", "source_snapshot", "code"}:
            errors.append(f"{label}: 未知kind")
        if artifact.get("kind") == "execution":
            if not _text(artifact.get("command")) or not _text(artifact.get("result")):
                errors.append(f"{label}: 執行證據需要實際command與result")
            environment = artifact.get("environment")
            if (
                not isinstance(environment, dict)
                or not environment
                or not all(_text(key) and _text(value) for key, value in environment.items())
            ):
                errors.append(f"{label}: 執行證據需要軟體版本／裝置environment")
    return result


def _sources(root, value, artifacts, errors):
    result = _items(value, errors, "sources")
    for identifier, source in result.items():
        label, kind = f"source {identifier}", _kind(source.get("kind"))
        if kind not in SOURCE_KINDS:
            errors.append(f"{label}: 來源庫／摘要不能直接當權威來源")
        if not _text(source.get("title")) or source.get("verified") is not True:
            errors.append(f"{label}: 需要title與實際核對的verified=true")
        if kind in {"paper", "official_docs", "official_source"}:
            try:
                url = urlsplit(source.get("url", "") if isinstance(source.get("url"), str) else "")
                valid_url = url.scheme == "https" and bool(url.netloc)
            except ValueError:
                valid_url = False
            if not valid_url:
                errors.append(f"{label}: 原始來源需要HTTPS URL")
            if not _text(source.get("version")) or not _text(source.get("authority_reason")):
                errors.append(f"{label}: 缺少原始版本或權威來源理由")
            if source.get("checked_original") is not True or not _text(source.get("inspection_note")):
                errors.append(f"{label}: 必須讀原始來源，不能只沿用來源庫摘要")
            try:
                date.fromisoformat(source.get("accessed_on", ""))
            except (TypeError, ValueError):
                errors.append(f"{label}: accessed_on需為YYYY-MM-DD")
        elif kind == "repository_code":
            path = source.get("path")
            if _text(path) and (root / path).resolve().is_relative_to((root / "docs/technical-sources").resolve()):
                errors.append(f"{label}: technical-sources是候選庫，不是程式契約")
            _hash_file(root, path, source.get("sha256"), errors, label)
            if not _text(source.get("version")) or not _text(source.get("inspection_note")):
                errors.append(f"{label}: 程式來源需要版本與實際讀取記錄")
        elif kind == "derivation":
            if not _text(source.get("details")):
                errors.append(f"{label}: 算例需要可追蹤的推導")
        elif kind == "execution":
            artifact_id = source.get("artifact_id")
            artifact = artifacts.get(artifact_id) if _text(artifact_id) else None
            if artifact is None or artifact.get("kind") != "execution":
                errors.append(f"{label}: 執行來源須引用實際execution artifact")
    return result


def _references(value, available, errors, label):
    if not isinstance(value, list) or not all(_text(identifier) for identifier in value):
        errors.append(f"{label}: 需要id清單")
        return []
    for identifier in value:
        if identifier not in available:
            errors.append(f"{label}: 未登記的證據：{identifier}")
    return value


def _claims(value, sources, artifacts, errors):
    result = _items(value, errors, "claims")
    for identifier, claim in result.items():
        label, kind = f"claim {identifier}", _kind(claim.get("kind"))
        if kind not in CLAIM_KINDS:
            errors.append(f"{label}: 未知kind")
        for field in ("statement", "location", "scope"):
            if not _text(claim.get(field)):
                errors.append(f"{label}: 缺少{field}")
        if claim.get("status") != "verified":
            errors.append(f"{label}: 未核實／矛盾主張不能通過，應維持revise")
        evidence = claim.get("evidence")
        eligible = set()
        if not isinstance(evidence, list):
            errors.append(f"{label}: 需要逐來源evidence清單")
            evidence = []
        for reference in evidence:
            if not isinstance(reference, dict):
                errors.append(f"{label}: 無效的來源證據")
                continue
            source_id = reference.get("source_id")
            source = sources.get(source_id) if _text(source_id) else None
            if source is None:
                errors.append(f"{label}: 引用未登記的source_id")
            else:
                eligible.add(_kind(source.get("kind")))
            if not _text(reference.get("locator")) or not _text(reference.get("supports")):
                errors.append(f"{label}: 每個引用需要equation/section/API/行號定位與支持範圍")
        ids = _references(claim.get("artifact_ids", []), artifacts, errors, f"{label} artifact_ids")
        if not evidence and not ids:
            errors.append(f"{label}: 沒有可追蹤的證據")
        if kind == "concept" and not eligible & {"paper", "official_docs", "official_source"}:
            errors.append(f"{label}: 方法／概念主張需核對原始論文或權威文件／原始碼")
        if kind in {"numeric", "software", "empirical"}:
            verification = claim.get("verification")
            if not isinstance(verification, dict):
                errors.append(f"{label}: 缺少數值／程式驗證")
                continue
            method = _kind(verification.get("method"))
            if method not in {"hand_calculation", "executed"}:
                errors.append(f"{label}: verification.method需為hand_calculation或executed")
            for field in ("expected", "observed", "details"):
                if not _text(verification.get(field)):
                    errors.append(f"{label}: 驗證缺少{field}")
            if kind == "numeric" and not _text(verification.get("tolerance")):
                errors.append(f"{label}: 數值核對要說明容忍差或精確相等")
            if kind in {"software", "empirical"} and method != "executed":
                errors.append(f"{label}: 軟體／實測主張要有實際執行證據")
            if method == "executed" and not any(artifacts.get(item, {}).get("kind") == "execution" for item in ids):
                errors.append(f"{label}: executed需引用execution artifact")
            if kind == "empirical":
                denominator = verification.get("denominators")
                if not isinstance(denominator, dict) or not denominator:
                    errors.append(f"{label}: 實測結論需列資料／token／樣本／步數等分母")
    return result


def _validate(root, source, lesson_id, body, report, reviewer_owners, reader_tasks):
    errors = []
    if not isinstance(report, dict):
        return ["報告必須是JSON object"]
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("review_stage") != "technical"
    ):
        errors.append("必須是schema_version=1的technical審閱，不能沿用reader pass")
    recorded_source = report.get("source", "")
    path, _, fragment = recorded_source.partition("#") if isinstance(recorded_source, str) else ("", "", "")
    if (
        report.get("lesson_id") != lesson_id
        or path != source.relative_to(root).as_posix()
        or (fragment and fragment != lesson_id)
    ):
        errors.append("報告來源／小節編號不符")
    if report.get("source_sha256") != hashlib.sha256(body.encode("utf-8")).hexdigest():
        errors.append("事實審閱不是目前小節原文")
    if report.get("verdict") != "pass":
        errors.append("審閱要求修訂／尚未核實")
    reviewer = report.get("reviewer_task")
    if not _task(reviewer) or report.get("reviewer_context") != "fresh":
        errors.append("缺少獨立fresh context的reviewer_task")
    if _text(reviewer) and len(reviewer_owners.get(reviewer, [])) != 1:
        errors.append("reviewer_task被其他小節重用，每節需要新審閱者")
    if _text(reviewer) and reviewer in reader_tasks:
        errors.append("不能重用reader審閱task，兩階段需分開")
    authors = report.get("author_tasks", [])
    if not isinstance(authors, list) or not all(_task(task, child=False) for task in authors):
        errors.append("author_tasks若提供必須是實際作者task清單")
    elif reviewer in authors:
        errors.append("作者不能自行核實自己的版本與證據")
    figures = report.get("figure_sha256")
    if not isinstance(figures, dict):
        errors.append("figure_sha256必須是字典，無圖時使用{}")
        figures = {}
    referenced = set()
    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", body):
        target = (source.parent / reference).resolve()
        if not target.is_relative_to(root.resolve()):
            errors.append(f"圖解超出repo：{reference}")
        else:
            referenced.add(target.relative_to(root.resolve()).as_posix())
    for figure in sorted(referenced | set(figures)):
        _hash_file(root, figure, figures.get(figure), errors, "figure")
    artifacts = _artifacts(root, report.get("artifacts"), errors)
    sources = _sources(root, report.get("sources"), artifacts, errors)
    claims = _claims(report.get("claims"), sources, artifacts, errors)
    if not claims:
        applicability = report.get("applicability")
        if (
            not isinstance(applicability, dict)
            or applicability.get("substantive_claims") is not False
            or not _text(applicability.get("reason"))
        ):
            errors.append("無實質主張須明確記applicability=false與理由")
        if "```" in body:
            errors.append("含程式／命令區塊，仍需核對實際軟體主張，不能整節NA")
    issues = report.get("issues")
    if not isinstance(issues, list):
        errors.append("缺少issues清單")
    elif any(
        not isinstance(issue, dict) or issue.get("status") != "resolved" or not _text(issue.get("resolution"))
        for issue in issues
    ):
        errors.append("尚有未解決問題，不能記為pass")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks必須是具名字典")
        checks = {}
    for name in CHECKS:
        value = checks.get(name)
        if not isinstance(value, dict) or not _text(value.get("details")):
            errors.append(f"{name}: 缺少狀態與具體核對記錄")
            continue
        status = value.get("status")
        na_allowed = (
            (name in {"factual_accuracy", "source_verification", "limitations"} and not claims)
            or (
                name == "numeric_verification"
                and not any(c.get("kind") in {"numeric", "empirical"} for c in claims.values())
            )
            or (name == "figure_consistency" and not referenced and not figures)
        )
        if status != "pass" and not (status == "not_applicable" and na_allowed):
            errors.append(f"{name}: 未通過或不適用的NA")
        _references(value.get("claim_ids", []), claims, errors, f"{name} claim_ids")
    return errors


def check(root=ROOT, lessons=None):
    root = Path(root).resolve()
    sources = sorted((root / "course/chapters").glob("*.md"))
    sources += [root / "course" / name for name in FRONT_MATTER]
    inventory, failures = {}, []
    for source in sources:
        try:
            for lesson_id, body in sections(source):
                if lesson_id in inventory:
                    failures.append(f"{lesson_id}: 小節編號重複")
                inventory[lesson_id] = (source, body)
        except (OSError, ValueError) as error:
            failures.append(str(error))
    selected = set(inventory) if lessons is None else set(lessons)
    for missing in sorted(selected - set(inventory)):
        failures.append(f"{missing}: 沒有這個小節")
    if not selected:
        failures.append("沒有小節可核對，不能視為通過")
    reports, reviewer_owners = {}, defaultdict(list)
    for path in sorted((root / "docs/technical-reviews").glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            if lessons is None or path.stem in selected:
                failures.append(f"{path.stem}: JSON無法讀取：{error}")
            continue
        reports[path.stem] = report
        if isinstance(report, dict) and _text(report.get("reviewer_task")):
            reviewer_owners[report["reviewer_task"]].append(path.stem)
        if path.stem not in inventory and lessons is None:
            failures.append(f"{path.stem}: 報告不對應目前編號小節")
    reader_tasks = set()
    for path in (root / "docs/reader-reviews").glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and _text(report.get("reviewer_task")):
            reader_tasks.add(report["reviewer_task"])
    passed = 0
    for lesson_id in sorted(selected & set(inventory)):
        if lesson_id not in reports:
            failures.append(f"{lesson_id}: 尚未有獨立事實審閱")
            continue
        source, body = inventory[lesson_id]
        errors = _validate(root, source, lesson_id, body, reports[lesson_id], reviewer_owners, reader_tasks)
        failures.extend(f"{lesson_id}: {error}" for error in errors)
        passed += not errors
    return {
        "total_sections": len(inventory),
        "checked_sections": len(selected & set(inventory)),
        "front_sections": sum(source.parent == root / "course" for source, _ in inventory.values()),
        "passed_sections": passed,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", action="append", help="先核對一節；可重複，發布時省略以核對全部小節")
    args = parser.parse_args()
    if (ROOT / "docs/course-reviews-active.json").exists():
        spec = importlib.util.spec_from_file_location(
            "active_grouped_review", Path(__file__).resolve().parents[1] / "docs/review-tools/phase7_review.py"
        )
        active = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(active)
        if active.route_active(ROOT, "technical"):
            return
    result = check(lessons=args.lesson)
    if result["failures"]:
        print("\n".join(result["failures"]))
        raise SystemExit(1)
    print(
        f"{result['passed_sections']}/{result['checked_sections']}節：獨立事實審閱、來源與證據版本一致；"
        "真偽判斷仍由實際獨立審閱記錄負責"
    )


if __name__ == "__main__":
    main()
