"""檢查每節是否由獨立讀者審閱，並核對審閱版本與目前正文。"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT_MATTER = ("first-steps.md", "README.md", "training.md", "glossary.md")
HEADINGS = re.compile(r"^## ([A-Z\d]+\.\d+) (.+)$", re.M)
DATA_FENCES = {"text", "plaintext", "json", "jsonl"}


def has_program_fence(body):
    """Data and displayed answers are not executable programs; unknown fences stay conservative."""
    fences = re.findall(r"(?ms)^```([^\r\n]*)\r?\n.*?^```[ \t]*$", body)
    return any(info.strip().lower() not in DATA_FENCES for info in fences)


def checked(value, *, allow_not_applicable=False):
    if isinstance(value, dict):
        return value.get("status") == "pass" or (
            allow_not_applicable and value.get("status") == "not_applicable" and bool(value.get("details"))
        )
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and bool(value.strip()) and value.lower() not in {"fail", "revise"}


def sections(source):
    raw = source.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## .+$", raw, re.M))
    for position, heading in enumerate(headings):
        lesson = HEADINGS.fullmatch(heading[0])
        if not lesson:
            raise ValueError(f"未編號的讀者小節：{source.relative_to(ROOT)}: {heading[0]}")
        end = headings[position + 1].start() if position + 1 < len(headings) else len(raw)
        yield lesson[1], raw[heading.start() : end]


def main():
    if (ROOT / "docs/course-reviews-active.json").exists():
        spec = importlib.util.spec_from_file_location(
            "active_grouped_review", Path(__file__).resolve().parents[1] / "docs/review-tools/phase7_review.py"
        )
        active = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(active)
        if active.route_active(ROOT, "reader"):
            return
    sources = sorted((ROOT / "course/chapters").glob("*.md"))
    sources += [ROOT / "course" / name for name in FRONT_MATTER]
    failures, seen, reviewers = [], set(), set()
    for source in sources:
        for lesson_id, body in sections(source):
            if lesson_id in seen:
                failures.append(f"{lesson_id}: 小節編號重複")
            seen.add(lesson_id)
            path = ROOT / "docs/reader-reviews" / f"{lesson_id}.json"
            if not path.exists():
                failures.append(f"{lesson_id}: 尚未有讀者審閱")
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            recorded_source, _, recorded_section = report.get("source", "").partition("#")
            if (
                report.get("lesson_id") != lesson_id
                or recorded_source != source.relative_to(ROOT).as_posix()
                or (recorded_section and recorded_section != lesson_id)
            ):
                failures.append(f"{lesson_id}: 報告的來源不符")
            if report.get("verdict") != "pass":
                failures.append(f"{lesson_id}: 讀者要求修正")
            if report.get("source_sha256") != hashlib.sha256(body.encode("utf-8")).hexdigest():
                failures.append(f"{lesson_id}: 通過的審閱不是目前正文")
            figure_hashes = report.get("figure_sha256") or {}
            figures = {
                (source.parent / reference).resolve().relative_to(ROOT).as_posix()
                for reference in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", body)
            }
            # 前置圖解若也記入報告，其後的修改同樣需要重新核對。
            figures.update(figure_hashes)
            for figure_path in sorted(figures):
                figure = ROOT / figure_path
                if (
                    not figure.is_file()
                    or figure_hashes.get(figure_path) != hashlib.sha256(figure.read_bytes()).hexdigest()
                ):
                    failures.append(f"{lesson_id}: 圖解未審閱或審閱之後已改變：{figure_path}")
            reviewer = report.get("reviewer_task", "")
            if not reviewer.startswith("/root/") or reviewer in reviewers:
                failures.append(f"{lesson_id}: 缺少獨立的新讀者")
            reviewers.add(reviewer)
            if not report.get("reader_summary") or not isinstance(report.get("issues"), list):
                failures.append(f"{lesson_id}: 缺少讀者理解或問題記錄")
            checks = report.get("checks", {})
            if isinstance(checks, dict):
                items = list(checks.items())
            elif isinstance(checks, list):
                items = [(value.get("name", "") if isinstance(value, dict) else "", value) for value in checks]
            else:
                items = []
            if len(items) < 5 or not all(
                checked(value, allow_not_applicable=name == "program_explanation" and not has_program_fence(body))
                for name, value in items
            ):
                failures.append(f"{lesson_id}: 缺少背景、術語、例子、程式或練習的檢查")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"{len(seen)} 節：獨立讀者審閱與正文版本全部一致")


if __name__ == "__main__":
    main()
