"""核對本輪新審閱身分與目前各來源的導言；不產生或修改審閱判定。"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/course-experiments/review-round-baseline.json"
ADDITIONS = ROOT / "docs/course-experiments/review-round-additions.json"
FRONT_MATTER = ("README.md", "first-steps.md", "training.md", "glossary.md")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_inventory(baseline, additions):
    """保留原審閱基準，新增範圍需另列；不製造不存在的舊報告。"""
    original = set(baseline["records"])
    if additions.get("baseline_revision") != baseline["baseline_revision"]:
        raise ValueError("新增範圍必須對應保留的原基準版本")
    sections = additions.get("sections", {})
    if not isinstance(sections, dict) or original & set(sections):
        raise ValueError("新增範圍必須是新小節，不能覆蓋原基準")
    for lesson, item in sections.items():
        if not re.fullmatch(r"[A-Z\d]+\.\d+", lesson) or not isinstance(item, dict):
            raise ValueError("新增範圍需有明確小節編號與來源")
        if not item.get("source", "").endswith(f"#{lesson}") or not item.get("reason", "").strip():
            raise ValueError("新增範圍需列來源與新增理由")
    return original | set(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reader", "all"), default="all")
    parser.add_argument("--output", type=Path, help="通過後才保存版本與身分核對紀錄")
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_bytes())
    additions = json.loads(ADDITIONS.read_bytes())
    expected = expected_inventory(baseline, additions)
    old_tasks = {r["reviewer_task"] for r in baseline["records"].values()}
    sources = sorted((ROOT / "course/chapters").glob("*.md"))
    sources += [ROOT / "course" / p for p in FRONT_MATTER]
    errors, records, introductions = [], [], []
    tasks = {"reader": set(), "technical": set()}
    stages = ("reader",) if args.stage == "reader" else ("reader", "technical")
    for source in sources:
        raw = source.read_bytes()
        headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
        if not headings:
            errors.append(f"沒有編號小節：{source.relative_to(ROOT)}")
            continue
        intro = raw[: headings[0].start()]
        first_id = headings[0][1].decode()
        introductions.append(
            {"source": source.relative_to(ROOT).as_posix(), "lesson_id": first_id, "sha256": digest(intro)}
        )
        for i, heading in enumerate(headings):
            lesson = heading[1].decode()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
            body_hash = digest(raw[heading.start() : end])
            item = {"lesson_id": lesson, "source_sha256": body_hash}
            for stage in stages:
                path = ROOT / f"docs/{stage}-reviews/{lesson}.json"
                if not path.is_file():
                    errors.append(f"{lesson}/{stage}: 尚無本輪報告")
                    continue
                report_raw = path.read_bytes()
                report = json.loads(report_raw)
                task = report.get("reviewer_task", "")
                if not task.startswith("/root/") or task in old_tasks or task in tasks[stage]:
                    errors.append(f"{lesson}/{stage}: 舊身分、重複或無效身分")
                tasks[stage].add(task)
                if report.get("verdict") != "pass" or report.get("source_sha256") != body_hash:
                    errors.append(f"{lesson}/{stage}: 尚未通過目前原文")
                if lesson == first_id:
                    if report.get("intro_sha256") != digest(intro) or not report.get("intro_summary", "").strip():
                        errors.append(f"{lesson}/{stage}: 導言沒有實際摘要或已改版")
                if stage == "technical" and report.get("reviewer_context") != "fresh":
                    errors.append(f"{lesson}/{stage}: 不是全新技術審閱背景")
                item[stage] = {"reviewer_task": task, "report_sha256": digest(report_raw)}
            records.append(item)
    if {r["lesson_id"] for r in records} != expected:
        errors.append("目前小節清單與原基準加明列新增範圍不同；不能略過原節或未登記的新節")
    for lesson, item in additions["sections"].items():
        if not any(
            f"{source.relative_to(ROOT).as_posix()}#{lesson}" == item["source"]
            for source in sources
            if re.search(rf"(?m)^## {re.escape(lesson)} ", source.read_text(encoding="utf-8"))
        ):
            errors.append(f"{lesson}: 新增範圍的来源與正文不一致")
    if tasks["reader"] & tasks["technical"]:
        errors.append("技術審閱沿用了本輪讀者審閱身分")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    result = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "stage": args.stage,
        "status": "passed",
        "scope": "Current raw UTF-8 section and introduction hashes, new unique reviewer identity declarations, and separation from the recorded legacy readers. Report content, claims, figures and artifacts require the separate review checkers; actual independent agent dispatch must also be checked by the coordinator. This program does not prove that a named agent read or verified a claim.",
        "checkout_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_revision": baseline["baseline_revision"],
        "baseline_sha256": digest(BASELINE.read_bytes()),
        "additions_sha256": digest(ADDITIONS.read_bytes()),
        "verification_program_sha256": digest(Path(__file__).read_bytes()),
        "sections": len(records),
        "reader_tasks": len(tasks["reader"]),
        "technical_tasks": len(tasks["technical"]),
        "introductions": introductions,
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("status", "stage", "sections", "reader_tasks", "technical_tasks")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
