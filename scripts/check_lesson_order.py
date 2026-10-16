"""核對章內連號，以及課文、索引、網站導覽與整合章路線的順序。"""

import argparse
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## ([A-Z\d]+)\.(\d+) .+$", re.M)
PAGE = re.compile(r"([A-Z\d]+)\.(\d+)\.md")


def navigation_pages(value):
    """依呈現順序取小節；字母單元可跨章分組，但組內節號仍須遞增。"""
    if isinstance(value, str):
        match = PAGE.fullmatch(value)
        return [f"{match[1]}.{match[2]}"] if match else []
    if isinstance(value, dict):
        return [page for item in value.values() for page in navigation_pages(item)]
    if isinstance(value, list):
        return [page for item in value for page in navigation_pages(item)]
    return []


def audit(root=ROOT):
    root = Path(root)
    failures, chapters = [], {}
    for path in sorted((root / "course/chapters").glob("*.md")):
        headings = HEADING.findall(path.read_text(encoding="utf-8"))
        if not headings:
            continue
        prefix = str(int(path.stem)) if path.stem.isdigit() else path.stem.removeprefix("0")
        ids = [f"{chapter}.{number}" for chapter, number in headings]
        expected = [f"{prefix}.{number}" for number in range(1, len(ids) + 1)]
        if ids != expected:
            failures.append(f"{path.relative_to(root)}: 章內節號應依序為1至{len(ids)}，實際為{', '.join(ids)}")
        chapters[prefix] = ids
    if not chapters:
        failures.append("沒有找到章節小節")
    index = json.loads((root / "course/lesson-index.json").read_text(encoding="utf-8"))
    navigation = tomllib.loads((root / "zensical.toml").read_text(encoding="utf-8"))
    nav_ids = navigation_pages(navigation["project"]["nav"])
    indexed = [row["id"] for row in index]
    for label, ids in [("course/lesson-index.json", indexed), ("zensical.toml", nav_ids)]:
        if len(ids) != len(set(ids)) or set(ids) != {pid for pages in chapters.values() for pid in pages}:
            failures.append(f"{label}: 小節有重複、缺漏或未知ID")
        for prefix, authored in chapters.items():
            actual = [pid for pid in ids if pid.split(".")[0] == prefix]
            if actual != authored:
                failures.append(f"{label}: 第{prefix}章順序與課文不同")
            if actual != sorted(actual, key=lambda pid: int(pid.split(".")[1])):
                failures.append(f"{label}: 第{prefix}章節號倒序")
    routes = json.loads((root / "course/reading-time-routes.json").read_text(encoding="utf-8"))["routes"]
    if "19" in chapters:
        capstone = next((route for route in routes if route["id"] == "capstone"), None)
        expected = ["chapter-19", *[f"19.{n}" for n in range(1, len(chapters["19"]) + 1)]]
        if capstone is None or capstone["page_ids"] != expected:
            failures.append("capstone閱讀路線應為第19章導讀，再依序走19.1至章末")
    return {
        "status": "failed" if failures else "passed",
        "chapters": len(chapters),
        "lessons": len(index),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
