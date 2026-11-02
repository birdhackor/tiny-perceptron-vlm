"""R.3 的離線導覽核對；不訓練或推論任何模型。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
FROZEN = ROOT / "docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.3.md"
EXPECTED = "d3c412956d74649c5d63239b7dd242e6c83a86eb423c92f937d6dc310e414443"
PROBES = {
    "01": ["1.1", "1.12"],
    "02": ["2.2", "2.3"],
    "03": ["3.2", "3.4"],
    "04": ["4.1", "4.5", "4.6"],
    "05": ["5.1", "5.7", "5.10"],
    "06": ["6.1", "6.2", "6.7"],
    "07": ["7.9", "7.19", "7.23", "7.25", "7.26"],
    "08": ["8.4", "8.5", "8.14"],
    "09": ["9.1", "9.5", "9.11", "9.16"],
    "10": ["10.1", "10.3", "10.12"],
    "11": ["11.8", "11.14", "11.15"],
    "12": ["12.8", "12.17", "12.18"],
    "13": ["13.10", "13.13", "13.17"],
    "14": ["14.1", "14.11", "14.19"],
    "15": ["15.1", "15.4", "15.11"],
    "16": ["16.15", "16.16"],
    "17": ["17.1", "17.15", "17.16", "17.17"],
    "18": ["18.1", "18.2", "18.3"],
    "19": ["19.13", "19.20", "19.21", "19.22", "19.25"],
    "20": ["20.1", "20.14", "20.15", "20.16"],
    "0A": ["A.3", "A.9", "A.10"],
    "0B": ["B.1", "B.2", "B.3", "B.4"],
    "0C": ["C.1", "C.3", "C.4", "C.5", "C.6", "C.7"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_map(text):
    heads = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", text, re.M))
    result = {}
    for index, heading in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        raw = text[heading.end():end]
        paragraph = next((p.strip() for p in raw.split("\n\n") if p.strip()), "")
        result[heading[1]] = {
            "heading": heading[0],
            "line": text[:heading.start()].count("\n") + 1,
            "first_paragraph": paragraph,
        }
    return result


def main():
    failures = []
    frozen = FROZEN.read_text(encoding="utf-8")
    readme = (ROOT / "course/README.md").read_text(encoding="utf-8")
    live = re.search(r"^## R\.3 .+?(?=^## |\Z)", readme, re.M | re.S)
    live_sha = hashlib.sha256(live[0].encode()).hexdigest() if live else None
    if sha(FROZEN) != EXPECTED or live_sha != EXPECTED:
        failures.append("凍結切片或目前 README R.3 的 SHA 不符")
    links = []
    for label, destination in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", frozen):
        path = (ROOT / "course" / destination).resolve()
        exists = path.is_file() and path.is_relative_to(ROOT)
        if not exists:
            failures.append(f"導覽連結不存在：{destination}")
        links.append({"label": label, "destination": destination, "exists": exists,
                      "sha256": sha(path) if exists else None})
    if len(links) != 26:
        failures.append(f"本節相對連結數異常：{len(links)}")
    lessons_path = ROOT / "course/lessons.md"
    lessons = lessons_path.read_text(encoding="utf-8")
    chapters = {}
    total_sections = 0
    for chapter, probes in PROBES.items():
        path = ROOT / "course/chapters" / f"{chapter}.md"
        text = path.read_text(encoding="utf-8")
        sections = section_map(text)
        total_sections += len(sections)
        coverage = []
        for lesson in sections:
            reading = f"../course/chapters/{chapter}.md#{lesson}"
            notebook = f"../notebooks/{chapter}/{lesson}.ipynb"
            row = next((line for line in lessons.splitlines()
                        if f"]({reading})" in line and f"]({notebook})" in line), None)
            notebook_exists = (ROOT / "course" / notebook).is_file()
            if row is None or not notebook_exists:
                failures.append(f"目錄或 Notebook 缺項：{lesson}")
            coverage.append({"lesson": lesson, "index_row_present": row is not None,
                             "notebook_exists": notebook_exists})
        for lesson in probes:
            if lesson not in sections:
                failures.append(f"對應學習問題的小節不存在：{lesson}")
        chapters[chapter] = {
            "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "title": text.splitlines()[0],
            "intro": text[:text.find("\n## ")].strip(),
            "probes": {lesson: sections.get(lesson) for lesson in probes},
            "section_count": len(sections), "index_notebook_coverage": coverage,
        }
    c_intro = chapters["0C"]["intro"]
    route_checks = {
        "chapter7_assigns_C1_C5": "必修[C.1–C.5]" in chapters["07"]["intro"],
        "C1_C5_after_chapter7": "C.1–C.5 接在 [7.23–7.25]" in c_intro,
        "C1_C5_before_chapter8": "讀完回到[第 8 章]" in c_intro,
        "C6_C7_after_chapter13": "C.6–C.7 要用到第 13 章" in c_intro,
        "chapter20_extension": "延伸" in chapters["20"]["title"],
    }
    failures.extend(key for key, passed in route_checks.items() if not passed)
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "platform": platform.platform()},
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_path": FROZEN.relative_to(ROOT).as_posix(),
        "source_sha256": sha(FROZEN), "live_readme_R3_sha256": live_sha,
        "links": links, "link_count": len(links), "route_checks": route_checks,
        "chapters": chapters, "indexed_chapter_sections": total_sections,
        "index_sha256": sha(lessons_path),
        "glossary_title": (ROOT / "course/glossary.md").read_text(encoding="utf-8").splitlines()[0],
        "training_title": (ROOT / "course/training.md").read_text(encoding="utf-8").splitlines()[0],
        "failures": failures, "result": "pass" if not failures else "revise",
        "scope": "只驗證R.3導覽及小節/Notebook目錄覆蓋；不執行Notebook，不推論成品、不重做各章研究或能力實驗。",
    }
    (ARTIFACT / "navigation-execution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"result": report["result"], "link_count": len(links),
                      "indexed_chapter_sections": total_sections, "failures": failures,
                      "environment": report["environment"]}, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
