"""CPU-only, read-only audit of the frozen R.3 navigation claims.

The output stores only the chapter introductions and headings needed to check
the map, rather than pinning unrelated complete chapters. It does not execute
linked lessons, train a model, or establish any general research claim.
"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3.json"
TASK = "/root/omni_technical_coordinator/technical_r_3"
HEADING = re.compile(r"^## ([A-Z\d]+\.\d+) (.+)$", re.M)
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def excerpt(path, first, last):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    selected = b"".join(lines[first - 1 : last])
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "first_line": first,
        "last_line": last,
        "sha256_of_selected_bytes": sha(selected),
        "text": selected.decode("utf-8"),
    }


def main():
    freeze = json.loads(FREEZE.read_bytes())
    page = next(page for page in freeze["pages"] if page["page_id"] == "R.3")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert sha(raw) == page["source_sha256"]
    source = ROOT / page["source"]
    current = source.read_text(encoding="utf-8")
    section_start = re.search(r"^## R\.3 .+$", current, re.M).start()
    next_heading = re.search(r"^## ", current[section_start + 1 :], re.M)
    section_end = section_start + 1 + next_heading.start() if next_heading else len(current)
    current_section = current[section_start:section_end].encode("utf-8")
    assert current_section == raw
    body = raw.decode("utf-8")

    links = []
    for label, target in LINK.findall(body):
        target_path = (source.parent / target.split("#", 1)[0]).resolve()
        links.append({"label": label, "target": target, "exists": target_path.is_file()})
    assert all(item["exists"] for item in links)
    rows = [line for line in body.splitlines() if line.startswith("| [")]
    assert len(rows) == 23
    assert "主線依序是第 1–7 章、C.1–C.5、第 8–13 章、C.6–C.7，再接第 14–19 章" in body
    assert "A、B 單元與第 20 章是延伸" in body

    selected_ids = {
        "01": ["1.1", "1.12", "1.14"],
        "02": ["2.1", "2.2", "2.3"],
        "03": ["3.1", "3.3", "3.4"],
        "04": ["4.1", "4.4", "4.5", "4.6"],
        "05": ["5.1", "5.7", "5.8", "5.10"],
        "06": ["6.1", "6.2", "6.3", "6.4"],
        "07": ["7.1", "7.9", "7.11", "7.19", "7.23", "7.24", "7.25", "7.26"],
        "08": ["8.1", "8.4", "8.5", "8.15", "8.16"],
        "09": ["9.1", "9.2", "9.4", "9.15", "9.16", "9.17", "9.18"],
        "10": ["10.1", "10.2", "10.5", "10.12"],
        "11": ["11.7", "11.8", "11.14", "11.15", "11.16"],
        "12": ["12.8", "12.9", "12.17", "12.18"],
        "13": ["13.5", "13.10", "13.13", "13.15", "13.17"],
        "14": ["14.1", "14.7", "14.11", "14.19"],
        "15": ["15.1", "15.4", "15.10", "15.11", "15.13"],
        "16": ["16.3", "16.4", "16.15", "16.16"],
        "17": ["17.1", "17.2", "17.16", "17.17"],
        "18": ["18.1", "18.2", "18.8", "18.10"],
        "19": ["19.13", "19.20", "19.21", "19.22", "19.23", "19.24"],
        "20": ["20.8", "20.14", "20.15", "20.16"],
        "0A": ["A.3", "A.4", "A.5", "A.10"],
        "0B": ["B.1", "B.2", "B.3", "B.4"],
        "0C": ["C.1", "C.2", "C.3", "C.4", "C.5", "C.6", "C.7"],
    }
    chapters = {}
    chapter_sections = {}
    for stem, ids in selected_ids.items():
        path = ROOT / "course/chapters" / (stem + ".md")
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        first_section = next(i for i, line in enumerate(lines) if line.startswith("## "))
        headings = {}
        for index, line in enumerate(lines, 1):
            if match := HEADING.fullmatch(line):
                section_id = match.group(1)
                chapter_sections[section_id] = {
                    "source": path.relative_to(ROOT).as_posix(),
                    "line": index,
                    "notebook": f"notebooks/{stem}/{section_id}.ipynb",
                }
                if section_id in ids:
                    headings[section_id] = {"line": index, "text": line}
        assert set(headings) == set(ids)
        chapters[stem] = {
            "introduction": excerpt(path, 1, first_section),
            "selected_headings": headings,
        }

    # Primary text of the three junctions, checked in addition to R.3 itself.
    route_checks = {
        "after_chapter_7": "接著必修[C.1–C.5]" in chapters["07"]["introduction"]["text"],
        "after_chapter_13": "[C.6–C.7]" in chapters["13"]["introduction"]["text"],
        "C1_to_C5_returns_to_8": "讀完回到[第 8 章]" in chapters["0C"]["introduction"]["text"],
        "C6_to_C7_after_13": "等讀完[第 13 章]" in chapters["0C"]["introduction"]["text"],
        "chapter_20_extension": "延伸" in chapters["20"]["introduction"]["text"],
        "old_products_distinct_from_new_modules": "舊三個成品的權重與成績保留" in chapters["19"]["introduction"]["text"],
    }
    assert all(route_checks.values())

    lessons_path = ROOT / "course/lessons.md"
    lessons = lessons_path.read_text(encoding="utf-8")
    index = {}
    for number, line in enumerate(lessons.splitlines(), 1):
        if not line.startswith("| ") or "[閱讀](" not in line:
            continue
        label = line.split("|", 2)[1].strip()
        section_id = label.split(" ", 1)[0]
        refs = LINK.findall(line)
        assert len(refs) == 2
        index[section_id] = {"line": number, "source": refs[0][1], "notebook": refs[1][1]}
    chapter_ids = set(chapter_sections)
    listed_ids = set(index)
    missing = sorted(chapter_ids - listed_ids)
    extra = sorted(listed_ids - chapter_ids)
    bad_links = []
    for section_id, item in index.items():
        section = chapter_sections.get(section_id)
        if section is None:
            bad_links.append({"section_id": section_id, "reason": "index id absent from chapter headings"})
            continue
        for field in ["source", "notebook"]:
            path = (lessons_path.parent / item[field].split("#", 1)[0]).resolve()
            if not path.is_file():
                bad_links.append({"section_id": section_id, "field": field, "reason": "missing file"})
        if item["source"] != "../" + section["source"] + "#" + section_id:
            bad_links.append({"section_id": section_id, "reason": "source target differs"})
        if item["notebook"] != "../" + section["notebook"]:
            bad_links.append({"section_id": section_id, "reason": "notebook target differs"})
    assert not missing and not extra and not bad_links
    repo_notebooks = {path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").rglob("*.ipynb")}
    listed_notebooks = {chapter_sections[section_id]["notebook"] for section_id in listed_ids}
    unlisted_notebooks = sorted(repo_notebooks - listed_notebooks)
    assert not unlisted_notebooks

    auxiliary_unlisted = {}
    for filename in ["first-steps.md", "README.md", "training.md", "glossary.md"]:
        path = ROOT / "course" / filename
        ids = [match.group(1) for match in HEADING.finditer(path.read_text(encoding="utf-8"))]
        auxiliary_unlisted[path.relative_to(ROOT).as_posix()] = [x for x in ids if x not in listed_ids]
    reference_pages = {
        "glossary": excerpt(ROOT / "course/glossary.md", 1, 5),
        "training": excerpt(ROOT / "course/training.md", 1, 3),
        "lessons": excerpt(lessons_path, 1, 6),
    }
    product_boundary_excerpts = {
        "audio_output_limited_and_planned": excerpt(ROOT / "course/chapters/19.md", 1681, 1685),
        "new_C_not_trained": excerpt(ROOT / "course/chapters/19.md", 1889, 1899),
    }
    legacy_product_records = {}
    for name, pointers in {
        "docs/selftrained/results/v2-final-public-results.json": [
            "/scope", "/hf_repo", "/hf_revision", "/release_run",
            "/architectures/moe/selected_source", "/architectures/moe/training_revision",
            "/architectures/moe/safe_weights", "/architectures/moe/capability_acceptance_passed",
        ],
        "docs/lesson-products/results/final-report-v1.json": [
            "/recorded_at", "/scope", "/gates",
        ],
        "tiny_perceptron/lesson_products/release.json": ["/repo", "/revision", "/products"],
    }.items():
        path = ROOT / name
        original = path.read_bytes()
        document = json.loads(original)
        selected = {}
        for pointer in pointers:
            value = document
            for component in pointer.lstrip("/").split("/"):
                value = value[component]
            selected[pointer] = value
        legacy_product_records[name] = {
            "original_file_sha256": sha(original),
            "selected_fields": selected,
            "scope": "Documentary existence/provenance only; no new model inference or reproduction of training/quality.",
        }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": TASK,
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "device": "CPU; filesystem parsing only; no torch/model/GPU execution",
        },
        "source": {
            "path": page["source"],
            "snapshot": page["snapshot"],
            "sha256": sha(raw),
            "current_section_matches_snapshot": current_section == raw,
            "figure_sha256": page["figures_sha256"],
            "code_fences": body.count("```"),
        },
        "r3_direct_links": links,
        "chapter_map_rows": len(rows),
        "chapters": chapters,
        "route_checks": route_checks,
        "index_coverage": {
            "scope": "All numbered chapters 1–20 and A/B/C; auxiliary R/W/T/G reported separately.",
            "chapter_section_count": len(chapter_ids),
            "listed_section_count": len(listed_ids),
            "missing_chapter_sections": missing,
            "extra_sections": extra,
            "bad_targets": bad_links,
            "repo_notebook_count": len(repo_notebooks),
            "listed_notebook_count": len(listed_notebooks),
            "unlisted_notebooks": unlisted_notebooks,
            "auxiliary_sections_not_listed": auxiliary_unlisted,
        },
        "reference_pages": reference_pages,
        "product_boundary_excerpts": product_boundary_excerpts,
        "legacy_product_records": legacy_product_records,
        "result": "PASS: frozen source, 26 direct links, 23 chapter targets, route junctions, and complete chapter/Notebook index verified; R/W/T/G index omission explicitly recorded for reviewer judgment.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
