"""R.3 bounded navigation audit; no model, dataset or training execution."""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def section(raw, lesson):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    selected = [i for i, h in enumerate(headings) if h[0].startswith(f"## {lesson} ".encode())]
    assert len(selected) == 1
    i = selected[0]
    end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
    return raw[headings[i].start():end], raw[:headings[i].start()].count(b"\n") + 1


def main():
    paths = [
        "course/README.md", "course/lessons.md", "course/lesson-contract.json",
        "course/lesson-index.json", "scripts/build_course.py",
        "docs/course-revision-20261005/outline.md",
        "docs/course-revision-20261005/rewrite-contract.md",
        "docs/review-tools/factual-reviewer-instructions.md",
        "scripts/check_technical_reviews.py", "docs/review-tools/section_facts.py",
        ".agents/skills/clear-tutorial/SKILL.md",
        ".agents/skills/clear-tutorial/references/review-protocol.md",
        "course/first-steps.md", "course/glossary.md", "course/training.md",
    ]
    chapters = [f"course/chapters/{i:02}.md" for i in range(1, 21)]
    chapters += [f"course/chapters/0{i}.md" for i in "ABC"]
    paths += chapters
    manifest = {}
    for name in paths:
        raw = (ROOT / name).read_bytes()
        target = OUT / "frozen-inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        assert digest(target.read_bytes()) == digest(raw)
        manifest[name] = {"frozen_snapshot": target.relative_to(ROOT).as_posix(),
                          "sha256": digest(raw), "bytes": len(raw)}
    save_json(OUT / "frozen-input-manifest.json", manifest)

    raw = (OUT / "frozen-inputs/course/README.md").read_bytes()
    body, first_line = section(raw, "R.3")
    (OUT / "section.md").write_bytes(body)
    assert b"```" not in body and b"~~~" not in body
    text = body.decode("utf-8")
    assert not re.search(r"!\[[^\]]*\]\([^)]*\)|<(?:img|svg)\b", text)
    assert not re.search(r"https?://", text)
    links = re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", text)
    expected = [f"chapters/{i:02}.md" for i in range(1, 21)]
    expected += [f"chapters/0{i}.md" for i in "ABC"]
    table_rows = re.findall(r"^\| \[([^\]]+)\]\(([^)]+)\) \| ([^\n]+) \|$", text, re.M)
    assert [row[1] for row in table_rows] == expected
    assert [target for _, target in links] == expected + ["lessons.md", "glossary.md", "training.md"]
    for label, target in links:
        assert (ROOT / "course" / target).is_file(), (label, target)

    inventory = []
    for name, row in zip(chapters, table_rows, strict=True):
        raw = (OUT / "frozen-inputs" / name).read_bytes()
        headings = []
        for line, title in enumerate(raw.decode("utf-8").splitlines(), 1):
            if title.startswith(("# ", "## ")):
                headings.append({"line": line, "heading": title})
        assert headings[0]["heading"].startswith("# ")
        inventory.append({"source": name, "sha256": digest(raw),
                          "frozen_snapshot": manifest[name]["frozen_snapshot"],
                          "navigation_label": row[0], "learning_question": row[2],
                          "inspected_headings": headings})
    save_json(OUT / "chapter-heading-inventory.json", inventory)

    contract = json.loads((OUT / "frozen-inputs/course/lesson-contract.json").read_bytes())
    index = json.loads((OUT / "frozen-inputs/course/lesson-index.json").read_bytes())
    assert contract["schema_version"] == 1
    ids = [entry["id"] for entry in index]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(contract["lesson_ids"])
    assert set(entry["source"] for entry in index) == set(chapters)
    actual_ids = []
    for item in inventory:
        for h in item["inspected_headings"]:
            match = re.fullmatch(r"## ([\dABC]+\.\d+) (.+)", h["heading"])
            if match:
                actual_ids.append(match[1])
    assert ids == actual_ids
    notebook_inventory = []
    for entry in index:
        path = ROOT / entry["notebook"]
        assert path.is_file()
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data["metadata"]["lesson_id"] == entry["id"]
        notebook_inventory.append({"id": entry["id"], "path": entry["notebook"],
                                   "sha256": digest(raw), "inspected_pointer": "/metadata/lesson_id"})
    save_json(OUT / "notebook-id-inventory.json", notebook_inventory)

    front = {}
    for name in ("first-steps.md", "README.md", "training.md", "glossary.md"):
        headings = re.findall(r"(?m)^## ([A-Z\d]+\.\d+) (.+)$",
                              (OUT / "frozen-inputs/course" / name).read_text())
        front[name] = [{"id": identifier, "title": title} for identifier, title in headings]
    assert not set(i["id"] for v in front.values() for i in v).intersection(ids)

    tree = ast.parse((OUT / "frozen-inputs/scripts/build_course.py").read_text())
    functions = {n.name: {"first_line": n.lineno, "last_line": n.end_lineno}
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    command = [str(ROOT / ".venv/bin/python"), "scripts/build_course.py", "--check"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    (OUT / "build-check.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (OUT / "build-check.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert f"{len(ids)} 個小節：同步檢查通過" in completed.stdout
    # A passing build must correspond to the same source bytes captured above.
    for name, record in manifest.items():
        assert digest((ROOT / name).read_bytes()) == record["sha256"], f"input changed during audit: {name}"
    environment = {"python": sys.version, "python_executable": sys.executable,
                   "platform": platform.platform(), "device": "CPU; filesystem and JSON only",
                   "scope": "No model inference, training, network or data download"}
    save_json(OUT / "environment.json", environment)
    results = {"source": "course/README.md#R.3", "source_sha256": digest(body),
               "source_first_line": first_line, "newline_policy": "raw UTF-8 bytes, no normalization",
               "frozen_whole_file_sha256": manifest["course/README.md"]["sha256"],
               "chapter_rows": len(table_rows), "local_links": len(links),
               "chapter_lessons": len(ids), "chapter_notebooks": len(notebook_inventory),
               "front_navigation_and_support_sections": front,
               "build_function_locators": functions,
               "json_inspection_pointers": {"lesson-contract.json": ["/schema_version", "/purpose", "/lesson_ids"],
                                            "lesson-index.json": ["/*/id", "/*/title", "/*/source", "/*/notebook"],
                                            "notebooks": ["/metadata/lesson_id"]},
               "python_fences": 0, "other_fences": 0, "image_references": 0,
               "build_check_command": command, "build_check_exit_code": completed.returncode,
               "build_check_stdout": completed.stdout,
               "all_assertions": "pass"}
    save_json(OUT / "navigation-results.json", results)
    print(json.dumps({k:results[k] for k in ["source_sha256", "chapter_rows", "local_links", "chapter_lessons", "chapter_notebooks", "python_fences", "image_references", "build_check_exit_code", "all_assertions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
