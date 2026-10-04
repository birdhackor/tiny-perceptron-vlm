"""Independent CPU checks for the navigation assertions in R.3; no model runs."""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs/technical-reviews/artifacts/natural-r3-navigation.json"
TOPIC_ENTRIES = {
    "01": ["1.1", "1.4", "1.6"],
    "02": ["2.1", "2.2", "2.5"],
    "03": ["3.1", "3.2", "3.4"],
    "04": ["4.1", "4.2", "4.5"],
    "05": ["5.1", "5.7", "5.10", "5.15"],
    "06": ["6.1", "6.2", "6.3", "6.7"],
    "07": ["7.1", "7.4", "7.17", "7.18"],
    "08": ["8.1", "8.5", "8.7", "8.10"],
    "09": ["9.2", "9.4", "9.6"],
    "10": ["10.2", "10.3", "10.6"],
    "11": ["11.8", "11.14", "11.15", "11.16"],
    "12": ["12.1", "12.2", "12.5", "12.10", "12.13", "12.14"],
    "13": ["13.10", "13.11", "13.15", "13.17"],
    "14": ["14.1", "14.2", "14.5"],
    "15": ["15.1", "15.3", "15.4", "15.5"],
    "16": ["16.1", "16.3", "16.8", "16.9"],
    "17": ["17.1", "17.2", "17.3", "17.8", "17.15"],
    "18": ["18.1", "18.2", "18.10", "18.11"],
    "19": ["19.2", "19.4", "19.6", "19.7", "19.12"],
    "20": ["20.1", "20.2", "20.4", "20.5", "20.6", "20.7"],
    "0A": ["A.2", "A.3", "A.4", "A.5"],
    "0B": ["B.1", "B.2", "B.3", "B.4"],
    "0C": ["C.1", "C.2", "C.3", "C.4", "C.5"],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(relative):
    raw = (ROOT / relative).read_bytes()
    return raw.decode("utf-8"), sha(raw)


def sections(text):
    headings = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", text, re.M))
    result = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        result[heading[1]] = {
            "title": heading[2],
            "line": text[: heading.start()].count("\n") + 1,
            "raw": text[heading.start() : end],
        }
    return result


readme, readme_hash = load("course/README.md")
r3 = sections(readme)["R.3"]["raw"]
assert "```" not in r3
assert not re.search(r"!\[", r3)
rows = re.findall(r"^\| \[([^]]+)\]\((chapters/[^)]+)\) \| (.+?) \|$", r3, re.M)
assert len(rows) == len(TOPIC_ENTRIES)
assert {Path(row[1]).stem for row in rows} == set(TOPIC_ENTRIES)
links = []
for label, target in re.findall(r"\[([^]]+)\]\(([^)]+)\)", r3):
    path, _, fragment = target.partition("#")
    resolved = (ROOT / "course" / path).resolve()
    assert resolved.is_relative_to(ROOT)
    assert resolved.is_file(), target
    if fragment:
        assert fragment in sections(resolved.read_bytes().decode("utf-8")), target
    links.append({"label": label, "target": target, "exists": True})

index, index_hash = load("course/lessons.md")
index_rows = re.findall(
    r"^\| ([A-Z\d]+\.\d+) (.+?) \| \[閱讀\]\(([^)]+)\) \| \[開啟\]\(([^)]+)\) \|$",
    index,
    re.M,
)
indexed = {row[0]: row for row in index_rows}
assert len(indexed) == len(index_rows)
entries = []
chapter_ids = set()
notebook_ledger = []
for label, target, question in rows:
    relative = "course/" + target
    chapter, digest = load(relative)
    lessons = sections(chapter)
    chapter_ids.update(lessons)
    selected = []
    for lesson_id in TOPIC_ENTRIES[Path(target).stem]:
        lesson = lessons[lesson_id]
        paragraphs = [p for p in lesson["raw"].split("\n\n")[1:] if p.strip()]
        selected.append({
            "id": lesson_id,
            "title": lesson["title"],
            "line": lesson["line"],
            "first_paragraph": paragraphs[0],
        })
    for lesson_id, lesson in lessons.items():
        row_id, title, reading, notebook = indexed[lesson_id]
        assert row_id == lesson_id and title == lesson["title"], lesson_id
        reading_path, _, fragment = reading.partition("#")
        assert (ROOT / "course" / reading_path).resolve() == ROOT / relative
        assert fragment == lesson_id
        notebook_path = (ROOT / "course" / notebook).resolve()
        assert notebook_path.is_relative_to(ROOT) and notebook_path.is_file(), lesson_id
        raw = notebook_path.read_bytes()
        content = json.loads(raw)
        assert content["nbformat"] == 4
        assert content["cells"][0]["cell_type"] == "markdown"
        notebook_title = "".join(content["cells"][0]["source"]).strip()
        assert notebook_title == f"# {lesson_id} {title}", lesson_id
        assert any(cell["cell_type"] == "code" for cell in content["cells"]), lesson_id
        notebook_ledger.append(f"{lesson_id}\t{notebook_path.relative_to(ROOT)}\t{sha(raw)}\n")
    entries.append({
        "table_label": label,
        "table_question": question,
        "target": relative,
        "target_sha256": digest,
        "chapter_title": chapter.splitlines()[0],
        "corresponding_notebooks_checked": len(lessons),
        "topic_entries": selected,
    })
assert chapter_ids == set(indexed)
assert indexed["7.4"][1] == "回答第一個 token 在哪個位置被預測？"
glossary, glossary_hash = load("course/glossary.md")
token_row = next(line for line in glossary.splitlines() if line.startswith("| token、tokenizer |"))
assert "小單位" in token_row and "6.7" in token_row
chapter20, chapter20_hash = load("course/chapters/20.md")
twenty = sections(chapter20)
assert "逐字稿" in twenty["20.1"]["raw"] and "同一份" in twenty["20.1"]["raw"]
assert "上游完成" in twenty["20.2"]["raw"]
assert "LoRA" in twenty["20.4"]["raw"]
assert "照片" in twenty["20.5"]["raw"]
assert "中文OCR" in twenty["20.6"]["raw"]
assert "聊天模型沒有直接接收波形" in twenty["20.7"]["raw"]
result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/natural-r3-check.py",
    "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU; stdlib only; no GPU or model run"},
    "result": "All file, chapter, lesson-index, notebook-title and route assertions passed.",
    "section": "R.3",
    "source_sha256": sha(r3.encode("utf-8")),
    "current_raw_section": r3,
    "readme_sha256": readme_hash,
    "direct_links_checked": links,
    "chapter_table": entries,
    "lesson_index_sha256": index_hash,
    "indexed_chapter_sections": len(indexed),
    "matching_notebooks": len(notebook_ledger),
    "notebook_path_sha256_ledger_digest": sha("".join(sorted(notebook_ledger)).encode()),
    "glossary_sha256": glossary_hash,
    "glossary_token_row": token_row,
    "chapter20_sha256": chapter20_hash,
    "chapter20_route_only": {
        "sections": ["20.1", "20.2", "20.4", "20.5", "20.6", "20.7"],
        "draft_result_markers": re.findall(r"<!-- NATURAL_(?:DRAFT_STATUS|RESULT_[^:]+):", chapter20),
        "scope": "Existing chapter text describes the planned shared-chat / upstream-base route. No model capability, fine-tuning completion, resource use or natural-input result was verified."
    },
    "limits": "Validates local navigation and corresponding Notebook presence/title only. Does not execute Notebook cells, build the website, train models or certify chapters' technical or empirical conclusions."
}
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ["result", "source_sha256", "indexed_chapter_sections", "matching_notebooks"]}, ensure_ascii=False))
