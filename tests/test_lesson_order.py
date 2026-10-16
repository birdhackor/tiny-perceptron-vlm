"""章內順序必須正確，即使多份來源彼此一致也不能一起排錯。"""

import json
from pathlib import Path

from scripts.check_lesson_order import audit


def course(root, chapters, *, indexed=None, navigation=None, capstone=None):
    folder = root / "course/chapters"
    folder.mkdir(parents=True)
    authored = []
    for prefix, numbers in chapters.items():
        stem = prefix.zfill(2) if prefix.isdigit() else "0" + prefix
        ids = [f"{prefix}.{number}" for number in numbers]
        (folder / f"{stem}.md").write_text("# 導讀\n\n" + "".join(f"## {pid} 問題\n\n正文。\n\n" for pid in ids))
        authored.extend(ids)
    (root / "course/lesson-index.json").write_text(json.dumps([{"id": pid} for pid in (indexed or authored)]))
    links = "".join(f'  {{ "{pid}" = "{pid}.md" }},\n' for pid in (navigation or authored))
    (root / "zensical.toml").write_text("[project]\nnav = [\n" + links + "]\n")
    routes = []
    if "19" in chapters:
        ids = capstone if capstone is not None else ["chapter-19", *[f"19.{n}" for n in chapters["19"]]]
        routes.append({"id": "capstone", "page_ids": ids})
    (root / "course/reading-time-routes.json").write_text(json.dumps({"routes": routes}))
    return root


def test_current_repository_has_natural_lesson_order():
    result = audit(Path(__file__).resolve().parents[1])
    assert result["failures"] == []


def test_consistently_misplaced_overview_is_rejected(tmp_path):
    result = audit(course(tmp_path, {"19": [3, 1, 2]}))
    assert result["status"] == "failed"
    assert any("course/chapters/19.md" in line for line in result["failures"])
    assert any("zensical.toml" in line for line in result["failures"])


def test_consistent_jump_back_in_a_chapter_is_rejected(tmp_path):
    assert audit(course(tmp_path, {"14": [1, 3, 4, 2]}))["status"] == "failed"


def test_numeric_sort_accepts_ten_after_nine(tmp_path):
    assert audit(course(tmp_path, {"14": list(range(1, 19))}))["status"] == "passed"


def test_index_alone_cannot_change_order(tmp_path):
    result = audit(course(tmp_path, {"19": [1, 2, 3]}, indexed=["19.3", "19.1", "19.2"]))
    assert any("lesson-index.json" in line for line in result["failures"])


def test_navigation_alone_cannot_change_order(tmp_path):
    result = audit(course(tmp_path, {"19": [1, 2, 3]}, navigation=["19.3", "19.1", "19.2"]))
    assert any("zensical.toml" in line for line in result["failures"])


def test_missing_number_is_rejected(tmp_path):
    assert audit(course(tmp_path, {"14": [1, 3]}))["status"] == "failed"


def test_capstone_must_include_all_lessons_in_order(tmp_path):
    result = audit(course(tmp_path, {"19": [1, 2, 3]}, capstone=["chapter-19", "19.1", "19.3"]))
    assert any("capstone" in line for line in result["failures"])


def test_lettered_unit_may_span_navigation_groups(tmp_path):
    root = course(tmp_path, {"7": [1], "8": [1], "C": [1, 2]}, navigation=["7.1", "C.1", "8.1", "C.2"])
    assert audit(root)["status"] == "passed"
