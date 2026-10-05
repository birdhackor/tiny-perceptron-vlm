"""網站發布必須拒絕錯誤或過期的執行副本，並保留可閱讀的前置連結。"""

import copy
import json
from pathlib import Path

import pytest

from scripts import export_course
from scripts.build_course import notebook_reading_links
from scripts.export_course import COURSE_URL, checked_notebook, output_html, reading_markdown


@pytest.mark.parametrize("problem", ["markdown", "code", "missing_cell", "not_executed", "error"])
def test_rejects_unverified_execution(tmp_path, problem):
    original = {
        "cells": [
            {"cell_type": "markdown", "source": ["一段教材"]},
            {"cell_type": "code", "source": ["print(1)"], "execution_count": None, "outputs": []},
        ]
    }
    executed = copy.deepcopy(original)
    executed["cells"][1].update(execution_count=1, outputs=[{"output_type": "stream", "text": "1\n"}])
    if problem == "markdown":
        executed["cells"][0]["source"] = ["舊的說明"]
    elif problem == "code":
        executed["cells"][1]["source"] = ["print(2)"]
    elif problem == "missing_cell":
        executed["cells"].pop()
    elif problem == "not_executed":
        executed["cells"][1]["execution_count"] = None
    else:
        executed["cells"][1]["outputs"] = [{"output_type": "error"}]
    path = tmp_path / "executed.ipynb"
    path.write_text(json.dumps(executed), encoding="utf-8")
    with pytest.raises(ValueError):
        checked_notebook(original, path)


def test_links_and_bookmarks_survive_without_editing_programs():
    source = f"## W.2 Python\n\n[前置]({COURSE_URL}first-steps.html#W.2)\n\n```python\n# 註解\nurl = '{COURSE_URL}1.1.html'\n```\n"
    result = reading_markdown(source, Path("lesson.md"), {}, {}, "main")
    assert "## W.2 Python {#W.2}" in result
    assert "[前置](first-steps.md#W.2)" in result
    assert f"# 註解\nurl = '{COURSE_URL}1.1.html'" in result


def test_execution_text_is_escaped():
    result = output_html([{"output_type": "stream", "text": ["<script>example</script>\n"]}])
    assert "&lt;script&gt;example&lt;/script&gt;" in result
    assert "<script>" not in result


def test_details_parse_markdown_in_the_published_site():
    source = "<details>\n<summary>補充</summary>\n\n### 操作\n\n[查資料](https://example.com)\n\n</details>\n"
    result = reading_markdown(source, Path("lesson.md"), {}, {}, "main")
    assert '<details markdown="1">' in result
    assert "[查資料](https://example.com)" in result
    assert "### 操作 {#操作}" in result


@pytest.mark.parametrize(
    ("filename", "page"),
    [("STUDENT.md", "natural-v4-student"), ("DATA.md", "natural-v4-data"), ("TRAINING.md", "natural-v4-training")],
)
def test_chapter_notebook_guide_link_stays_in_the_reading_site(filename, page):
    source = export_course.ROOT / "course/chapters/20.md"
    prose = f"[操作指引](../../docs/natural-assistant/v4/{filename})"
    portable = notebook_reading_links(prose, source)
    assert portable == f"[操作指引]({COURSE_URL}{page}.html)"
    published = reading_markdown(portable, source, {}, {}, "a" * 40)
    assert published == f"[操作指引]({page}.md)\n"


@pytest.mark.parametrize(
    ("slug", "other_pages"),
    [
        ("natural-v4-student", ("natural-v4-data", "natural-v4-training")),
        ("natural-v4-data", ("natural-v4-student", "natural-v4-training")),
        ("natural-v4-training", ("natural-v4-student", "natural-v4-data")),
    ],
)
def test_v4_guides_use_local_pages_and_keep_source_files_on_github(slug, other_pages):
    root = export_course.ROOT
    source = root / export_course.DOCUMENTS[slug]
    targets = {(root / path).resolve(): name + ".md" for name, path in export_course.DOCUMENTS.items()}
    chapter = root / "course/chapters/20.md"
    # The published section route is independent of the notebook regeneration step.
    lessons = {chapter.resolve(): {"20.1": "20.1.md"}}
    fixture = (
        "\n[章節](../../../course/chapters/20.md#20.1)\n"
        "[清單](manifest.json)\n[來源](data/ocr-sources.json)\n"
        "[程式](../../../scripts/natural_assistant.py)\n"
        "```bash\npython scripts/natural_assistant.py --manifest manifest.json\n```\n"
    )
    result = reading_markdown(source.read_text(encoding="utf-8") + fixture, source, targets, lessons, "a" * 40)
    for page in other_pages:
        assert f"]({page}.md)" in result
    assert "[章節](20.1.md)" in result
    external = f"https://github.com/{export_course.REPOSITORY}/blob/" + "a" * 40 + "/"
    assert f"[清單]({external}docs/natural-assistant/v4/manifest.json)" in result
    assert f"[來源]({external}docs/natural-assistant/v4/data/ocr-sources.json)" in result
    assert f"[程式]({external}scripts/natural_assistant.py)" in result
    assert "```bash\npython scripts/natural_assistant.py --manifest manifest.json\n```" in result
