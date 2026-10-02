"""網站發布必須拒絕錯誤或過期的執行副本，並保留可閱讀的前置連結。"""

import copy
import json
from pathlib import Path

import pytest

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
