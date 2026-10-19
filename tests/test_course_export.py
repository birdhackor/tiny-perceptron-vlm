"""網站發布必須拒絕錯誤或過期的執行副本，並保留可閱讀的前置連結。"""

import copy
import json
from html.parser import HTMLParser
from pathlib import Path, PureWindowsPath

import pytest

from scripts import build_course, export_course
from scripts.build_course import notebook_reading_links
from scripts.export_course import COURSE_URL, checked_notebook, output_html, reading_markdown, section_footer


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


@pytest.mark.parametrize(
    ("target", "expected"),
    [("#71-對話怎麼表示", "7.1"), ("05.md#52-一次看多少例子", "5.2")],
)
def test_full_heading_links_reach_the_split_lesson_in_site_and_notebook(tmp_path, monkeypatch, target, expected):
    chapters = tmp_path / "course/chapters"
    chapters.mkdir(parents=True)
    source = chapters / "07.md"
    source.write_text("## 7.1 對話怎麼表示？\n", encoding="utf-8")
    (chapters / "05.md").write_text("## 5.2 一次看多少例子？\n", encoding="utf-8")
    monkeypatch.setattr(build_course, "ROOT", tmp_path)
    prose = f"[回讀]({target})"
    lessons = {source: {"7.1": "7.1.md"}, chapters / "05.md": {"5.2": "5.2.md"}}
    assert notebook_reading_links(prose, source) == f"[回讀]({COURSE_URL}{expected}.html)"
    assert reading_markdown(prose, source, {}, lessons, "main") == f"[回讀]({expected}.md)\n"


def test_unknown_same_page_fragment_is_preserved(tmp_path, monkeypatch):
    source = tmp_path / "course/chapters/07.md"
    source.parent.mkdir(parents=True)
    source.write_text("## 7.1 對話怎麼表示？\n", encoding="utf-8")
    monkeypatch.setattr(build_course, "ROOT", tmp_path)
    prose = "[細節](#other-detail)"
    assert notebook_reading_links(prose, source) == prose
    assert reading_markdown(prose, source, {}, {source: {"7.1": "7.1.md"}}, "main") == prose + "\n"


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


# ---- 每節頁尾 ----------------------------------------------------------------------------------------------

ITEM = {"id": "9.1", "title": "標題", "source": "course/chapters/09.md", "notebook": "notebooks/09/9.1.ipynb"}
# 有教材程式的小節，頁尾說明會指向暖身頁的本機操作；沒有程式就沒有東西要跑。
RUN_GUIDE = "first-steps.md"
PROSE = "只有文字。\n"
FIGURE = "文字。\n\n![圖](../figures/demo.svg)\n\n更多文字。\n"
CODE = "文字。\n\n```python\nprint(1)\n```\n"
FIGURE_AND_CODE = "文字。\n\n![圖](../figures/demo.svg)\n\n```python\nprint(1)\n```\n"


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.found.append(dict(attrs))


def anchors_of(footer):
    parser = Anchors()
    parser.feed(footer.partition("</div>")[0])
    return parser.found


def note_of(footer):
    return footer.rpartition("</div>\n\n")[2].strip()


@pytest.fixture
def make_notebook(tmp_path, monkeypatch):
    """用 build_course 產生真實的 cell 結構，包含圖解與準備工具用的程式格。"""
    (tmp_path / "course/figures").mkdir(parents=True)
    (tmp_path / "course/figures/demo.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(build_course, "ROOT", tmp_path)
    return lambda body: build_course.notebook("9.1", "標題", body, tmp_path / "course/chapters/09.md")


def test_a_figure_only_section_still_has_code_cells_that_are_not_teaching_code(make_notebook):
    codes = [cell for cell in make_notebook(FIGURE)["cells"] if cell["cell_type"] == "code"]
    assert [("course_setup" in cell["metadata"], "course_figure" in cell["metadata"]) for cell in codes] == [
        (True, False),
        (False, True),
    ]


@pytest.mark.parametrize("executed", [False, True])
@pytest.mark.parametrize(
    ("body", "has_code"),
    [
        pytest.param(PROSE, False, id="prose-only"),
        pytest.param(FIGURE, False, id="figure-only"),
        pytest.param(CODE, True, id="python-block"),
        pytest.param(FIGURE_AND_CODE, True, id="figure-and-python-block"),
    ],
)
def test_footer_note_depends_on_whether_the_section_has_teaching_code(make_notebook, body, has_code, executed):
    note = note_of(section_footer(ITEM, make_notebook(body), "main", executed))
    assert (RUN_GUIDE in note) is has_code


def test_footer_notes_differ_between_sections_with_and_without_code(make_notebook):
    with_code = note_of(section_footer(ITEM, make_notebook(CODE), "main", True))
    without_code = note_of(section_footer(ITEM, make_notebook(PROSE), "main", True))
    assert with_code and without_code and with_code != without_code


def test_footer_claims_outputs_only_when_the_build_carries_them(make_notebook):
    notebook = make_notebook(CODE)
    with_outputs = note_of(section_footer(ITEM, notebook, "main", True))
    without_outputs = note_of(section_footer(ITEM, notebook, "main", False))
    assert "輸出" in with_outputs and "輸出" not in without_outputs
    assert RUN_GUIDE in with_outputs and RUN_GUIDE in without_outputs
    prose = make_notebook(PROSE)
    assert note_of(section_footer(ITEM, prose, "main", True)) == note_of(section_footer(ITEM, prose, "main", False))


@pytest.mark.parametrize("body", [PROSE, CODE], ids=["no-code", "code"])
def test_download_and_colab_are_equal_buttons_with_the_links_the_site_check_expects(make_notebook, body):
    download, colab = anchors_of(section_footer(ITEM, make_notebook(body), "abc123", True))
    assert download["href"] == ITEM["notebook"] and "download" in download
    assert colab["href"] == (
        f"https://colab.research.google.com/github/{export_course.REPOSITORY}/blob/abc123/{ITEM['notebook']}"
    )
    assert colab["target"] == "_blank"
    assert [download["class"], colab["class"]] == ["md-button", "md-button"]


COURSE_FILES = {
    "course/chapters/01.md": (
        "# 第 1 章\n\n導言。\n\n"
        "## 1.1 有程式\n\n文字。\n\n```python\nprint(1)\n```\n\n"
        "## 1.2 只有文字\n\n文字。\n\n"
        "## 1.3 只有圖解\n\n![圖](../figures/demo.svg)\n"
    ),
    "course/figures/demo.svg": "<svg/>",
    "course/README.md": "# 閱讀指南\n\n選一節。\n",
    "README.md": "# 專案\n\n參考。\n",
    "course/lesson-contract.json": json.dumps({"schema_version": 1, "lesson_ids": ["1.1", "1.2", "1.3"]}),
    "course/web/course.css": "/* test */",
    "course/web/course.js": "/* test */",
    "course/web/mathjax.js": "/* test */",
    "zensical.toml": (
        '[project]\ndocs_dir = "outputs/docs"\nsite_dir = "outputs/site"\n'
        'nav = ["1.1.md", "1.2.md", "1.3.md", "chapter-01.md", "course.md", "readme.md", "index.md"]\n'
    ),
}


def build_fake_course(root, monkeypatch):
    """在假的課程根目錄寫入三節（有程式、只有文字、只有圖解），用 build_course 產生 Notebook、索引與小節表。"""
    for relative, text in COURSE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    monkeypatch.setattr(build_course, "ROOT", root)
    build_course.build()


def build_fake_site(root, monkeypatch, executed):
    """在假的課程根目錄建置三節（有程式、只有文字、只有圖解），回傳各節產生的 Markdown 頁面。"""
    build_fake_course(root, monkeypatch)
    executed_root = None
    if executed:
        executed_root = root / "outputs/notebooks"
        (executed_root / "01").mkdir(parents=True)
        for source in (root / "notebooks/01").glob("*.ipynb"):
            notebook = json.loads(source.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["execution_count"] = 1
                    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
            (executed_root / "01" / source.name).write_text(json.dumps(notebook), encoding="utf-8")
    monkeypatch.setattr(export_course, "ROOT", root)
    monkeypatch.setattr(export_course, "DOCUMENTS", {"course": "course/README.md", "readme": "README.md"})
    monkeypatch.setattr(export_course, "version", lambda package: "0.0.0-test")
    monkeypatch.setattr(export_course, "prepare_assets", lambda root: None)
    monkeypatch.setattr(export_course, "stage_assets", lambda destination, prepared: None)
    destination = root / "outputs/site"
    monkeypatch.setattr(export_course.subprocess, "run", lambda *args, **kwargs: destination.mkdir(parents=True))
    export_course.build(destination, executed_root, "abc123", root / "absent.json", root / "absent-routes.json")
    return {
        lesson: (root / "outputs/docs" / f"{lesson}.md").read_text(encoding="utf-8") for lesson in ("1.1", "1.2", "1.3")
    }


def test_each_published_page_gets_the_footer_for_its_own_section(tmp_path, monkeypatch):
    pages = build_fake_site(tmp_path, monkeypatch, executed=False)
    assert {lesson: RUN_GUIDE in note_of(page) for lesson, page in pages.items()} == {
        "1.1": True,
        "1.2": False,
        "1.3": False,
    }
    for lesson, page in pages.items():
        links = [anchor["href"] for anchor in anchors_of(page.rpartition('<div class="actions">')[2])]
        assert links == [
            f"notebooks/01/{lesson}.ipynb",
            f"https://colab.research.google.com/github/{export_course.REPOSITORY}/blob/abc123/notebooks/01/{lesson}.ipynb",
        ]


def test_the_published_footer_claims_outputs_only_for_a_build_with_executed_notebooks(tmp_path, monkeypatch):
    plain = build_fake_site(tmp_path / "plain", monkeypatch, executed=False)
    executed = build_fake_site(tmp_path / "executed", monkeypatch, executed=True)
    assert "輸出" in note_of(executed["1.1"]) and "輸出" not in note_of(plain["1.1"])
    assert note_of(executed["1.2"]) == note_of(plain["1.2"])


# ---- 索引與小節表裡的路徑 ----------------------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_index_paths_use_slashes_even_for_windows_paths():
    # PureWindowsPath 在任何作業系統上都照 Windows 的規則；對它取 str() 會得到反斜線。
    root = PureWindowsPath(r"C:\Users\student\tiny-perceptron-vlm")
    assert build_course.repo_relative(root / r"course\chapters\01.md", root) == "course/chapters/01.md"
    assert build_course.repo_relative(root / r"notebooks\01\1.1.ipynb", root) == "notebooks/01/1.1.ipynb"


def test_rebuilt_index_and_lesson_table_use_the_slash_paths_of_the_committed_files(tmp_path, monkeypatch):
    # 主機的路徑寫法只有在 Windows 上跑這條時才看得到；轉換本身由
    # test_index_paths_use_slashes_even_for_windows_paths 在每個作業系統上核對。
    committed = {
        item["id"]: item
        for item in json.loads((REPOSITORY_ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    }
    committed_table = (REPOSITORY_ROOT / "course/lessons.md").read_text(encoding="utf-8")
    build_fake_course(tmp_path, monkeypatch)
    rebuilt = json.loads((tmp_path / "course/lesson-index.json").read_text(encoding="utf-8"))
    table = (tmp_path / "course/lessons.md").read_text(encoding="utf-8")
    assert [item["id"] for item in rebuilt] == ["1.1", "1.2", "1.3"]
    for item in rebuilt:
        expected = {"source": "course/chapters/01.md", "notebook": f"notebooks/01/{item['id']}.ipynb"}
        assert {key: item[key] for key in expected} == expected
        assert {key: committed[item["id"]][key] for key in expected} == expected
        for link in (f"[閱讀](../{expected['source']}#{item['id']})", f"[開啟](../{expected['notebook']})"):
            assert link in table and link in committed_table
