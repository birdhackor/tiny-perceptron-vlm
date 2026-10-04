"""閱讀時間只使用逐頁評審資料，來源或圖解改動後不能沿用舊估時。"""

import copy
import hashlib
import json
import tomllib

import pytest

from scripts import export_course
from scripts.reading_time import (
    build_inventory,
    format_range,
    load_estimates,
    load_routes,
    render_page,
    source_sha256,
    total_range,
)


@pytest.fixture
def source_tree(tmp_path):
    (tmp_path / "course/chapters").mkdir(parents=True)
    (tmp_path / "course/figures").mkdir()
    (tmp_path / "course/figures/example.svg").write_text("<svg>one</svg>", encoding="utf-8")
    chapter = (
        "# 第 1 章\n\n這是獨立章前言。\n\n"
        "## 1.1 第一個問題\n\n逐行看這段程式。\n\n```python\nprint(1)\n```\n\n"
        "![一張圖](../figures/example.svg)\n\n"
        "## 1.2 第二個問題\n\n停下來想想結果。\n"
    )
    (tmp_path / "course/chapters/01.md").write_text(chapter, encoding="utf-8")
    (tmp_path / "course/README.md").write_text("# 閱讀指南\n\n選一個問題。\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# 專案介紹\n\n參考頁。\n", encoding="utf-8")
    index = [
        {"id": "1.1", "title": "第一個問題", "source": "course/chapters/01.md", "notebook": "notebooks/01/1.1.ipynb"},
        {"id": "1.2", "title": "第二個問題", "source": "course/chapters/01.md", "notebook": "notebooks/01/1.2.ipynb"},
    ]
    documents = {"course": "course/README.md", "readme": "README.md"}
    return tmp_path, index, documents


def make_inventory(source_tree, snapshots=None):
    root, index, documents = source_tree
    return build_inventory(root, index, documents, "# 首頁\n\n先讀第一節。\n", snapshots)


def write_metadata(path, inventory, page_ids=None):
    # 這些區間是驗證程式的合成 fixture，沒有寫入專案的正式估時 metadata。
    pages = [
        {
            "page_id": page["page_id"],
            "minutes_min": 2,
            "minutes_max": 5,
            "reviewer_task": "/test-fixtures/reader",
            "reason": "僅供測試來源核對、區間與合計的合成資料。",
            "source_sha256": page["source_sha256"],
            "figures_sha256": page["figures_sha256"],
        }
        for page in inventory["pages"]
        if page_ids is None or page["page_id"] in page_ids
    ]
    payload = {"schema_version": 1, "estimate_kind": "ai_estimate", "pages": pages}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_inventory_freezes_independent_sources_without_navigation(source_tree):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree, root / "outputs/sources")
    pages = {page["page_id"]: page for page in inventory["pages"]}
    assert set(pages) == {"1.1", "1.2", "chapter-01", "course", "readme", "index"}
    assert inventory["counts"] == {"chapter_introduction": 1, "course_guide": 1, "home": 1, "lesson": 2, "reference": 1}
    assert inventory["groups"][0]["page_ids"] == ["1.1", "1.2", "chapter-01", "course"]
    assert "本章小節" not in (root / pages["chapter-01"]["snapshot"]).read_text()
    assert "print(1)" in (root / pages["1.1"]["snapshot"]).read_text()
    assert "1.2" not in (root / pages["1.1"]["snapshot"]).read_text()
    assert pages["chapter-01"]["figures_sha256"] == {}
    assert pages["1.2"]["figures_sha256"] == {}
    assert pages["1.1"]["figures_sha256"] == {
        "course/figures/example.svg": hashlib.sha256(b"<svg>one</svg>").hexdigest()
    }
    for page in pages.values():
        assert source_sha256((root / page["snapshot"]).read_text()) == page["source_sha256"]
        assert "minutes_min" not in page


def test_complete_real_inventory_matches_every_configured_page():
    root = export_course.ROOT
    index = json.loads((root / "course/lesson-index.json").read_text())
    inventory = build_inventory(root, index, export_course.DOCUMENTS, export_course.home_introduction(index, True))
    paths = {page["page_path"] for page in inventory["pages"]}

    def nav_paths(entries):
        for entry in entries:
            for value in entry.values():
                yield from nav_paths(value) if isinstance(value, list) else [value]

    config = tomllib.loads((root / "zensical.toml").read_text())
    nav = list(nav_paths(config["project"]["nav"]))
    assert set(nav) == paths
    assert len(nav) == len(paths)
    assert len(paths) == len(index) + len({item["source"] for item in index}) + len(export_course.DOCUMENTS) + 1
    course_group = inventory["groups"][0]["page_ids"]
    assert "index" not in course_group
    assert "readme" not in course_group
    assert len(course_group) == len(set(course_group))


@pytest.mark.parametrize("changed", ["lesson", "introduction", "figure"])
def test_changed_source_or_figure_invalidates_estimate(source_tree, changed):
    root, _, _ = source_tree
    before = make_inventory(source_tree)
    path = root / "metadata.json"
    write_metadata(path, before)
    assert len(load_estimates(path, before, require_complete=True)) == 6
    if changed == "figure":
        (root / "course/figures/example.svg").write_text("<svg>two</svg>")
    else:
        source = root / "course/chapters/01.md"
        content = source.read_text()
        content = (
            content.replace("逐行看這段程式。", "改讀另一段程式。")
            if changed == "lesson"
            else content.replace("獨立章前言", "不同章前言")
        )
        source.write_text(content)
    after = make_inventory(source_tree)
    before_pages = {page["page_id"]: page for page in before["pages"]}
    after_pages = {page["page_id"]: page for page in after["pages"]}
    assert before_pages["1.2"] == after_pages["1.2"]
    if changed == "lesson":
        assert before_pages["chapter-01"] == after_pages["chapter-01"]
    with pytest.raises(ValueError, match="目前正文|圖檔 SHA"):
        load_estimates(path, after)


def test_missing_metadata_and_partial_metadata_need_explicit_completion(source_tree):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree)
    path = root / "metadata.json"
    assert load_estimates(path, inventory) == {}
    with pytest.raises(ValueError, match="缺少閱讀時間 metadata"):
        load_estimates(path, inventory, require_complete=True)
    write_metadata(path, inventory, {"1.1"})
    partial = load_estimates(path, inventory)
    assert set(partial) == {"1.1"}
    assert total_range(inventory["groups"][0]["page_ids"], partial) is None
    with pytest.raises(ValueError, match="缺少 5 頁"):
        load_estimates(path, inventory, require_complete=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("minutes_min", True),
        ("minutes_min", 0),
        ("minutes_max", 1),
        ("minutes_max", "5"),
        ("reviewer_task", " "),
        ("reason", ""),
        ("source_sha256", "old"),
        ("figures_sha256", {}),
        ("page_id", "unknown"),
    ],
)
def test_invalid_estimate_is_rejected(source_tree, field, value):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree)
    path = root / "metadata.json"
    payload = write_metadata(path, inventory)
    payload["pages"][0][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_estimates(path, inventory)


def test_duplicate_page_and_missing_figures_are_rejected(source_tree):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree)
    path = root / "metadata.json"
    payload = write_metadata(path, inventory)
    payload["pages"].append(copy.deepcopy(payload["pages"][0]))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="重複"):
        load_estimates(path, inventory)
    payload["pages"].pop()
    payload["pages"][0].pop("figures_sha256")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="圖檔 SHA"):
        load_estimates(path, inventory)


def test_routes_deduplicate_real_page_ids_and_refuse_unknown_pages(source_tree):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree)
    path = root / "routes.json"
    route = {"id": "start", "label": "初學路線", "page_ids": ["1.1", "1.1", "1.2"]}
    payload = {"schema_version": 1, "routes": [route]}
    path.write_text(json.dumps(payload))
    routes = load_routes(path, inventory)
    assert routes[0]["page_ids"] == ["1.1", "1.2"]
    metadata_path = root / "metadata.json"
    write_metadata(metadata_path, inventory)
    estimates = load_estimates(metadata_path, inventory)
    assert total_range(route["page_ids"], estimates) == (4, 10)
    payload["routes"][0]["page_ids"].append("missing-page")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical page ID"):
        load_routes(path, inventory)


def test_page_banner_preserves_body_and_distinguishes_own_time_from_totals(source_tree):
    root, _, _ = source_tree
    inventory = make_inventory(source_tree)
    path = root / "metadata.json"
    write_metadata(path, inventory)
    estimates = load_estimates(path, inventory)
    content = "# 題名 {#1.1}\n\n正文。\n\n```python\nprint('約不是估時')\n```\n"
    rendered = render_page(content, "1.1", estimates, inventory)
    assert rendered.startswith("# 題名 {#1.1}\n")
    assert rendered.count("閱讀約 2–5 分鐘 · AI 估計") == 1
    assert content.partition("\n")[2] in rendered
    assert "閱讀總量" not in rendered
    homepage = render_page(
        content, "index", estimates, inventory, [{"id": "start", "label": "<入門>", "page_ids": ["1.1", "1.1"]}]
    )
    assert "全部教材正文：約 8–20 分鐘" in homepage
    assert "&lt;入門&gt;：約 2–5 分鐘" in homepage
    assert "理解範例程式與輸出" in homepage
    assert "不是實測平均" in homepage
    assert render_page(content, "1.1", {}, inventory) == content
    missing = render_page(content, "index", {}, inventory)
    assert "閱讀約" not in missing and "閱讀總量" not in missing


def test_hour_ranges_are_rounded_outward():
    assert format_range((2, 5)) == "2–5 分鐘"
    assert format_range((59, 61), total=True) == "0.9–1.1 小時"
    assert format_range((61, 120), total=True) == "1.0–2.0 小時"


def test_export_uses_estimates_for_each_page_and_stops_before_writing_stale_data(source_tree, monkeypatch):
    root, index, documents = source_tree
    monkeypatch.setattr(export_course, "ROOT", root)
    monkeypatch.setattr(export_course, "DOCUMENTS", documents)
    (root / "course/lesson-index.json").write_text(json.dumps(index))
    (root / "notebooks/01").mkdir(parents=True)
    for item in index:
        notebook = {
            "cells": [{"cell_type": "markdown", "source": ["# " + item["title"] + "\n\n正文。"], "metadata": {}}]
        }
        (root / item["notebook"]).write_text(json.dumps(notebook))
    (root / "course/web").mkdir()
    for name in ("course.css", "course.js", "mathjax.js"):
        (root / "course/web" / name).write_text("/* test */")
    (root / "zensical.toml").write_text(
        '[project]\ndocs_dir = "outputs/docs"\nsite_dir = "outputs/site"\n'
        'extra_css = ["stylesheets/course.css"]\n'
        'nav = [{"One"="1.1.md"},{"Two"="1.2.md"},{"Intro"="chapter-01.md"},'
        '{"Guide"="course.md"},{"Reference"="readme.md"},{"Home"="index.md"}]\n'
    )
    inventory = build_inventory(root, index, documents, export_course.home_introduction(index))
    path = root / "metadata.json"
    write_metadata(path, inventory)
    destination = root / "outputs/site"

    def build_site(*args, **kwargs):
        destination.mkdir(parents=True)

    monkeypatch.setattr(export_course.subprocess, "run", build_site)
    export_course.build(
        destination, reading_times=path, reading_routes=root / "absent-routes.json", require_reading_times=True
    )
    for page in inventory["pages"]:
        content = (root / "outputs/docs" / page["page_path"]).read_text()
        assert content.count("閱讀約 2–5 分鐘 · AI 估計") == 1
    assert json.loads((destination / "build-info.json").read_text())["reading_time_complete"] is True
    previous = (root / "outputs/docs/1.1.md").read_text()
    (root / "course/chapters/01.md").write_text((root / "course/chapters/01.md").read_text().replace("逐行", "仔細"))
    with pytest.raises(ValueError, match="目前正文"):
        export_course.build(destination, reading_times=path, reading_routes=root / "absent-routes.json")
    assert (root / "outputs/docs/1.1.md").read_text() == previous
