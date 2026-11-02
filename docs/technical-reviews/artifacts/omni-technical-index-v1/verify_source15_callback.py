"""index 同owner source15 callback：精確依賴diff與當前導航／輸出核對。"""

import contextlib
import difflib
import hashlib
import importlib.metadata
import io
import json
import re
import sys
import tomllib
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.check_site import Links, check
from scripts.export_course import checked_notebook, home_introduction, output_html

BASE = Path(__file__).parent
CACHE = Path("/workspace/work/omni-technical-index-v1/source15-current")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(text):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", text)).split())


def selected_section(text, lesson_id):
    start = re.search(r"^## " + re.escape(lesson_id) + r" .+$", text, re.M).start()
    end = re.search(r"^## ", text[start + 3:], re.M)
    return text[start:start + 3 + end.start()] if end else text[start:]


def flatten_nav(items):
    for item in items:
        if isinstance(item, str):
            yield item
        else:
            for value in item.values():
                yield from flatten_nav(value) if isinstance(value, list) else [value]


def main():
    prior_path = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_index/technical-original-v1.json"
    prior = json.loads(prior_path.read_bytes())
    assert sha(prior_path) == "1fd5b4998be384d31c90199638b4974a9f98b0baa9c81f17737a76baba1ff4c4"
    folder = ROOT / "docs/course-revision-20261029-omni/reviews"
    old_freeze_path = folder / "source-freeze-technical-fix13.json"
    freeze_path = folder / "source-freeze-technical-fix15.json"
    assert sha(freeze_path) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
    old = {p["page_id"]: p for p in json.loads(old_freeze_path.read_bytes())["pages"]}
    current = {p["page_id"]: p for p in json.loads(freeze_path.read_bytes())["pages"]}
    changed = [id for id, page in current.items() if page["source_sha256"] != old[id]["source_sha256"]
               or page["figures_sha256"] != old[id]["figures_sha256"]]
    assert changed == ["5.7", "12.2", "chapter-10", "chapter-14"]
    diffs = {}
    for id in changed:
        before = (ROOT / old[id]["snapshot"]).read_text()
        after = (ROOT / current[id]["snapshot"]).read_text()
        diffs[id] = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                               fromfile="source13/" + id, tofile="source15/" + id, n=1))
    dependencies = []
    for source in prior["sources"]:
        if source["kind"] == "repository_code":
            digest = sha(ROOT / source["path"])
            dependencies.append({"source_id": source["id"], "path": source["path"],
                                 "original_sha256": source["sha256"], "current_sha256": digest,
                                 "unchanged": source["sha256"] == digest})
    assert [p["source_id"] for p in dependencies if not p["unchanged"]] == ["s_c12"]
    authenticated_old = ROOT / "docs/technical-reviews/artifacts/omni-author-visuals-index-fix01/chapter12-before-optional12_2.md"
    old_chapter = authenticated_old.read_text()
    chapter_path = ROOT / "course/chapters/12.md"
    chapter = chapter_path.read_text()
    assert sha(authenticated_old) == next(s["sha256"] for s in prior["sources"] if s["id"] == "s_c12")
    clause = "，也把兩行print中的頻率名稱改成9500Hz、6500Hz"
    assert chapter.count(clause) == 1 and chapter.replace(clause, "") == old_chapter
    page_12_17 = current["12.17"]
    section_path = ROOT / page_12_17["snapshot"]
    section = section_path.read_text()
    assert section.encode() == selected_section(chapter, "12.17").encode()
    assert sha(section_path) == page_12_17["source_sha256"] == old["12.17"]["source_sha256"]
    opening = section.split("```python\n", 1)[0]
    assert opening == selected_section(old_chapter, "12.17").split("```python\n", 1)[0]
    authored_start = chapter.index("## 12.17 ")
    authored_start_line = chapter[:authored_start].count("\n") + 1
    authored_end_line = chapter[:authored_start + len(opening)].count("\n")
    page = current["index"]
    snapshot = ROOT / page["snapshot"]
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    assert home_introduction(index, True).encode() == snapshot.read_bytes()
    assert page["source_sha256"] == prior["source_sha256"] == sha(snapshot)
    assert page["figures_sha256"] == prior["figure_sha256"] == {}
    site = ROOT / "outputs/omni-review-site-technical-fix15"
    executed = ROOT / "outputs/omni-notebooks-technical-fix15-current"
    nav = list(flatten_nav(tomllib.loads((ROOT / "zensical.toml").read_text())["project"]["nav"]))
    ids = {p["id"] for p in index}
    assert len(index) == len(ids) == 331
    assert all(nav.count(p["id"] + ".md") == 1 for p in index)
    search_path = site / "search.json"
    search_pages = {p["location"].split("#")[0] for p in json.loads(search_path.read_bytes())["items"]}
    assert all(p["id"] + ".html" in search_pages for p in index)
    counts = {key: 0 for key in ["notebooks", "download_bytes_match", "colab_upload_entries", "visible_code_cells",
                               "visible_code_cells_with_output", "output_fragments_found"]}
    rows, failures = [], []
    for item in index:
        original_path = ROOT / item["notebook"]
        execution_path = executed / Path(item["notebook"]).relative_to("notebooks")
        original = json.loads(original_path.read_bytes())
        notebook = checked_notebook(original, execution_path)
        counts["notebooks"] += 1
        assert (site / item["notebook"]).read_bytes() == original_path.read_bytes()
        counts["download_bytes_match"] += 1
        html_path = site / (item["id"] + ".html")
        markup = html_path.read_text()
        links = Links(); links.feed(markup)
        assert item["notebook"] in links.links
        assert "https://colab.research.google.com/" in links.links and "colab-upload-instructions" in links.ids
        counts["colab_upload_entries"] += 1
        visible = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                   and not cell.get("metadata", {}).get("course_setup")
                   and "course_figure" not in cell.get("metadata", {})]
        counts["visible_code_cells"] += len(visible)
        assert markup.count('class="language-python highlight"') >= len(visible)
        for cell in visible:
            output = output_html(cell.get("outputs", []))
            if not output:
                continue
            counts["visible_code_cells_with_output"] += 1
            present = re.search(r'src="([^"]+)"', output)[1] in markup if 'src="data:image/png;base64,' in output else plain(output) in plain(markup)
            if present:
                counts["output_fragments_found"] += 1
            else:
                failures.append(item["id"] + ": executed output fragment missing")
        rows.append({"id": item["id"], "notebook_sha256": sha(original_path), "executed_sha256": sha(execution_path),
                     "html_sha256": sha(html_path)})
    info = json.loads((site / "build-info.json").read_bytes())
    assert info["executed_cpu_outputs"] is True and info["colab_mode"] == "upload" and info["publication_ready"] is False
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        check(site)
    CACHE.mkdir(parents=True, exist_ok=True)
    details = CACHE / "all-331-navigation-output-details.json"
    detail_text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if details.exists():
        assert details.read_text() == detail_text
    else:
        with details.open("x") as file:
            file.write(detail_text)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_source15_callback.py",
        "environment": {"python": sys.version.split()[0], "torch": importlib.metadata.version("torch"),
                        "zensical": importlib.metadata.version("zensical"), "device": "CPU; file checks only; no kernel training"},
        "original_report": {"path": str(prior_path.relative_to(ROOT)), "sha256": sha(prior_path)},
        "source_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)},
        "index_snapshot": {"path": page["snapshot"], "sha256": sha(snapshot), "source_and_figures_unchanged": True},
        "changed_canonical_pages": changed, "exact_diffs": diffs, "registered_source_dependency_comparison": dependencies,
        "old_chapter12_authentication_only": {"path": str(authenticated_old.relative_to(ROOT)), "sha256": sha(authenticated_old),
                                               "current_chapter_sha256": sha(chapter_path), "only_inserted_clause": clause,
                                               "used_as_current_authority": False},
        "current_necessary_12_17": {"path": page_12_17["snapshot"], "sha256": sha(section_path),
                                   "matches_current_authored_section": True, "same_as_source13_section": True,
                                   "inspected_slice": "from ## 12.17 through immediately before first Python fence",
                                   "opening_slice_sha256": hashlib.sha256(opening.encode()).hexdigest(),
                                   "opening_slice_bytes": len(opening.encode()),
                                   "current_authored_lines": [authored_start_line, authored_end_line],
                                   "current_snapshot_lines": [1, len(opening.splitlines())],
                                   "claim_scope": "c05 only common-time problem, local support and explicitly artificial rule; no whole chapter review"},
        "counts": counts, "all_331_nav_targets_once": True, "all_331_search_pages_included": True,
        "site_check_complete_stdout": output.getvalue(), "build_info": info,
        "site_index_sha256": sha(site / "index.html"), "site_search_sha256": sha(search_path),
        "external_details": {"path": str(details), "sha256": sha(details)}, "failures": failures,
        "prior_attempt_note": "source15-callback-check.json原封保留；首跑opening slice SHA正確，但原始行號尾端誤沿用788。本次由當前字元offset計算775–786，未把Python fence/import當成opening slice。",
        "scope": "本次實際核source15資料與全部331入口／既有輸出版本，不執行331 kernels、不重審全部小節、不宣稱wave02 seal完成。",
    }
    with (BASE / "source15-callback-check-v2.json").open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2); file.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"exact_diffs", "registered_source_dependency_comparison"}}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
