"""Independent, read-only index review: count, links, frozen sources and saved outputs.

Does not execute lesson code, neural inference, training, or publication.
"""

import hashlib
import json
import platform
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.export_course import home_introduction, joined, output_html

FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
PREVIEW = ROOT / "outputs/omni-P5-preview-v8"
BUNDLE = ROOT / "outputs/omni-P5-current-kernel-bundle-v8"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def paths(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from paths(item)


def output_signature(div):
    return {
        "label": div.select_one(".output-label").get_text(),
        "pre": [x.get_text() for x in div.find_all("pre")],
        "images": [sha(x["src"].encode()) for x in div.find_all("img")],
    }


def main():
    freeze = json.loads(FREEZE.read_bytes())
    pages = {x["page_id"]: x for x in freeze["pages"]}
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    ids = [x["id"] for x in index]
    home = (ROOT / pages["index"]["snapshot"]).read_bytes()
    home_soup = BeautifulSoup((PREVIEW / "index.html").read_bytes(), "html.parser")
    home_article = home_soup.select_one("article.md-content__inner")
    lesson_pages = {x["page_id"] for x in freeze["pages"] if x["kind"] == "lesson"}
    nav = list(paths(tomllib.loads((ROOT / "zensical.toml").read_text())["project"]["nav"]))
    nav_lessons = [x.removesuffix(".md") for x in nav if x.removesuffix(".md") in set(ids)]
    home_nav_links = {x.get("href", "").removeprefix("./") for x in home_soup.select("nav.md-nav--primary a[href]")}
    rows = []
    failures = []
    output_cells = 0
    teaching_code_cells = 0
    for item in index:
        page_id = item["id"]
        nb_path = ROOT / item["notebook"]
        executed_path = BUNDLE / Path(item["notebook"]).relative_to("notebooks")
        current = json.loads(nb_path.read_bytes())
        executed = json.loads(executed_path.read_bytes())
        copied_nb = PREVIEW / item["notebook"]
        frozen_notebook = json.loads(copied_nb.read_bytes())
        current_signature = [(c["cell_type"], joined(c["source"])) for c in current["cells"]]
        frozen_signature = [(c["cell_type"], joined(c["source"])) for c in frozen_notebook["cells"]]
        executed_signature = [(c["cell_type"], joined(c["source"])) for c in executed["cells"]]
        source_match = current_signature == executed_signature
        page_path = PREVIEW / f"{page_id}.html"
        soup = BeautifulSoup(page_path.read_bytes(), "html.parser")
        article = soup.select_one("article.md-content__inner")
        expected_outputs = []
        cell_failures = []
        code_count = 0
        for number, cell in enumerate(executed["cells"]):
            if cell["cell_type"] != "code":
                continue
            if cell.get("execution_count") is None:
                cell_failures.append(f"cell {number}: execution_count is null")
            if any(x["output_type"] == "error" for x in cell.get("outputs", [])):
                cell_failures.append(f"cell {number}: saved error")
            if cell.get("metadata", {}).get("course_setup") or "course_figure" in cell.get("metadata", {}):
                continue
            code_count += 1
            rendered = output_html(cell.get("outputs", []))
            if rendered:
                expected_outputs.append(output_signature(BeautifulSoup(rendered, "html.parser").div))
        actual_outputs = [output_signature(x) for x in article.select("div.output")]
        actions = article.select(".actions a")
        downloads = [x for x in actions if x.has_attr("download")]
        colab_links = [x for x in actions if x.get("href", "").startswith("https://colab.research.google.com/")]
        download_ok = len(downloads) == 1 and downloads[0]["href"] == item["notebook"]
        download_bytes_equal = copied_nb.exists() and copied_nb.read_bytes() == nb_path.read_bytes()
        record = {
            "id": page_id,
            "notebook_sha256": sha(nb_path.read_bytes()),
            "notebook_matches_freeze": sha(nb_path.read_bytes()) == pages[page_id]["notebook_sha256"],
            "current_and_executed_cell_type_source_equal": source_match,
            "preview_notebook_sha256": sha(copied_nb.read_bytes()),
            "preview_notebook_matches_freeze": sha(copied_nb.read_bytes()) == pages[page_id]["notebook_sha256"],
            "frozen_preview_and_executed_cell_type_source_equal": frozen_signature == executed_signature,
            "executed_file": file_record(executed_path),
            "teaching_code_cells": code_count,
            "saved_output_blocks": len(expected_outputs),
            "saved_and_page_output_signatures_equal": expected_outputs == actual_outputs,
            "saved_output_signature_sha256": sha(json.dumps(expected_outputs, sort_keys=True).encode()),
            "has_download_footer": download_ok,
            "downloaded_notebook_bytes_equal_current": download_bytes_equal,
            "has_colab_footer": len(colab_links) == 1,
            "colab_href": colab_links[0]["href"] if colab_links else None,
            "preview_in_primary_navigation": f"{page_id}.html" in home_nav_links,
            "execution_cell_failures": cell_failures,
        }
        if not all(record[k] for k in (
            "preview_notebook_matches_freeze", "frozen_preview_and_executed_cell_type_source_equal",
            "saved_and_page_output_signatures_equal", "has_download_footer",
            "has_colab_footer", "preview_in_primary_navigation",
        )) or cell_failures:
            failures.append(page_id)
        rows.append(record)
        output_cells += len(expected_outputs)
        teaching_code_cells += code_count
    links = []
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", home.decode()):
        html_target = target.removesuffix(".md") + ".html"
        actual = home_article.find("a", string=label)
        links.append({"label": label, "target": target, "page_exists": (PREVIEW / html_target).is_file(),
                      "rendered_href": actual.get("href") if actual else None,
                      "expected_rendered_href": html_target})
    search = json.loads((PREVIEW / "search.json").read_bytes())
    title_probe = next(x for x in search["items"] if x.get("location") == "1.1.html")
    body_probe = next(x for x in search["items"] if "1.1.html" in x.get("location", "") and "字表" in x.get("text", ""))
    result = {
        "reviewer_task": "/root/omni_p5_technical_index",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; static JSON/HTML inspection only"},
        "canonical_freeze": file_record(FREEZE),
        "source_snapshot": file_record(ROOT / pages["index"]["snapshot"]),
        "exporter": file_record(ROOT / "scripts/export_course.py"),
        "audit_script": file_record(Path(__file__)),
        "lesson_index": file_record(ROOT / "course/lesson-index.json"),
        "preview_home": file_record(PREVIEW / "index.html"),
        "search_index": file_record(PREVIEW / "search.json"),
        "build_info": json.loads((PREVIEW / "build-info.json").read_bytes()),
        "home_source_sha_matches_freeze": sha(home) == pages["index"]["source_sha256"],
        "exporter_home_exact_frozen_bytes": home_introduction(index, True).encode() == home,
        "lesson_count": len(index),
        "unique_lesson_count": len(set(ids)),
        "canonical_lesson_count": len(lesson_pages),
        "index_ids_equal_canonical_lesson_ids": set(ids) == lesson_pages,
        "configured_navigation_count": len(nav_lessons),
        "configured_navigation_is_one_per_lesson": sorted(nav_lessons) == sorted(ids),
        "home_links": links,
        "search_title_probe": {k: title_probe[k] for k in ("location", "title")},
        "search_body_probe": {"location": body_probe["location"], "probe": "字表", "present_in_body": True},
        "teaching_code_cells": teaching_code_cells,
        "rendered_saved_output_blocks": output_cells,
        "failed_lesson_ids": failures,
        "live_worktree_notebooks_differing_from_canonical_freeze": [x["id"] for x in rows if not x["notebook_matches_freeze"]],
        "rows": rows,
        "limits": [
            "No saved lesson code was rerun. Execution counts and outputs are preserved records, not a fresh 331-kernel run.",
            "Output signatures establish faithful display of preserved results; they do not independently establish every lesson's numerical or scientific correctness.",
            "Only local preview was inspected. No live public deployment, Colab authentication/runtime, neural inference, training or workflow was attempted.",
            "The frozen preview's downloaded notebooks are verified against canonical notebook_sha256 values. Concurrent live working-tree drift is separately recorded and does not replace the canonical source.",
        ],
    }
    dest = Path(__file__).with_name("static-audit-v2.json")
    with dest.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"result": str(dest.relative_to(ROOT)), "lessons": len(index), "teaching_code_cells": teaching_code_cells,
                      "output_blocks": output_cells, "failed_lesson_ids": failures, "home_exact": result["exporter_home_exact_frozen_bytes"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
