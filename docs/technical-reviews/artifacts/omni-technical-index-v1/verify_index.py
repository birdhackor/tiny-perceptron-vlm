"""index 導覽承諾的離線核對；不把連結巡覽當全書技術審閱。"""

import hashlib
import importlib.metadata
import json
import re
import sys
import tomllib
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.export_course import checked_notebook, home_introduction, joined, output_html
from scripts.check_site import Links, check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_nav(items):
    for item in items:
        if isinstance(item, str):
            yield item
        else:
            for value in item.values():
                yield from flatten_nav(value) if isinstance(value, list) else [value]


def plain(markup):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def main():
    site = ROOT / "outputs/omni-review-site-technical-fix13"
    executed = ROOT / "outputs/omni-notebooks-technical-fix13-current"
    freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
    freeze = json.loads(freeze_path.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "index")
    snapshot = ROOT / page["snapshot"]
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    home = home_introduction(index, executed_outputs=True)
    failures = []
    assert home.encode() == snapshot.read_bytes()
    assert sha(snapshot) == page["source_sha256"]
    assert page["figures_sha256"] == {}
    assert len(index) == len({x["id"] for x in index}) == 331
    headings = []
    for source in sorted({x["source"] for x in index}):
        headings.extend(re.findall(r"^## ([A-Z\d]+\.\d+) .+$", (ROOT / source).read_text(), re.M))
    assert set(headings) == {x["id"] for x in index}
    assert len(headings) == 331
    config = tomllib.loads((ROOT / "zensical.toml").read_text())
    nav = list(flatten_nav(config["project"]["nav"]))
    assert all(nav.count(x["id"] + ".md") == 1 for x in index)
    search_path = site / "search.json"
    search = json.loads(search_path.read_bytes())
    search_pages = {item["location"].split("#")[0] for item in search["items"]}
    assert all(x["id"] + ".html" in search_pages for x in index)
    rows = []
    counters = dict(lessons=0, downloaded_notebooks=0, colab_upload_entries=0,
                    verified_executed_notebooks=0, visible_code_cells=0,
                    visible_code_cells_with_output=0, output_fragments_found=0)
    for item in index:
        original_path = ROOT / item["notebook"]
        original = json.loads(original_path.read_bytes())
        execution_path = executed / Path(item["notebook"]).relative_to("notebooks")
        nb = checked_notebook(original, execution_path)
        counters["verified_executed_notebooks"] += 1
        download_path = site / item["notebook"]
        assert download_path.read_bytes() == original_path.read_bytes()
        counters["downloaded_notebooks"] += 1
        html_path = site / (item["id"] + ".html")
        markup = html_path.read_text()
        links = Links()
        links.feed(markup)
        assert item["notebook"] in links.links
        assert "https://colab.research.google.com/" in links.links
        assert "colab-upload-instructions" in links.ids
        counters["colab_upload_entries"] += 1
        visible = [cell for cell in nb["cells"] if cell["cell_type"] == "code"
                   and not cell.get("metadata", {}).get("course_setup")
                   and "course_figure" not in cell.get("metadata", {})]
        counters["visible_code_cells"] += len(visible)
        assert markup.count('class="language-python highlight"') >= len(visible)
        page_plain = plain(markup)
        for cell in visible:
            rendered_output = output_html(cell.get("outputs", []))
            if rendered_output:
                counters["visible_code_cells_with_output"] += 1
                # text/stream 輸出核關鍵原文；圖片輸出核 data URI。
                if 'src="data:image/png;base64,' in rendered_output:
                    output_present = re.search(r'src="([^"]+)"', rendered_output)[1] in markup
                else:
                    output_present = plain(rendered_output) in page_plain
                if output_present:
                    counters["output_fragments_found"] += 1
                else:
                    failures.append(item["id"] + ": executed output fragment missing")
        counters["lessons"] += 1
        rows.append({"id": item["id"], "notebook": item["notebook"],
                     "source_sha256": sha(original_path), "executed_sha256": sha(execution_path),
                     "html_sha256": sha(html_path), "visible_code_cells": len(visible)})
    home_html = (site / "index.html").read_text()
    parsed = Links()
    parsed.feed(home_html)
    assert all(link in parsed.links for link in ["1.1.html", "first-steps.html", "course.html"])
    assert plain(home.split("\n\n")[1]) in plain(home_html)
    info = json.loads((site / "build-info.json").read_bytes())
    assert info["executed_cpu_outputs"] is True
    assert info["colab_mode"] == "upload" and info["publication_ready"] is False
    # 既有工具另外核全導覽的存在、片段、下載內容、Colab 模式；不執行教材。
    check(site)
    cache = Path("/workspace/work/omni-technical-index-v1")
    detail_path = cache / "all-331-navigation-output-details.json"
    detail_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_index.py",
        "environment": {"python": sys.version.split()[0], "torch": importlib.metadata.version("torch"),
                        "zensical": importlib.metadata.version("zensical"), "device": "CPU; no model training"},
        "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path)},
        "source_snapshot": {"path": page["snapshot"], "sha256": sha(snapshot)},
        "current_generated_home_matches_snapshot": True,
        "figures": {}, "section_count": len(index), "numbered_heading_count": len(headings),
        "nav_all_331_exactly_once": True, "search_all_331_included": True,
        "home_three_links_present": True, "counts": counters, "build_info": info,
        "detailed_receipt_git_external": {"path": str(detail_path), "sha256": sha(detail_path)},
        "site_index_sha256": sha(site / "index.html"), "search_json_sha256": sha(search_path),
        "failures": failures,
        "scope": "全331份導覽、下載與已執行副本的版本和輸出呈現；沒有逐頁重審理論或重新訓練，沒有登入或執行Colab。",
    }
    result["normalization_note"] = (
        "首跑把HTML標籤刪成空字串，兩格無起始換行的輸出因此與CPU標籤黏連，產生15.8及19.6假陰性；"
        "本次以標籤邊界作空白後核對。首跑navigation-output-check.json原封保留。"
    )
    destination = Path(__file__).with_name("navigation-output-check-r2.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
