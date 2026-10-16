"""Independent index-only audit: inspect existing kernel artifacts; do not rerun notebooks."""

import hashlib
import json
import platform
import re
import tomllib
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.export_course import checked_notebook, home_introduction, output_html


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from playwright.sync_api import sync_playwright

    preview = ROOT / "outputs/revision-20261016-executed-preview-09-math-safe"
    evidence = ROOT / "docs/technical-reviews/artifacts/revision-20261016"
    screenshots = Path("/workspace/work/curriculum-20261016/index-final-assets-callback-01")
    screenshots.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-09.json").read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "index")
    snapshot = ROOT / page["snapshot"]
    body = snapshot.read_text()
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    receipt_path = evidence / "kernel-execution-09.json"
    receipt = json.loads(receipt_path.read_text())
    records = {r["id"]: {"executed_local_path": r["output_path"], "notebook_sha256": r["input_sha256"], "executed_sha256": r["output_sha256"]} for r in receipt["composition"]["rows"]}
    html = (preview / "index.html").read_text()
    ids = {item["id"] for item in index}
    nav_ids = set(re.findall(r'href="(?:\./)?([A-Z\d]+\.\d+)\.html"', html))
    checks = {
        "snapshot_sha256": sha(snapshot),
        "snapshot_matches_freeze": sha(snapshot) == page["source_sha256"],
        "current_home_function_matches_snapshot": home_introduction(index, True) == body,
        "canonical_scope": "home_introduction(index, True) only; generated chapter list and reading-time UI excluded",
        "lesson_index_count": len(index),
        "unique_lesson_ids": len(ids),
        "homepage_navigation_lesson_count": len(ids & nav_ids),
        "missing_navigation_ids": sorted(ids - nav_ids),
        "author_receipt_sha256": sha(receipt_path),
        "author_actor": receipt["author_task"],
        "author_execution_not_repeated": True,
    }
    settings = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    def nav_paths(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from nav_paths(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from nav_paths(item)
    nav_items = [p for p in nav_paths(settings["nav"]) if re.fullmatch(r"[A-Z\d]+\.\d+\.md", p)]
    from scripts.stage_mathjax import validate_manifest, read_asset
    manifest_path = ROOT / "course/web/mathjax-assets.json"
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    for item in manifest["files"]:
        read_asset(preview / "javascripts/mathjax" / item["path"], item)
    checks.update({
        "current_sources_sha256": {p: sha(ROOT / p) for p in ["scripts/export_course.py", "zensical.toml", "course/lesson-index.json", "scripts/stage_mathjax.py", "course/web/mathjax-assets.json"]},
        "current_config_lesson_count": len(nav_items),
        "current_config_unique_lesson_count": len(set(nav_items)),
        "current_config_missing_lesson_ids": sorted({i["id"] + ".md" for i in index} - set(nav_items)),
        "current_extra_javascript": settings["extra_javascript"],
        "local_mathjax_files_verified_against_manifest": len(manifest["files"]),
        "original_report_sha256": sha(ROOT / "docs/course-revision-20261016/reviews/technical-index-original.json"),
        "informed_callback_scope": "Original reviewer returning after author disclosed C navigation grouping and bundled MathJax changes; not a new fresh independent review. Reread current home/output/footer/build/config and MathJax staging code. Browser and output checks inspect author-built safe preview; no site build or kernel execution was performed by reviewer.",
    })
    failures = []
    visible_outputs = 0
    checked_code = 0
    footer_count = 0
    for item in index:
        ident = item["id"]
        record = records[ident]
        source = ROOT / item["notebook"]
        executed = ROOT / record["executed_local_path"]
        if sha(source) != record["notebook_sha256"] or sha(executed) != record["executed_sha256"]:
            failures.append({"id": ident, "problem": "author receipt file hash mismatch"})
        notebook = checked_notebook(json.loads(source.read_text()), executed)
        lesson_html = (preview / (ident + ".html")).read_text()
        compact_html = re.sub(r">\s+<", "><", lesson_html)
        checked_code += sum(cell["cell_type"] == "code" for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and not any(cell.get("metadata", {}).get(key) for key in ("course_setup", "course_figure")):
                rendered = output_html(cell.get("outputs", []))
                if rendered:
                    visible_outputs += 1
                    if re.sub(r">\s+<", "><", rendered) not in compact_html:
                        failures.append({"id": ident, "problem": "executed output absent from preview"})
        revision = json.loads((preview / "build-info.json").read_text())["revision"]
        colab = "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/" + revision + "/" + item["notebook"]
        download = preview / item["notebook"]
        if colab not in lesson_html or 'href="' + item["notebook"] + '" download' not in lesson_html:
            failures.append({"id": ident, "problem": "footer action link missing"})
        elif not download.is_file() or sha(download) != sha(source):
            failures.append({"id": ident, "problem": "Notebook download differs from current source"})
        else:
            footer_count += 1
    checks.update({
        "author_notebooks_rechecked_for_current_sources_hashes_and_no_errors": len(index),
        "code_cells_with_execution_counts": checked_code,
        "nonempty_visible_output_blocks_checked": visible_outputs,
        "export_semantics": "course_setup cells are hidden; course_figure cells become SVG figures rather than their IPython display-object repr. Navigation hrefs may start with ./. Zensical may add whitespace between HTML tags, normalized only at tag boundaries; pre text remains compared.",
        "lesson_pages_with_colab_and_matching_notebook_download": footer_count,
        "artifact_failures": failures,
        "preview_build_info": json.loads((preview / "build-info.json").read_text()),
        "preview_index_sha256": sha(preview / "index.html"),
        "preview_search_sha256": sha(preview / "search.json"),
    })

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(preview)))
    Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        tab = browser.new_page(viewport={"width": 1280, "height": 900})
        tab.set_default_timeout(10000)
        tab.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="domcontentloaded")
        tab.locator(".md-search__button").click()
        searches = []
        for query, role in [("可還原", "section title"), ("去除重複字", "body phrase absent from 1.1 title")]:
            tab.locator("input[role=combobox]").fill(query)
            tab.wait_for_timeout(1200)
            links = tab.locator("a").evaluate_all("(els)=>els.filter(e=>e.getRootNode() instanceof ShadowRoot).map(e=>({text:e.innerText,href:e.getAttribute('href')}))")
            hit = next((link for link in links if "/1.1.html" in link["href"]), None)
            searches.append({"query": query, "query_role": role, "expected": "1.1.html result", "observed": {"href": hit["href"].split("/")[-1], "title": hit["text"].splitlines()[2], "body_phrase_present": query in hit["text"]} if hit else None, "result_count": len(links)})
        tab.screenshot(path=str(screenshots / "search-body.png"))
        tab.close()
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, device_scale_factor=1)
        mobile.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="domcontentloaded")
        menu = mobile.locator('label[for="__drawer"]').filter(has=mobile.locator("svg")).first
        before = menu.bounding_box()
        mobile.screenshot(path=str(screenshots / "mobile-before.png"))
        menu.click()
        mobile.wait_for_timeout(300)
        drawer_checked = mobile.locator("#__drawer").is_checked()
        mobile.screenshot(path=str(screenshots / "mobile-menu-open.png"))
        checks["browser"] = {"chromium": browser.version, "desktop_viewport": "1280x900", "searches": searches, "mobile_viewport": "390x844", "mobile_menu_rect": before, "mobile_menu_opens_drawer": drawer_checked, "external_mathjax_blocked": "none; current preview uses local javascripts/mathjax/tex-chtml.js"}
        browser.close()
    server.shutdown()
    checks["screenshots_outside_git"] = [{"path": str(p), "sha256": sha(p)} for p in sorted(screenshots.glob("*.png")) if p.name in {"search-body.png", "mobile-before.png", "mobile-menu-open.png"}]
    result = {
        "reviewer_task": "/root/tech_20261016_index",
        "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/index-final-assets-callback-01-audit.py",
        "environment": {"python": platform.python_version(), "torch": version("torch"), "device": "CPU; existing preview inspected in headless /usr/bin/chromium", "playwright": version("playwright"), "zensical": version("zensical"), "platform": platform.platform()},
        "checks": checks,
        "limits": "Reviewer inspected existing author-generated outputs and current preview; did not run notebook kernels, training, remote Colab, or deployment. The author built this local preview; reviewer only inspected it. Preview revision does not prove remote ref availability. Homepage narrative previews later teaching; no learned quality claim is asserted here.",
    }
    target = evidence / "index-final-assets-callback-01-execution.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
