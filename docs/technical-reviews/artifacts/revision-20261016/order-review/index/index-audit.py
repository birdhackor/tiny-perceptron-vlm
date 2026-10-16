"""Index order callback evidence; inspect existing outputs, never execute kernels."""

import hashlib
import json
import platform
import re
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path.cwd().resolve()
import sys

sys.path.insert(0, str(ROOT))
from scripts.export_course import checked_notebook, home_introduction, output_html


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nav_paths(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from nav_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nav_paths(item)


def main():
    base = ROOT / "docs/technical-reviews/artifacts/revision-20261016/order-review/index"
    shots = Path("/workspace/work/course-order-20261016/index")
    preview = ROOT / "outputs/order-review-20261016-preview-01"
    prior_path = ROOT / "docs/course-revision-20261016/reviews/technical-index-final-assets-callback-01.json"
    prior = json.loads(prior_path.read_text())
    freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-18.json").read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == "index")
    body = (ROOT / page["snapshot"]).read_text()
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    routes = json.loads((ROOT / "course/reading-time-routes.json").read_text())["routes"]
    ids = [i["id"] for i in index]
    navigation = [p[:-3] for p in nav_paths(config["nav"]) if re.fullmatch(r"[A-Z\d]+\.\d+\.md", p)]
    html = (preview / "index.html").read_text()
    html_ids = list(dict.fromkeys(re.findall(r'href="(?:\./)?([A-Z\d]+\.\d+)\.html"', html)))
    known_pages = {p["page_id"] for p in freeze["pages"]}
    author_path = ROOT / "docs/technical-reviews/artifacts/revision-20261016/kernel-execution-order-01.json"
    author = json.loads(author_path.read_text())
    composed = Path(author["composed_directory"])
    failures = []
    outputs = 0
    code_cells = 0
    actions = 0
    revision = json.loads((preview / "build-info.json").read_text())["revision"]
    for item in index:
        source = ROOT / item["notebook"]
        result_path = composed / Path(item["notebook"]).relative_to("notebooks")
        notebook = checked_notebook(json.loads(source.read_text()), result_path)
        text = (preview / (item["id"] + ".html")).read_text()
        compact = re.sub(r">\s+<", "><", text)
        code_cells += sum(c["cell_type"] == "code" for c in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and not any(cell.get("metadata", {}).get(k) for k in ("course_setup", "course_figure")):
                rendered = output_html(cell.get("outputs", []))
                if rendered:
                    outputs += 1
                    if re.sub(r">\s+<", "><", rendered) not in compact:
                        failures.append({"id": item["id"], "error": "visible output missing"})
        colab = "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/" + revision + "/" + item["notebook"]
        if colab not in text or ('href="' + item["notebook"] + '" download') not in text:
            failures.append({"id": item["id"], "error": "action link missing"})
        elif sha(preview / item["notebook"]) != sha(source):
            failures.append({"id": item["id"], "error": "download differs from current notebook"})
        else:
            actions += 1
    checks = {
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "home_function_equals_freeze18": home_introduction(index, True) == body,
        "canonical_figure_hashes": page["figures_sha256"],
        "current_sources_sha256": {p: sha(ROOT / p) for p in ["scripts/export_course.py", "course/lesson-index.json", "zensical.toml", "course/reading-time-routes.json", "course/README.md"]},
        "previous_report_sha256": sha(prior_path),
        "prior_report_reused_artifacts_all_hashes_match": all(sha(ROOT / a["path"]) == a["sha256"] for a in prior["artifacts"]),
        "export_source_unchanged_since_own_previous_review": sha(ROOT / "scripts/export_course.py") == next(s["sha256"] for s in prior["sources"] if s["id"] == "s-export"),
        "index_count": len(ids), "unique_index_count": len(set(ids)),
        "config_count": len(navigation), "unique_config_count": len(set(navigation)),
        "preview_navigation_count": len(html_ids),
        "missing_from_config": sorted(set(ids) - set(navigation)),
        "missing_from_preview": sorted(set(ids) - set(html_ids)),
        "chapter14_orders": {"index": [p for p in ids if p.startswith("14.")], "config": [p for p in navigation if p.startswith("14.")], "preview": [p for p in html_ids if p.startswith("14.")]},
        "chapter19_orders": {"index": [p for p in ids if p.startswith("19.")], "config": [p for p in navigation if p.startswith("19.")], "preview": [p for p in html_ids if p.startswith("19.")]},
        "routes_unknown_pages": {r["id"]: sorted(set(r["page_ids"]) - known_pages) for r in routes},
        "capstone_route": next(r["page_ids"] for r in routes if r["id"] == "capstone"),
        "author_order_receipt_sha256": sha(author_path),
        "author_order_record_newly_executed": author["newly_executed"],
        "author_order_record_reused": author["unchanged_notebooks_reused"],
        "author19_12_source_hash_matches": sha(ROOT / "notebooks/19/19.12.ipynb") == author["source_sha256"],
        "author19_12_executed_hash_matches": sha(author["executed_path"]) == author["executed_sha256"],
        "current_existing_notebooks_source_and_execution_fields_checked": len(index),
        "code_cells_with_execution_count": code_cells,
        "current_preview_nonempty_visible_outputs_checked": outputs,
        "current_preview_colab_and_matching_downloads": actions,
        "failures": failures,
        "preview_build_info": json.loads((preview / "build-info.json").read_text()),
        "preview_home_sha256": sha(preview / "index.html"),
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        tab = browser.new_page(viewport={"width": 1280, "height": 900})
        tab.set_default_timeout(10000)
        start = "http://127.0.0.1:8897/index.html"
        entrances = []
        for label, target in [("從第一節開始", "1.1.html"), ("先看基礎暖身", "first-steps.html"), ("閱讀指南", "course.html")]:
            tab.goto(start, wait_until="domcontentloaded")
            served = tab.content()
            tab.locator("article").get_by_role("link", name=label, exact=True).click()
            tab.wait_for_url("**/" + target)
            entrances.append({"label": label, "expected": target, "actual": tab.url.split("/")[-1], "heading": tab.locator("article h1").inner_text()})
        tab.goto(start, wait_until="domcontentloaded")
        tab.locator(".md-search__button").click()
        searches = []
        for query in ["可還原", "去除重複字"]:
            tab.locator("input[role=combobox]").fill(query)
            tab.wait_for_timeout(900)
            links = tab.locator("a").evaluate_all("els=>els.filter(e=>e.getRootNode() instanceof ShadowRoot).map(e=>({text:e.innerText,href:e.getAttribute('href')}))")
            hits = [r for r in links if "/1.1.html" in r["href"]]
            searches.append({"query": query, "hit_1_1": bool(hits), "results": len(links)})
        tab.screenshot(path=str(shots / "search-body.png"))
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
        mobile.goto(start, wait_until="domcontentloaded")
        menu = mobile.locator('label[for="__drawer"]').filter(has=mobile.locator("svg")).first
        rect = menu.bounding_box()
        mobile.screenshot(path=str(shots / "mobile-before.png"))
        menu.click()
        mobile.wait_for_timeout(250)
        checks["browser"] = {"version": browser.version, "home_entrances": entrances, "searches": searches, "mobile_viewport": "390x844", "menu_rect": rect, "drawer_opened": mobile.locator("#__drawer").is_checked()}
        mobile.screenshot(path=str(shots / "mobile-menu-open.png"))
        browser.close()
    checks["screenshots_outside_git"] = [{"path": str(p), "sha256": sha(p)} for p in sorted(shots.glob("*.png"))]
    result = {
        "reviewer_task": "/root/tech_20261016_index", "current_context": "informed", "recorded_at": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/index/index-audit.py",
        "environment": {"python": platform.python_version(), "torch": version("torch"), "playwright": version("playwright"), "device": "CPU; /usr/bin/chromium browser", "platform": platform.platform()},
        "checks": checks,
        "limits": "Own checks only parse current files and inspect author-generated preview/output artifacts. Did not build a site, run notebook kernels, train models, execute formal tests, download HF assets, or validate reading-time estimates. Previous personal evidence is explicitly inherited; no new fresh-context claim.",
    }
    (base / "index-execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checks": {k: v for k, v in checks.items() if k not in {"screenshots_outside_git", "chapter14_orders", "chapter19_orders"}}, "environment": result["environment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
