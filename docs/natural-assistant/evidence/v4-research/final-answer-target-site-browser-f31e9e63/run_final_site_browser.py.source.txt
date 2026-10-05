"""Final built-site DOM/layout/browser checks; never builds or edits the course."""

import argparse
import hashlib
import json
import re
import sys
import threading
import tomllib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import export_course, reading_time  # noqa: E402

V4 = [f"20.{i}" for i in range(1, 14)]
GUIDES = ["natural-v4-student", "natural-v4-data", "natural-v4-training"]
ENTRIES = [
    "environment",
    "asset-storage",
    "validation",
    "publishing",
    "index",
    "course",
    "curriculum",
    "first-steps",
    "training",
    "chapter-20",
    "19.12",
    "11.14",
    "11.15",
    "11.16",
    "12.13",
    "12.14",
    "1.1",
    "2.5",
    "7.18",
    "14.2",
    "17.14",
    "16.1",
]
NEW_FIGURES = {
    "natural-v4-training-cat.svg",
    "natural-v4-family-crops.svg",
    "natural-v4-answer-mask.svg",
    "natural-v4-asr-two-routes.svg",
    "curriculum_learning_flow.svg",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}


def nav_pages(value):
    result = []
    for row in value:
        for item in row.values():
            result.extend(nav_pages(item) if isinstance(item, list) else [item.removesuffix(".md")])
    return result


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence directory, preserving previous real browser runs")
    site = args.site.resolve()
    build = json.loads((site / "build-info.json").read_text())
    if build.get("builder") != "zensical" or build.get("reading_time_complete") is not True:
        parser.error("Require a final Zensical build with complete source-bound reading-time metadata")
    if args.expected_revision and build.get("revision") != args.expected_revision:
        parser.error("Built-site source revision differs from expected actual build revision")
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    lessons = {item["id"]: item for item in index}
    if not set(V4) <= set(lessons):
        parser.error("Final regenerated lesson-index must contain all 20.1–20.13")
    inventory = reading_time.build_inventory(
        ROOT,
        index,
        export_course.DOCUMENTS,
        export_course.home_introduction(index, bool(build.get("executed_cpu_outputs"))),
    )
    estimates = reading_time.load_estimates(ROOT / "course/reading-times.json", inventory, require_complete=True)
    titles = {item["page_id"]: item["title"] for item in inventory["pages"]}
    routes = reading_time.load_routes(ROOT / "course/reading-time-routes.json", inventory)
    natural_route = next(row for row in routes if row["id"] == "natural-assistant")
    if not set(V4 + GUIDES) <= set(natural_route["page_ids"]):
        parser.error("Final natural reading-time route must include 13 lessons and 3 guides")
    settings = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    order = nav_pages(settings["nav"])
    pages = list(dict.fromkeys(ENTRIES + V4 + GUIDES))
    for page_id in pages:
        if not (site / (page_id + ".html")).is_file():
            parser.error("Final built-site page is missing: " + page_id)
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    canonical = [
        ROOT / path
        for path in {
            "zensical.toml",
            "scripts/export_course.py",
            "scripts/reading_time.py",
            "course/web/course.css",
            "course/web/course.js",
            "course/reading-times.json",
            "course/reading-time-routes.json",
            "course/lesson-index.json",
            "course/chapters/20.md",
            *[export_course.DOCUMENTS[x] for x in GUIDES + ["environment", "asset-storage", "validation", "publishing"]],
        }
    ]
    canonical += sorted((ROOT / "course/figures").glob("natural-v4*.svg"))
    canonical += [
        ROOT / source for source in sorted({item["source"] for item in index}) if ROOT / source not in canonical
    ]
    before = [record(path) for path in sorted(canonical)]
    report = {
        "schema_version": 1,
        "scope": "Actual final local built-site browser/DOM/layout trial; not model UI or model capability",
        "build_info": build,
        "build_info_sha256": sha(site / "build-info.json"),
        "viewports": [1440, 358],
        "sources_before": before,
        "rows": [],
        "failures": [],
        "outside_source_links": [],
        "new_figures_seen": [],
    }
    for path in (Path(__file__), HERE / "svg_metrics.js"):
        (output / path.name).write_bytes(path.read_bytes())
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    svg_js = (HERE / "svg_metrics.js").read_text()
    target_cache = {}

    def check_target(url, context, source):
        parsed = urlsplit(url)
        if (parsed.scheme, parsed.netloc) != ("http", urlsplit(base).netloc):
            return {"url": url, "external": True}
        path = (site / unquote(parsed.path).lstrip("/")).resolve()
        if not path.is_relative_to(site) or not path.is_file():
            raise AssertionError("Local reference missing/escapes site: " + url)
        if parsed.fragment and path.suffix == ".html":
            if path not in target_cache:
                other = context.new_page()
                try:
                    response = other.goto(base + path.relative_to(site).as_posix(), wait_until="domcontentloaded")
                    if response.status != 200:
                        raise AssertionError("Reference page failed: " + url)
                    target_cache[path] = set(other.locator("[id]").evaluate_all("nodes => nodes.map(n=>n.id)"))
                finally:
                    other.close()
            if unquote(parsed.fragment) not in target_cache[path]:
                raise AssertionError("Local anchor missing: " + url)
        return {"url": url, "external": False, "file_sha256": sha(path), "source_link_text": source}

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"]
            )
            report["browser_version"] = browser.version
            for width in (1440, 358):
                context = browser.new_context(
                    viewport={"width": width, "height": 1000}, device_scale_factor=1, reduced_motion="reduce"
                )
                page = context.new_page()
                page.set_default_timeout(args.timeout_ms)
                discovered = []
                for page_id in pages:
                    row = {
                        "page": page_id,
                        "viewport_width": width,
                        "figures": [],
                        "local_links": [],
                        "prerequisites": [],
                    }
                    try:
                        response = page.goto(base + page_id + ".html", wait_until="domcontentloaded")
                        row["status"] = response.status
                        if response.status != 200:
                            raise AssertionError("Page is not HTTP 200")
                        page.wait_for_function(
                            "Array.from(document.querySelectorAll('.md-content img')).every(i=>i.complete && i.naturalWidth>0)"
                        )
                        page.evaluate(
                            "async()=>{if(window.MathJax?.startup?.promise) await MathJax.startup.promise; await document.fonts.ready}"
                        )
                        row["heading"] = page.locator(".md-content h1").first.inner_text()
                        row["reading_time"] = page.locator(".md-content .reading-time").all_text_contents()
                        estimate = estimates[page_id]
                        expected = f"{estimate['minutes_min']}–{estimate['minutes_max']}"
                        if (
                            len(row["reading_time"]) != 1
                            or expected not in row["reading_time"][0]
                            or "AI" not in row["reading_time"][0]
                        ):
                            raise AssertionError("Reading-time badge missing/stale/duplicated")
                        row["reading_time_source_sha256"] = estimate["source_sha256"]
                        row["overflow"] = page.evaluate(
                            "({outer:document.documentElement.scrollWidth>innerWidth+1, body:document.body.scrollWidth>innerWidth+1, article:Array.from(document.querySelectorAll('.md-content')).map(n=>({client:n.clientWidth,scroll:n.scrollWidth}))})"
                        )
                        if (
                            row["overflow"]["outer"]
                            or row["overflow"]["body"]
                            or any(item["scroll"] > item["client"] + 1 for item in row["overflow"]["article"])
                        ):
                            raise AssertionError("Horizontal page overflow")
                        if page_id in lessons:
                            if (
                                not row["heading"].startswith(page_id + " ")
                                or lessons[page_id]["title"] not in row["heading"]
                            ):
                                raise AssertionError("Lesson title differs from final lesson-index")
                            download = page.locator(".md-content .actions a[download]")
                            colab = page.locator(".md-content .actions a[href^='https://colab.research.google.com/']")
                            if download.count() != 1 or colab.count() != 1:
                                raise AssertionError("Notebook/Colab action missing or duplicated")
                            href = urljoin(page.url, download.get_attribute("href"))
                            row["notebook"] = check_target(href, context, "download notebook")
                            served = context.request.get(href)
                            row["notebook"]["HTTP_bytes"] = len(served.body())
                            row["notebook"]["HTTP_sha256"] = hashlib.sha256(served.body()).hexdigest()
                            notebook = served.json()
                            if served.status != 200 or notebook.get("nbformat") != 4 or not notebook.get("cells"):
                                raise AssertionError("Downloaded notebook is not a real nbformat4 file")
                            if (
                                row["notebook"]["file_sha256"] != sha(ROOT / lessons[page_id]["notebook"])
                                or row["notebook"]["HTTP_sha256"] != row["notebook"]["file_sha256"]
                            ):
                                raise AssertionError("Built notebook bytes differ from current canonical notebook")
                            with page.expect_download() as pending:
                                download.click()
                            delivered = pending.value
                            if delivered.failure() is not None:
                                raise AssertionError("Actual browser notebook download failed")
                            downloaded = output / f"{page_id}-{width}-download.ipynb"
                            delivered.save_as(str(downloaded))
                            row["actual_browser_notebook_download"] = record(downloaded)
                            if sha(downloaded) != row["notebook"]["file_sha256"]:
                                raise AssertionError("Browser-downloaded notebook bytes differ from exact source")
                            row["colab"] = colab.get_attribute("href")
                            if "/blob/" + build["revision"] + "/" + lessons[page_id]["notebook"] not in row["colab"]:
                                raise AssertionError("Colab revision/notebook differs from actual build info")
                        if page_id in order:
                            position = order.index(page_id)
                            for label, offset in (("prev", -1), ("next", 1)):
                                link = page.locator(".md-footer__link--" + label)
                                expected_neighbor = (
                                    order[position + offset] if 0 <= position + offset < len(order) else None
                                )
                                if expected_neighbor is not None:
                                    if (
                                        link.count() != 1
                                        or Path(urlsplit(urljoin(page.url, link.get_attribute("href"))).path).stem
                                        != expected_neighbor
                                    ):
                                        raise AssertionError("Footer " + label + " does not follow final nav")
                                    row[label] = check_target(
                                        urljoin(page.url, link.get_attribute("href")), context, label
                                    )
                        for link in page.locator(".md-content a[href]").all():
                            href = urljoin(page.url, link.get_attribute("href"))
                            if urlsplit(href).scheme not in ("http", "https"):
                                continue
                            item = check_target(href, context, link.inner_text())
                            if item.get("external"):
                                if re.search(r"\.(?:jsonl?|py)(?:$|[?#])", href):
                                    report["outside_source_links"].append({"page": page_id, **item})
                            else:
                                row["local_links"].append(item)
                        for paragraph in page.locator(".md-content p").all():
                            if "前置" in paragraph.inner_text():
                                for link in paragraph.locator("a[href]").all():
                                    href = urljoin(page.url, link.get_attribute("href"))
                                    item = check_target(href, context, link.inner_text())
                                    row["prerequisites"].append(item)
                                    if not item.get("external") and urlsplit(href).path.endswith(".html"):
                                        found = Path(urlsplit(href).path).stem
                                        discovered.append(found)
                                        if width == 1440 and page_id in V4 + GUIDES and found not in pages:
                                            pages.append(found)
                        target = output / f"{page_id}-{width}-full.png"
                        page.screenshot(path=str(target), full_page=True)
                        row["full_screenshot"] = record(target)
                        row["built_html"] = record(site / (page_id + ".html"))
                        figures = page.locator(".md-content img.diagram")
                        for number in range(figures.count()):
                            image = figures.nth(number)
                            figure = image.locator("xpath=ancestor::figure[1]")
                            src = urljoin(page.url, image.get_attribute("src"))
                            name = Path(urlsplit(src).path).name
                            native = (site / unquote(urlsplit(src).path).lstrip("/")).resolve()
                            details = image.evaluate(
                                "n=>({loaded:n.complete&&n.naturalWidth>0,natural_width:n.naturalWidth,natural_height:n.naturalHeight,alt:n.alt,rect:{width:n.getBoundingClientRect().width,height:n.getBoundingClientRect().height}})"
                            )
                            if not details["loaded"] or not details["alt"]:
                                raise AssertionError("Diagram incomplete/missing alt")
                            measurements = page.evaluate(
                                svg_js,
                                {
                                    "url": src,
                                    "renderedWidth": details["rect"]["width"],
                                    "renderedHeight": details["rect"]["height"],
                                },
                            )
                            if measurements["text_outside_viewbox"] or measurements["image_outside_viewbox"]:
                                raise AssertionError("SVG text/raster image is outside its viewBox")
                            artifact = output / f"{page_id}-{width}-figure{number}.png"
                            page.add_style_tag(content="header.md-header{visibility:hidden!important}")
                            image.screenshot(path=str(artifact))
                            item = {
                                "name": name,
                                "source": record(native),
                                "normal": details,
                                "normal_font_metrics": measurements,
                                "screenshot": record(artifact),
                                "capture_note": "Sticky header hidden only after unmodified page checks/full-page screenshot; no content/diagram/font edits",
                            }
                            if name in NEW_FIGURES:
                                report["new_figures_seen"].append(name)
                                if sha(native) != sha(ROOT / "course/figures" / name):
                                    raise AssertionError("Built new SVG bytes differ from current canonical source")
                                button = figure.locator(".diagram-zoom")
                                if not button.is_visible():
                                    raise AssertionError("Diagram enlarge control did not initialize")
                                button.click()
                                page.wait_for_function(
                                    "selector=>document.querySelector(selector).getAttribute('aria-expanded')==='true'",
                                    arg="[aria-controls='" + button.get_attribute("aria-controls") + "']",
                                )
                                box = image.bounding_box()
                                item["enlarged_font_metrics"] = page.evaluate(
                                    svg_js, {"url": src, "renderedWidth": box["width"], "renderedHeight": box["height"]}
                                )
                                viewport = figure.locator(".diagram-viewport")
                                item["enlarged_scroll"] = viewport.evaluate(
                                    "n=>({client:n.clientWidth,scroll:n.scrollWidth,body_overflow:document.documentElement.scrollWidth>innerWidth+1})"
                                )
                                if box["width"] < 799 or item["enlarged_scroll"]["body_overflow"]:
                                    raise AssertionError(
                                        "Enlarge does not create readable image width / causes page overflow"
                                    )
                                if width == 358:
                                    viewport.evaluate("n=>n.scrollLeft=n.scrollWidth")
                                    item["scroll_right"] = viewport.evaluate(
                                        "n=>({left:n.scrollLeft,max:n.scrollWidth-n.clientWidth})"
                                    )
                                    if item["scroll_right"]["left"] < item["scroll_right"]["max"] - 1:
                                        raise AssertionError("Cannot scroll to full diagram right edge")
                                    screenshot = output / f"{page_id}-{width}-figure{number}-enlarged-right.png"
                                    viewport.screenshot(path=str(screenshot))
                                    item["enlarged_right_screenshot"] = record(screenshot)
                                    viewport.evaluate("n=>n.scrollLeft=(n.scrollWidth-n.clientWidth)/2")
                                    screenshot = output / f"{page_id}-{width}-figure{number}-enlarged-middle.png"
                                    viewport.screenshot(path=str(screenshot))
                                    item["enlarged_middle_screenshot"] = record(screenshot)
                                    viewport.evaluate("n=>n.scrollLeft=0")
                                    screenshot = output / f"{page_id}-{width}-figure{number}-enlarged-left.png"
                                    viewport.screenshot(path=str(screenshot))
                                    item["enlarged_left_screenshot"] = record(screenshot)
                                button.click()
                                if button.get_attribute("aria-expanded") != "false":
                                    raise AssertionError("Diagram restore control failed")
                            row["figures"].append(item)
                        row["actual_footer_navigation"] = []
                        for label in ("prev", "next"):
                            if label not in row:
                                continue
                            target_url = row[label]["url"]
                            target_id = Path(urlsplit(target_url).path).stem
                            page.locator(".md-footer__link--" + label).click()
                            page.wait_for_url(target_url)
                            page.wait_for_function(
                                "title=>document.querySelector('.md-content h1')?.textContent.includes(title)",
                                arg=titles[target_id],
                            )
                            row["actual_footer_navigation"].append(
                                {
                                    "direction": label,
                                    "url": page.url,
                                    "heading": page.locator(".md-content h1").first.inner_text(),
                                }
                            )
                            page.goto(base + page_id + ".html", wait_until="domcontentloaded")
                            page.wait_for_function(
                                "Array.from(document.querySelectorAll('.md-content img')).every(i=>i.complete&&i.naturalWidth>0)"
                            )
                        if page_id == "16.1":
                            method = page.get_by_role("link", name="T.8的效率量測方法", exact=True)
                            href = urljoin(page.url, method.get_attribute("href"))
                            target_id = "選讀效率實驗的量測條件"
                            if unquote(urlsplit(href).fragment) != target_id:
                                raise AssertionError("Efficiency method link does not target its descriptive section")
                            method.click()
                            page.wait_for_load_state("domcontentloaded")
                            page.wait_for_function(
                                "id=>{const n=document.getElementById(id);if(!n)return false;const y=n.getBoundingClientRect().top;return y>=0&&y<innerHeight}",
                                arg=target_id,
                            )
                            target = page.locator("#" + target_id)
                            row["actual_efficiency_method_navigation"] = {
                                "url": page.url,
                                "target_id": target_id,
                                "heading": target.inner_text(),
                                "bounding_box": target.bounding_box(),
                            }
                            image_path = output / f"16.1-{width}-efficiency-method-target.png"
                            page.screenshot(path=str(image_path))
                            row["actual_efficiency_method_navigation"]["screenshot"] = record(image_path)
                        row["status_result"] = "passed"
                    except Exception as error:
                        row.update(status_result="failed", error={"type": type(error).__name__, "message": str(error)})
                        report["failures"].append({"page": page_id, "width": width, **row["error"]})
                    report["rows"].append(row)
                    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    print(
                        json.dumps(
                            {"page": page_id, "width": width, "status": row["status_result"]}, ensure_ascii=False
                        ),
                        flush=True,
                    )
                if width == 1440:
                    pages.extend(x for x in dict.fromkeys(discovered) if x not in pages)
                context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        report["sources_after"] = [record(path) for path in sorted(canonical)]
        report["source_bytes_unchanged"] = report["sources_before"] == report["sources_after"]
        missing = sorted(NEW_FIGURES - set(report["new_figures_seen"]))
        if missing:
            report["failures"].append({"missing_new_figures": missing})
        if not report["source_bytes_unchanged"]:
            report["failures"].append({"mutable_sources": True})
        report["status"] = "failed" if report["failures"] else "passed"
        report["visual_review_scope"] = (
            "Actual browser screenshots and DOM/font measurements; human/independent screenshot reading must still be recorded before final layout conclusion"
        )
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["failures"]:
        raise SystemExit("Actual browser/layout checks failed; inspect preserved report")
    print(json.dumps({"status": "passed", "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
