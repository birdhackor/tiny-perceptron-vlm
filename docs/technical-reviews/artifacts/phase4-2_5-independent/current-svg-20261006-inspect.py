"""Bounded current SVG/browser inspection; reuse verified original CPU evidence."""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent / "current-svg-20261006"
OUT.mkdir(exist_ok=True)
ROOT = OUT.parents[4]
BASE = OUT.relative_to(ROOT).as_posix()
PRIOR_SHA = "5fe107886db31da1747c42cd0dcc45e8329ada3f8b2091393c08818c6f6fb05d"
ARCHIVE = "docs/technical-reviews/history/phase4-2_5-own-before-current-svg-" + PRIOR_SHA + ".json"
FIGURE = "course/figures/rewrite-02-visible-window.svg"
EXPECTED_FIGURE = "82d9a43f009b9f7dd394094fa71addfdd715f5e48733276f0f01d05ef1148054"
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def identify(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": digest(raw), "bytes": len(raw)}
def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
raw_report = (ROOT / "docs/technical-reviews/2.5.json").read_bytes()
assert digest(raw_report) == PRIOR_SHA
assert digest((ROOT / ARCHIVE).read_bytes()) == PRIOR_SHA
report = json.loads(raw_report)
spec = importlib.util.spec_from_file_location("facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
section, chapter, first = facts.original_section(ROOT / "course/chapters/02.md", "2.5")
assert digest(section) == report["source_sha256"] == "db68ba26722abbe479298b1aa35ba583b6c9f3b5040c322afe903b866b34b478"
(OUT / "section.md").write_bytes(section)
current = (ROOT / FIGURE).read_bytes()
assert digest(current) == EXPECTED_FIGURE
(OUT / "current-window.svg").write_bytes(current)
prior = (OUT.parent / "figures/rewrite-02-visible-window.svg").read_bytes()
assert digest(prior) == report["figure_sha256"][FIGURE]
ns = {"s": "http://www.w3.org/2000/svg"}
old, new = ET.fromstring(prior), ET.fromstring(current)
old_text = old.findall(".//s:text", ns)
new_text = new.findall(".//s:text", ns)
assert len(old_text) == len(new_text)
changed = [{"index": i, "old": {"text": a.text, "attributes": a.attrib},
            "current": {"text": b.text, "attributes": b.attrib}}
           for i, (a,b) in enumerate(zip(old_text, new_text)) if ET.tostring(a) != ET.tostring(b)]
assert [item["current"]["text"] for item in changed] == ["答案：圓", "前文", "答案：方", "前文"]
assert all(ET.tostring(a) == ET.tostring(b) for a,b in zip(old.findall(".//s:rect",ns),new.findall(".//s:rect",ns)))
assert old.attrib == new.attrib
boxes = [item.attrib for item in new.findall(".//s:rect", ns) if item.attrib.get("fill") == "#e7f1ff"]
assert [int(item["width"]) for item in boxes] == [88,264,440,88,264,440]
unchanged_artifacts = []
for artifact in report["artifacts"]:
    item = identify(artifact["path"])
    assert item["sha256"] == artifact["sha256"], artifact["id"]
    unchanged_artifacts.append({"id": artifact["id"], **item})
unchanged_sources = []
for source in report["sources"]:
    if source["kind"] == "repository_code":
        item = identify(source["path"])
        assert item["sha256"] == source["sha256"], source["id"]
        unchanged_sources.append({"id": source["id"], **item})
necessary_code = []
for name in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/train_simple.py", "scripts/course_experiments/text.py"):
    current_code = (ROOT/name).read_bytes()
    frozen = (OUT.parent/"current-code"/name).read_bytes()
    assert current_code == frozen
    necessary_code.append(identify(name))
assert digest((ROOT / "course/figures/window_training.svg").read_bytes()) == report["figure_sha256"]["course/figures/window_training.svg"]
probe = json.loads((OUT.parent/"probe-result.json").read_bytes())
assert probe["windows"] == {"1":["是","是"],"3":["物體是","物體是"],"4":["色物體是","色物體是"],"5":["紅色物體是","藍色物體是"]}
assert [probe["denominators"][s]["effective_next_character_targets"] for s in ("train","validation","test")] == [103,11,22]
model_values = {name: {s: values["metrics"][s] for s in ("train","validation","test")}
                for name, values in probe["evaluations"].items()}
render_command = ["inkscape", str(OUT/"current-window.svg"), "--export-type=png", "--export-filename="+str(OUT/"current-window.png")]
with (OUT/"inkscape-stdout.txt").open("wb") as stdout, (OUT/"inkscape-stderr.txt").open("wb") as stderr:
    render = subprocess.run(render_command, stdout=stdout, stderr=stderr, timeout=30, check=False)
assert render.returncode == 0
version = subprocess.run(["inkscape","--version"], capture_output=True, text=True, check=True).stdout.strip()
browser_receipt = {"page_url":"http://127.0.0.1:8765/2.5.html", "model_execution":False}
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium",headless=True,
            args=["--no-sandbox","--disable-dev-shm-usage"], timeout=20000)
        page = browser.new_page(viewport={"width":1024,"height":1000}, device_scale_factor=1)
        response = page.goto(browser_receipt["page_url"],wait_until="domcontentloaded",timeout=20000)
        browser_receipt["page_http_status"] = response.status
        browser_receipt["browser_version"] = browser.version
        browser_receipt["playwright_version"] = importlib.metadata.version("playwright")
        (OUT/"served-page.html").write_text(page.content())
        figure = page.locator('img[src*="rewrite-02-visible-window.svg"]')
        assert figure.count() == 1, figure.count()
        figure.scroll_into_view_if_needed(timeout=10000)
        figure.wait_for(state="visible",timeout=10000)
        assert figure.evaluate("img => img.complete && img.naturalWidth > 0")
        source_url = urllib.parse.urljoin(browser_receipt["page_url"], figure.get_attribute("src"))
        with urllib.request.urlopen(source_url,timeout=10) as response:
            served = response.read()
            browser_receipt["svg_http_status"] = response.status
        assert digest(served) == EXPECTED_FIGURE
        (OUT/"served-window.svg").write_bytes(served)
        browser_receipt.update(svg_url=source_url, svg_sha256=digest(served),
            image_geometry=figure.evaluate("img => ({naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,displayWidth:img.getBoundingClientRect().width,displayHeight:img.getBoundingClientRect().height})"))
        figure.screenshot(path=str(OUT/"browser-current-window.png"), timeout=15000)
        browser_receipt["screenshot_created"] = True
        browser.close()
except Exception as error:
    browser_receipt.update(screenshot_created=False, tool_limitation={"type":type(error).__name__,"message":str(error)})
receipt = {
    "kind":"current_svg_actual_reinspection_preparation",
    "reviewer_task":report["reviewer_task"], "accessed_on":"2026-10-06",
    "prior_report_history":{"path":ARCHIVE,"sha256":PRIOR_SHA},
    "source": "course/chapters/02.md#2.5", "source_sha256":digest(section), "section_first_line":first,
    "current_figure":identify(FIGURE), "prior_figure_sha256":digest(prior),
    "actual_svg_text_changes":changed, "unchanged_window_boxes":boxes,
    "unchanged_registered_artifacts":unchanged_artifacts,
    "unchanged_repository_sources":unchanged_sources,
    "unchanged_current_necessary_code":necessary_code,
    "unchanged_original_probe_numeric_scope":{"windows":probe["windows"],"denominators":probe["denominators"],"model_values":model_values},
    "reused_original_support": "Full current section read; original JMLR section2 pp1141–1143 and official CPython slicing notes(3)(4) reread. Original code/sources/CPU numeric evidence kept by fingerprint; no original model execution repeated.",
    "render":{"command_argv":render_command,"exit_code":render.returncode,"renderer":version},
    "browser":browser_receipt,
    "environment":{"python":sys.version,"device":"cpu","torch_imported":False},
    "model_execution":False,"training":False,"weights_downloaded":False,
    "manual_view_pending":True,
}
write_json("preparation-receipt.json", receipt)
print(json.dumps({"source_sha256":receipt["source_sha256"],"current_figure_sha256":EXPECTED_FIGURE,
                  "actual_text_change_count":len(changed),"all_original_artifacts_unchanged":True,
                  "native_render_exit_code":render.returncode,"browser":browser_receipt,
                  "manual_view_pending":True,"model_execution":False}, ensure_ascii=False, indent=2))
