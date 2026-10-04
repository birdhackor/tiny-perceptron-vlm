"""Actual Chromium guide navigation and prerequisite SVG rendering."""
import hashlib
import json
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
events = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width":1280,"height":900})
    response = page.goto("http://127.0.0.1:8782/", wait_until="networkidle")
    events.append({"action":"goto", "url":page.url,"status":response.status,"title":page.title()})
    links = page.locator('a[href*="natural-v4-student"]').all()
    events.append({"guide_links":[{"text":a.inner_text(),"href":a.get_attribute("href")} for a in links]})
    if not links:
        raise RuntimeError("No student guide link in actual root DOM")
    # The first sidebar link is inside a collapsed navigation group. Navigate
    # in the actual browser to its observed href, then use guide content links.
    page.goto(links[0].get_attribute("href"), wait_until="networkidle")
    page.wait_for_load_state("networkidle")
    events.append({"action":"browser address navigation to observed sidebar href", "url":page.url,"title":page.title()})
    main = page.locator("article")
    if main.count() != 1:
        main = page.locator("main")
    text = main.inner_text()
    (OUT/"browser-guide-visible.txt").write_text(text,encoding="utf-8")
    page.screenshot(path=str(OUT/"browser-guide-top.png"))
    page.get_by_text("兩個官方模型快照",exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(OUT/"browser-guide-measurements.png"))
    events.append({"guide_heading_count":main.locator("h2").count(),"main_visible_characters":len(text),"visible_reading_record":"browser-guide-visible.txt"})
    anchors = main.locator("a").evaluate_all("els=>els.map(e=>({text:e.innerText,href:e.href}))")
    events.append({"guide_links":anchors})
    target = main.get_by_role("link",name="20.1 的輸入分工",exact=True)
    target.click()
    page.wait_for_load_state("networkidle")
    events.append({"action":"click 20.1 input division", "url":page.url,"visible_heading":page.locator("h2").filter(has_text="20.1 ").first.inner_text()})
    image = page.locator('img[src*="natural_shared_chat.svg"]')
    if image.count()!=1:
        raise RuntimeError("Expected shared-chat figure in actual chapter")
    svg_url=image.get_attribute("src")
    events.append({"prerequisite_figure_src":svg_url})
    page.goto("http://127.0.0.1:8782/figures/natural_shared_chat.svg",wait_until="networkidle")
    page.set_viewport_size({"width":720,"height":1010})
    page.screenshot(path=str(OUT/"natural_shared_chat.render.png"),full_page=True)
    events.append({"action":"SVG render", "url":page.url,"browser":browser.version})
    browser.close()

source = ROOT/"docs/natural-assistant/v4/STUDENT.md"
old = subprocess.check_output(["git","show","307c325:docs/natural-assistant/v4/STUDENT.md"],cwd=ROOT)
events.append({"current_fullfile_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"preview_revision":"307c325","preview_revision_source_sha256":hashlib.sha256(old).hexdigest(),"source_bytes_match_preview_revision":old==source.read_bytes(),"reading_times":"not finalized; this inspection is not publication approval"})
(OUT/"browser-inspection.json").write_text(json.dumps(events,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(events,ensure_ascii=False,indent=2))
print(text)
