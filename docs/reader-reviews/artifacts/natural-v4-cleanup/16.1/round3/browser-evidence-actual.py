from pathlib import Path
import json
import sys
from urllib.parse import unquote
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1])
base = "http://127.0.0.1:8789/"
log = {"base_url": base, "requested_source_url": base + "16.1.html", "performed_actual_browser_click": False, "browser_engine": "Playwright Chromium", "scope": "Only current chapter introduction/16.1 and clicked efficiency-method subsection inspected; no whole chapter or document reading"}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    context = browser.new_context(viewport={"width": 1280, "height": 850}, device_scale_factor=1)
    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base) else route.abort())
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    page.set_default_timeout(10000)
    log["chromium_version"] = browser.version
    response = page.goto(log["requested_source_url"], wait_until="domcontentloaded")
    log["source_response_status"] = response.status if response else None
    log["source_actual_url"] = page.url
    link = page.get_by_role("link", name="T.8的效率量測方法", exact=True)
    log["source_link_count"] = link.count()
    link.scroll_into_view_if_needed()
    log["authored_rendered_href"] = link.get_attribute("href")
    log["resolved_href"] = link.evaluate("el => el.href")
    paragraph = link.locator("xpath=..")
    log["source_method_paragraph"] = paragraph.inner_text()
    paragraph.screenshot(path=str(out / "source-method-link.png"))
    log["source_section_headings"] = page.get_by_role("heading").filter(has_text="16.1 最慢或最佔空間的是哪裡？").evaluate_all("els => els.map(el => ({tag: el.tagName, text: el.innerText}))")
    link.click()
    page.wait_for_load_state("domcontentloaded")
    log["performed_actual_browser_click"] = True
    log["destination_actual_url"] = page.url
    fragment = unquote(page.url.split("#", 1)[1])
    log["decoded_destination_fragment"] = fragment
    heading = page.locator("h3").filter(has_text="選讀：效率實驗的量測條件")
    log["target_heading_count"] = heading.count()
    log["target_heading_tag"] = heading.evaluate("el => el.tagName")
    log["target_heading_id"] = heading.get_attribute("id")
    log["target_heading_text"] = heading.inner_text()
    log["fragment_identifies_actual_h3"] = heading.evaluate("(el, id) => document.getElementById(id) === el", fragment)
    (out / "browser-click-progress.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    extracted = heading.evaluate(r"""el => {
        const nodes = [el];
        let current = el.nextElementSibling;
        while (current && !/^H[1-3]$/.test(current.tagName)) {
            nodes.push(current);
            current = current.nextElementSibling;
        }
        return {text: nodes.map(n => n.innerText).join('\n\n'), html: nodes.map(n => n.outerHTML).join('\n'), svg_refs: nodes.flatMap(n => [...n.querySelectorAll('img[src$=".svg"]')].map(i => i.getAttribute('src'))), first_body_tag: nodes[1]?.tagName ?? null};
    }""")
    log["first_body_tag"] = extracted["first_body_tag"]
    log["target_svg_refs"] = extracted["svg_refs"]
    (out / "clicked-target-subsection.txt").write_text(extracted["text"], encoding="utf-8")
    (out / "clicked-target-subsection.html").write_text(extracted["html"], encoding="utf-8")
    heading.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, -80)")
    page.screenshot(path=str(out / "clicked-target-heading.png"))
    log["target_heading_top_px_after_navigation"] = heading.bounding_box()["y"]
    (out / "browser-navigation-evidence.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context.tracing.stop(path=str(out / "actual-click-trace.zip"))
    browser.close()
print(json.dumps(log, ensure_ascii=False, indent=2))
print("ACTUALLY CLICKED TARGET SUBSECTION:")
print(extracted["text"])
