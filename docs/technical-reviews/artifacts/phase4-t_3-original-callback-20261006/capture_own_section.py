"""Render current website, inspect only T.3 H2 up to next H2, save true screenshots."""

import importlib.metadata
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765/training.html"
extract = """() => {
 const h = document.getElementById('T.3');
 if (!h) throw new Error('Own T.3 heading unavailable');
 const next = [...document.querySelectorAll('h2')].find(x => x !== h && Boolean(h.compareDocumentPosition(x) & Node.DOCUMENT_POSITION_FOLLOWING));
 const range = document.createRange(); range.setStartBefore(h); if(next) range.setEndBefore(next); else range.setEndAfter(h.parentElement.lastChild);
 const fragment = range.cloneContents(); const host = document.createElement('div'); host.append(fragment);
 const links = [...host.querySelectorAll('a')].map(x=>({text:x.textContent,href:x.getAttribute('href')}));
 const details = [...document.querySelectorAll('details')].filter(x => range.intersectsNode(x));
 details.forEach(x=>x.open=true);
 return {heading:h.textContent.trim(),html:host.innerHTML,text:host.innerText || host.textContent,
         links, imageCount:host.querySelectorAll('img,svg,canvas').length,detailsOpened:details.length,
         startId:h.id,nextBoundaryPresent:Boolean(next),
         bounds:{left:h.parentElement.getBoundingClientRect().left,right:h.parentElement.getBoundingClientRect().right,
                 top:h.getBoundingClientRect().top+window.scrollY,
                 bottom:next ? next.getBoundingClientRect().top+window.scrollY : h.parentElement.getBoundingClientRect().bottom+window.scrollY}};
}"""
screens = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                args=["--no-sandbox", "--disable-dev-shm-usage"])
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto(URL, wait_until="networkidle", timeout=30000)
        assert response and response.status == 200
        page.locator('h2[id="T.3"]').scroll_into_view_if_needed()
        own = page.evaluate(extract)
        assert own["nextBoundaryPresent"]
        assert "固定比較包含兩類模型、四個設定" in own["text"]
        assert "一字、三字、五字視窗的 MLP" in own["text"]
        assert any(link["text"].strip() == "2.5" and link["href"] == "http://127.0.0.1:8765/2.5.html" for link in own["links"]), own["links"]
        # Expand only own details, recompute actual page bounds, clip only own section.
        own = page.evaluate(extract)
        bounds = own["bounds"]
        x = max(0, bounds["left"])
        right = min(width, bounds["right"])
        screenshot = OUT / ("own-T.3-" + label + ".png")
        (OUT / ("own-T.3-" + label + "-geometry.json")).write_text(json.dumps({"bounds": bounds, "clip_x": x, "clip_right": right}, indent=2) + "\n")
        page.screenshot(path=str(screenshot), full_page=True, clip={"x": x, "y": bounds["top"], "width": right-x,
                                                  "height": bounds["bottom"]-bounds["top"]})
        (OUT / ("own-T.3-" + label + ".html")).write_text(own["html"])
        (OUT / ("own-T.3-" + label + ".txt")).write_text(own["text"])
        screens.append({"viewport": {"width": width, "height": height}, "label": label, "url": URL,
                        "http_status": response.status, "own_heading": own["heading"], "start_id": own["startId"],
                        "next_h2_boundary": True, "details_opened": own["detailsOpened"],
                        "image_count": own["imageCount"], "screenshot": screenshot.name,
                        "clip": {"x": x, "y": bounds["top"], "width": right-x, "height": bounds["bottom"]-bounds["top"]},
                        "own_links": own["links"]})
        page.close()
    version = browser.version
    browser.close()
metadata = {"kind": "actual_own_section_website_render", "url": URL, "scope": "Only own T.3 H2 through immediately next H2; no other page/section/root-parity evidence used.",
            "environment": {"python": sys.version, "python_executable": sys.executable,
                            "playwright": importlib.metadata.version("playwright"), "browser": version,
                            "browser_executable": "/usr/bin/chromium", "headless": "true"},
            "captures": screens, "human_visual_inspection": "Pending explicit view_image calls by original reviewer after capture."}
(OUT / "website-render.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "captured_actual_own_section", "browser": version, "viewports": [x["viewport"] for x in screens], "images_in_own_section": [x["image_count"] for x in screens]}, ensure_ascii=False))
