"""Capture this reviewer’s actual current-page browser observations."""

import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765/10.2.html"
observations = []
with sync_playwright() as runtime:
    browser = runtime.chromium.launch(
        executable_path="/usr/bin/chromium",
        headless=True,
        args=["--no-sandbox", "--disable-gpu"],
    )
    version = browser.version
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto(URL, wait_until="networkidle", timeout=15000)
        assert response.status == 200
        page.evaluate("document.fonts.ready")
        code_blocks = page.locator("pre").all_inner_texts()
        assert any('image = scene("red", "square")[None]' in block for block in code_blocks)
        image = page.locator('img[src*="rewrite-10-patch-order.svg"]')
        assert image.count() == 1
        assert image.evaluate("element => element.complete && element.naturalWidth === 640 && element.naturalHeight === 550")
        box = image.bounding_box()
        assert box["width"] > 0 and box["height"] > 0
        screenshot = OUT / f"{label}-current.png"
        page.screenshot(path=str(screenshot), full_page=True)
        if label == "desktop":
            (OUT / "current-page.html").write_bytes(response.body())
        observations.append({
            "label": label,
            "viewport": [width, height],
            "url": URL,
            "status": response.status,
            "explicit_current_scene_call_in_rendered_code": True,
            "figure_source": image.get_attribute("src"),
            "figure_natural_dimensions": [640, 550],
            "figure_visible_box": box,
            "document_scroll_width": page.evaluate("document.documentElement.scrollWidth"),
            "screenshot": screenshot.name,
            "screenshot_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        })
        page.close()
    browser.close()
result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-10_2-callback-20261006/page/capture_current_page.py",
    "chromium": version,
    "executable": "/usr/bin/chromium",
    "scope": "The actual current section page only. The reviewer must inspect screenshots separately; DOM observations do not establish readability.",
    "observations": observations,
}
(OUT / "browser-receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
