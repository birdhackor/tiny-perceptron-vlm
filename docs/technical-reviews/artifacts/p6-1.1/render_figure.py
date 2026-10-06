"""Render the frozen SVG in Chromium; keep a full native screenshot for inspection."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

A = Path(__file__).resolve().parent
svg = A / "frozen/course/figures/rewrite-01-character-ids.svg"
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 640, "height": 710}, device_scale_factor=1)
    page.set_content("<style>html,body{margin:0;padding:0}svg{display:block}</style>" + svg.read_text(), wait_until="load")
    page.evaluate("document.fonts.ready")
    assert page.locator("svg").count() == 1
    texts = page.locator("svg text").all_text_contents()
    page.screenshot(path=str(A / "figure-native.png"), full_page=True)
    result = {"browser": browser.version, "device": "CPU headless Chromium",
              "command": "/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-1.1/render_figure.py",
              "source": "course/figures/rewrite-01-character-ids.svg",
              "source_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
              "viewport": {"width":640, "height":710}, "texts": texts,
              "screenshot": "figure-native.png", "screenshot_sha256": hashlib.sha256((A / "figure-native.png").read_bytes()).hexdigest(),
              "network": "none; frozen SVG bytes supplied through page.set_content",
              "initial_tool_limitation": "Chromium administrator policy rejects file:// navigation. Rendering the identical saved SVG bytes with page.set_content succeeded."}
    browser.close()
(A / "figure-render-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
