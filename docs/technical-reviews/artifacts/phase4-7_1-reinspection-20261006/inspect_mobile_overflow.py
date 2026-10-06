"""Check the actual mobile code/output scroll contract after inspecting the render."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    page.set_default_timeout(15000)
    response = page.goto("http://127.0.0.1:8765/7.1.html", wait_until="networkidle", timeout=20000)
    assert response and response.status == 200
    measurements = page.locator("article pre").evaluate_all("elements => elements.map(e => ({text: e.innerText, clientWidth:e.clientWidth, scrollWidth:e.scrollWidth, overflowX:getComputedStyle(e).overflowX}))")
    output = page.locator("article pre").filter(has_text="答案標籤Y").last
    output.scroll_into_view_if_needed()
    output.evaluate("element => { element.scrollLeft = element.scrollWidth; }")
    after = output.evaluate("e => ({clientWidth:e.clientWidth, scrollWidth:e.scrollWidth, scrollLeft:e.scrollLeft, overflowX:getComputedStyle(e).overflowX})")
    assert after["scrollLeft"] > 0 and after["overflowX"] in ("auto", "scroll")
    page.screenshot(path=str(HERE / "mobile-output-scrolled.png"))
    browser.close()
(HERE / "mobile-overflow.json").write_text(json.dumps({"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-7_1-reinspection-20261006/inspect_mobile_overflow.py", "before":measurements,"scrolled_output":after},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(after,ensure_ascii=False))
