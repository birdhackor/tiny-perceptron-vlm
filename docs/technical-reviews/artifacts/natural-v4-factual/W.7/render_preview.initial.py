"""Capture the actual served W.7 directory table, without source changes."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8788/first-steps.html#W.7"
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=1)
    response = page.goto(URL, wait_until="networkidle", timeout=25000)
    page.locator('[id="W.7"]').scroll_into_view_if_needed()
    table = page.locator('[id="W.7"]').locator("xpath=following-sibling::table[1]")
    table.screenshot(path=str(ART / "preview-table.png"))
    rows = table.locator("tr").evaluate_all("rows => rows.map(row => Array.from(row.children).map(cell => cell.innerText))")
    box = table.bounding_box()
    result = {"url": URL, "http_status": response.status, "browser": browser.version, "rows": rows, "table_box": box, "screenshot": "preview-table.png", "viewport": {"width": 1200, "height": 900}, "source_of_table": "actual executed preview, no rebuilt source"}
    assert len(rows) == 8 and all(len(row) == 2 for row in rows), result
    (ART / "preview-render.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    browser.close()
