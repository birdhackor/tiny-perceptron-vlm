"""Observe the new local12.2 preview and download its actual current Notebook."""

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[4]
OUT = Path("/workspace/work/omni-technical-r-1-callback-v4/browser")
OUT.mkdir(parents=True, exist_ok=False)
URL = "http://127.0.0.1:8775/12.2.html"
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/workspace/work/browsers/chromium-1234/chrome-linux64/chrome", args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="en-US", accept_downloads=True)
    page = context.new_page()
    response = page.goto(URL, wait_until="networkidle", timeout=45000)
    assert response.status == 200
    body = page.locator("body").inner_text()
    assert "也把兩行print中的頻率名稱改成9500Hz、6500Hz" in body
    assert "每一點都只差正負號 True" in body
    assert "尚未發布的預覽版" in body and "Upload notebook" in body and "File → Upload notebook" in body
    (OUT / "12.2-body.txt").write_text(body)
    page.get_by_text("也把兩行print中的頻率名稱改成9500Hz、6500Hz", exact=False).first.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "12.2-exercise-desktop.png"))
    page.locator("#colab-upload-instructions").scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "preview-footer-desktop.png"))
    colab_link = page.locator('.actions a[href="https://colab.research.google.com/"]')
    assert colab_link.count() == 1
    with page.expect_download() as event:
        page.locator(".actions a[download]").click()
    download = event.value
    downloaded = OUT / "actual-downloaded-12.2.ipynb"
    download.save_as(str(downloaded))
    assert downloaded.read_bytes() == (ROOT / "notebooks/12/12.2.ipynb").read_bytes()
    mobile = browser.new_context(viewport={"width": 390, "height": 844}, locale="en-US")
    phone = mobile.new_page()
    phone.goto(URL, wait_until="networkidle", timeout=45000)
    phone.locator("#colab-upload-instructions").scroll_into_view_if_needed()
    phone.screenshot(path=str(OUT / "preview-footer-mobile.png"))
    version = browser.version
    browser.close()
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu --with playwright python docs/technical-reviews/artifacts/omni-technical-r-1-v1/view_preview_upload_callback_v4.py",
    "environment": {"python": sys.version, "playwright": importlib.metadata.version("playwright"), "chromium": version, "device": "CPU; fresh local desktop/mobile browser contexts"},
    "url": URL, "http_status": response.status, "new_exercise_text_and_CPU_output_present": True,
    "actual_download": {"path_outside_git": str(downloaded), "suggested_filename": download.suggested_filename, "sha256": sha256(downloaded.read_bytes()).hexdigest(), "matches_current_notebook": True},
    "outside_evidence": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in [OUT / "12.2-body.txt", OUT / "12.2-exercise-desktop.png", OUT / "preview-footer-desktop.png", OUT / "preview-footer-mobile.png"]],
    "scope": "Actual undeployed source14 preview and browser download. Images require separate personal view_image inspection before claiming visual review. No Colab navigation/upload/login/Connect/cell, publication, or training. Prior official Colab controls/account evidence only reused within its dated observation scope."}
with Path(__file__).with_name("preview-upload-browser-callback-v4.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
