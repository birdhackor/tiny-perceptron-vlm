"""Observe and download the actual preview; open official Colab home only."""

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[4]
OUT = Path("/workspace/work/omni-technical-r-1-callback-v3/browser")
OUT.mkdir(parents=True, exist_ok=False)
URL = "http://127.0.0.1:8774/12.17.html"
records = {}
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/workspace/work/browsers/chromium-1234/chrome-linux64/chrome", args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="en-US", accept_downloads=True)
    page = context.new_page()
    response = page.goto(URL, wait_until="networkidle", timeout=45000)
    assert response.status == 200
    page.locator("#colab-upload-instructions").scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    local_text = page.locator("body").inner_text()
    assert "尚未發布的預覽版" in local_text
    assert "Upload notebook" in local_text and "File → Upload notebook" in local_text
    (OUT / "preview-body.txt").write_text(local_text)
    page.screenshot(path=str(OUT / "preview-desktop.png"))
    with page.expect_download() as event:
        page.locator(".actions a[download]").click()
    download = event.value
    download_path = OUT / "actual-downloaded-12.17.ipynb"
    download.save_as(str(download_path))
    assert sha256(download_path.read_bytes()).hexdigest() == sha256((ROOT / "notebooks/12/12.17.ipynb").read_bytes()).hexdigest()
    with context.expect_page() as event:
        page.locator('.actions a[href="https://colab.research.google.com/"]').click()
    home = event.value
    home.wait_for_load_state("domcontentloaded", timeout=45000)
    home.wait_for_timeout(2000)
    home_text = home.locator("body").inner_text()
    (OUT / "official-home-body.txt").write_text(home_text)
    home.screenshot(path=str(OUT / "official-home.png"))
    records["official_home"] = {"actual_url": home.url, "upload_notebook_text_present": "Upload notebook" in home_text, "upload_text_present": "Upload" in home_text, "scope": "Opened official home anonymously only; no file upload, login, Connect, cell, or cloud runtime."}
    mobile = browser.new_context(viewport={"width": 390, "height": 844}, locale="en-US")
    phone = mobile.new_page()
    phone.goto(URL, wait_until="networkidle", timeout=45000)
    phone.locator("#colab-upload-instructions").scroll_into_view_if_needed()
    phone.wait_for_timeout(500)
    phone.screenshot(path=str(OUT / "preview-mobile.png"))
    version = browser.version
    browser.close()

result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu --with playwright python docs/technical-reviews/artifacts/omni-technical-r-1-v1/view_preview_upload_callback_v3.py",
    "environment": {"python": sys.version, "playwright": importlib.metadata.version("playwright"), "chromium": version, "device": "CPU; fresh anonymous desktop/mobile contexts"},
    "local_preview_url": URL, "local_http_status": response.status,
    "actual_download": {"suggested_filename": download.suggested_filename, "path_outside_git": str(download_path), "sha256": sha256(download_path.read_bytes()).hexdigest(), "matches_current_notebook": True},
    "official_home_observation": records["official_home"],
    "outside_evidence": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in [OUT / "preview-desktop.png", OUT / "preview-mobile.png", OUT / "official-home.png", OUT / "preview-body.txt", OUT / "official-home-body.txt"]],
    "scope": "Actual local undeployed preview download and official-home navigation. Images must be separately viewed before claiming visual inspection. No Colab upload, login, cloud runtime, or publication was performed."}
with Path(__file__).with_name("preview-upload-browser-callback-v3.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
