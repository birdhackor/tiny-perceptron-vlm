"""Observe the official anonymous Colab Connect account gate; never sign in."""

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


ARTIFACTS = Path(__file__).parent
OUTSIDE = Path("/workspace/work/omni-technical-r-1-callback-v1/colab-account")
OUTSIDE.mkdir(parents=True, exist_ok=False)
URL = "https://colab.research.google.com/notebooks/intro.ipynb"
started = datetime.now(UTC).isoformat()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path="/workspace/work/browsers/chromium-1234/chrome-linux64/chrome",
        args=["--no-sandbox"],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="en-US")
    page = context.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    button = page.locator("colab-connect-button")
    button.wait_for(state="visible", timeout=30000)
    page.wait_for_timeout(2500)
    button.click()
    page.get_by_text("You must be logged in with a Google Account to continue.", exact=True).wait_for(timeout=15000)
    text = page.locator("body").inner_text()
    dialogs = page.get_by_role("dialog").all_text_contents()
    screenshot = OUTSIDE / "after.png"
    dom = OUTSIDE / "body-after.txt"
    page.screenshot(path=str(screenshot))
    dom.write_text(text)
    assert "You must be logged in with a Google Account to continue." in text
    observed_url = page.url
    browser_version = browser.version
    browser.close()

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "started_at": started,
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu --with playwright python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_colab_account_callback_v1.py",
    "environment": {"python": sys.version, "playwright": importlib.metadata.version("playwright"), "chromium": browser_version, "device": "CPU; fresh anonymous Chromium context; en-US"},
    "official_url": URL,
    "observed_url": observed_url,
    "action": "Open official intro, click Connect once, inspect account gate, close browser; no login, cell run, or successful cloud runtime connection.",
    "observed_primary_text": "You must be logged in with a Google Account to continue.",
    "dialogs": dialogs,
    "account_login_performed": False,
    "cloud_runtime_execution_verified": False,
    "screenshot": {"path": str(screenshot), "sha256": sha256(screenshot.read_bytes()).hexdigest()},
    "full_DOM_text": {"path": str(dom), "sha256": sha256(dom.read_bytes()).hexdigest()},
    "scope": "Directly supports the Google Account prerequisite for this observed official hosted-Connect route; no course or cloud cell execution is established.",
}
with (ARTIFACTS / "colab-account-callback-v1.json").open("x") as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
