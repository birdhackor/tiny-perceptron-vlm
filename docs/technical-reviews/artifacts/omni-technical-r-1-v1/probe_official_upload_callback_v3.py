"""Inspect the real official-home upload control without selecting a file."""

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

OUT = Path("/workspace/work/omni-technical-r-1-callback-v3/official-upload-probe")
OUT.mkdir(parents=True, exist_ok=False)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/workspace/work/browsers/chromium-1234/chrome-linux64/chrome", args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="en-US")
    page = context.new_page()
    page.goto("https://colab.research.google.com/", wait_until="domcontentloaded", timeout=45000)
    upload = page.get_by_role("button", name="Upload notebook", exact=True)
    upload.wait_for(state="visible", timeout=30000)
    observed = {"url": page.url, "accessible_role": "button", "accessible_name": "Upload notebook", "visible": upload.is_visible(), "text": upload.inner_text(), "root": upload.evaluate("element => ({isShadowRoot: element.getRootNode() instanceof ShadowRoot, host: element.getRootNode().host?.localName ?? null})")}
    page.screenshot(path=str(OUT / "official-upload-control.png"))
    browser_version = browser.version
    browser.close()
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu --with playwright python docs/technical-reviews/artifacts/omni-technical-r-1-v1/probe_official_upload_callback_v3.py",
    "environment": {"python": sys.version, "chromium": browser_version, "device": "CPU; fresh anonymous browser"},
    "official_control_observed": observed,
    "screenshot": {"path": str(OUT / "official-upload-control.png"), "sha256": sha256((OUT / "official-upload-control.png").read_bytes()).hexdigest()},
    "scope": "Observed the official visible Upload notebook control inside its shadow root. No click, file selection, upload, login, Connect, cell execution, or cloud runtime."}
with Path(__file__).with_name("official-upload-control-callback-v3.json").open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
