from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

root = Path.cwd()
out = root / "docs/technical-reviews/artifacts/natural-v4-factual/20.7"
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
    page = browser.new_page(viewport={"width": 800, "height": 760}, device_scale_factor=1)
    receipts = []
    for name in ["natural-v4-answer-mask.svg", "foundations_answer_alignment.svg"]:
        source = root / "course/figures" / name
        page.set_content(source.read_text())
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(out / (name + ".png")), full_page=True)
        receipts.append({"source": source.relative_to(root).as_posix(),
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "render": (out / (name + ".png")).relative_to(root).as_posix(),
                         "viewport": {"width": 800, "height": 760},
                         "browser": browser.version, "device_scale_factor": 1})
    browser.close()
(out / "render-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
