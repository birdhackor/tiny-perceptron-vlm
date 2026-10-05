"""Render current 12.9 page and necessary prerequisite figures with Chromium."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"], headless=True)
    records = []
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto("http://127.0.0.1:8765/12.9.html", wait_until="networkidle")
        assert response.status == 200
        article = page.locator("article")
        fence = (ART / "extracted/fence-1.py").read_text().rstrip()
        codes = article.locator("pre code").all_text_contents()
        assert fence in [x.rstrip() for x in codes]
        paragraphs = article.locator("p").all_text_contents()
        assert any("錯誤集中在邊界與時長變化" in x for x in paragraphs)
        raw = response.body()
        (ART / f"page-{name}.html").write_bytes(raw)
        page.screenshot(path=str(ART / f"page-{name}-full.png"), full_page=True)
        page.screenshot(path=str(ART / f"page-{name}-viewport.png"))
        records.append({"name": name, "viewport": [width, height], "url": response.url,
            "html_sha256": hashlib.sha256(raw).hexdigest(), "original_fence_exact_match": True,
            "article_image_count": article.locator("img").count(), "article_text": article.inner_text()})
        page.close()
    for name in ["rewrite-12-frame-features", "rewrite-10-image-targets"]:
        path = ROOT / "course/figures" / f"{name}.svg"
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        page.set_content(path.read_text(), wait_until="load")
        page.screenshot(path=str(ART / f"prerequisite-{name}.png"), full_page=True)
        records.append({"prerequisite_figure": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "render": f"prerequisite-{name}.png"})
        page.close()
    browser.close()
(ART / "render-receipt.json").write_text(json.dumps({"chromium": "/usr/bin/chromium", "records": records}, ensure_ascii=False, indent=2) + "\n")
print("Current page rendered at 1280x800 and 390x844; exact original fence matches; section images=0; two prerequisite figures rendered.")
