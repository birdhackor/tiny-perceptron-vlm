"""Render the current local section; block unrelated remote page resources."""
import hashlib
import json
import platform
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
url = "http://127.0.0.1:8765/11.10.html"
evidence = {"url": url, "python": platform.python_version(), "browser_executable": "/usr/bin/chromium", "remote_resources": "aborted to avoid unrelated external page loads", "screens": []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-gpu"])
    evidence["chromium_version"] = browser.version
    for label,width,height in [("desktop",1280,800),("mobile",390,844)]:
        page = browser.new_page(viewport={"width":width,"height":height}, device_scale_factor=1)
        page.route("**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:8765/") or route.request.url.startswith("data:") else route.abort())
        response = page.goto(url,wait_until="domcontentloaded",timeout=20000)
        page.wait_for_timeout(1000)
        text = page.locator("article").inner_text()
        for phrase in ["每列是一個要讀取資訊的位置", "76176", "兩者最後圖片題都是 9/12"]:
            assert phrase in text
        (HERE/f"render/{label}-article.txt").write_text(text)
        page.screenshot(path=str(HERE/f"render/{label}-viewport.png"), full_page=False)
        page.screenshot(path=str(HERE/f"render/{label}-full.png"), full_page=True)
        evidence["screens"].append({"label":label,"viewport":[width,height],"http_status":response.status,
                                  "article_sha256":hashlib.sha256(text.encode()).hexdigest(),
                                  "full_page":f"render/{label}-full.png","viewport_file":f"render/{label}-viewport.png"})
        page.close()
    browser.close()
(HERE/"render-results.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(evidence,ensure_ascii=False,indent=2))
