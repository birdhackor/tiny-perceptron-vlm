"""Render only the already served 11.4 page; retain browser DOM and screenshots."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
URL = "http://127.0.0.1:8765/11.4.html"
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
    receipts = []
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        context = browser.new_context(viewport={"width":width,"height":height}, device_scale_factor=1)
        page = context.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=20000)
        image = page.locator('article img[src$="figures/rewrite-11-same-image-questions.svg"]')
        image.wait_for()
        page.wait_for_function("document.querySelector('article img').complete && document.querySelector('article img').naturalWidth > 0", timeout=15000)
        page.screenshot(path=str(HERE / f"page-{name}.png"), full_page=True, timeout=15000)
        article = page.locator("article")
        (HERE / f"page-{name}-article.html").write_text(article.inner_html(), encoding="utf-8")
        body = page.evaluate("""() => {
            const article=document.querySelector('article');
            return {title: article.querySelector('h1,h2').textContent,
              paragraphs:[...article.querySelectorAll('p')].map(x=>x.textContent),
              fences:[...article.querySelectorAll('pre code')].map(x=>x.textContent),
              figure_url:article.querySelector('img').src};
        }""")
        source = (HERE / "section.md").read_text()
        paragraphs = [x for x in source.split('\n\n') if x.startswith(('同一張紅圓圖','兩筆記錄','訓練時','再固定問題','回答「'))]
        assert len(paragraphs) == 5
        # Inline code delimiters are Markdown formatting, not article text.
        assert all(x.replace('`','') in body['paragraphs'] for x in paragraphs)
        assert (HERE / "fence-1.py").read_text().strip() in [x.strip() for x in body['fences']]
        response = context.request.get(body['figure_url'])
        served = response.body()
        local = (ROOT / 'course/figures/rewrite-11-same-image-questions.svg').read_bytes()
        assert response.ok and served == local
        receipts.append({"viewport":name,"width":width,"height":height,"browser_version":browser.version,
            "url":URL,"paragraphs_match":5,"fence_matches":True,
            "served_figure_sha256":hashlib.sha256(served).hexdigest(),
            "image_box":image.bounding_box(),"article_box":article.bounding_box(),"body":body})
        context.close()
    browser.close()
(HERE/'render-receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k != 'body'} for r in receipts],ensure_ascii=False))
