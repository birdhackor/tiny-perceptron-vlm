"""Read-only route checks against the actually running 8788 course preview."""
import hashlib
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts/natural-v4-factual/R.3"
BASE = "http://127.0.0.1:8788/"
result = {"base_url": BASE, "chapter_routes": [], "section_notebook_routes": [], "figures_rendered": []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    result["browser_version"] = browser.version
    page = browser.new_page(viewport={"width": 1360, "height": 1000}, device_scale_factor=1)
    response = page.goto(BASE + "course.html#R.3", wait_until="networkidle")
    assert response.status == 200
    heading = page.locator('[id="R.3"]')
    rows = heading.evaluate('el => {let rows=[]; for(let n=el.nextElementSibling;n&&!/^H[12]$/.test(n.tagName);n=n.nextElementSibling){rows.push(...Array.from(n.querySelectorAll("a")).map(a=>({text:a.innerText,url:a.href})));}return rows;}')
    result["r3_links"] = rows
    chapter_links = [r for r in rows if "chapter-" in r["url"]]
    assert len(chapter_links) == 23
    index_link = next(r for r in rows if r["text"] == "章節目錄")
    response = page.goto(index_link["url"], wait_until="networkidle")
    article = page.locator("article")
    assert response.status == 200 and article.locator('a[href$="/chapter-07.html"]').count() == 1
    result["index_route"] = {"url": page.url, "status": response.status, "chapter_7_link": True}
    all_ids = []
    for row in chapter_links:
        response = page.goto(row["url"], wait_until="domcontentloaded")
        stem = re.search(r"chapter-(.+)\.html", row["url"])[1]
        source = ROOT / "course/chapters" / (stem + ".md")
        ids = re.findall(r"^## ([A-Z\d]+\.\d+) ", source.read_text(), re.M)
        article = page.locator("article")
        visible_links = article.locator("a").evaluate_all('els=>els.map(a=>({text:a.innerText,url:a.href}))')
        for identifier in ids:
            assert any(x["url"] == BASE + identifier + ".html" for x in visible_links), (stem, identifier)
        result["chapter_routes"].append({"url": page.url, "status": response.status, "heading": article.locator("h1").inner_text().strip(), "sections_listed": len(ids), "all_expected_section_links_present": True})
        assert response.status == 200
        all_ids.extend(ids)
    for identifier in all_ids:
        response = page.request.get(BASE + identifier + ".html")
        assert response.status == 200
        html = response.text()
        notebook_links = re.findall(r'href="([^"]+' + re.escape(identifier) + r'\.ipynb)"', html)
        assert notebook_links, identifier
        result["section_notebook_routes"].append({"section": identifier, "status": response.status, "notebook_links": notebook_links})
    response = page.goto(BASE + "chapter-07.html", wait_until="networkidle")
    page.locator('article a[href$="/7.4.html"]').click()
    page.wait_for_load_state("networkidle")
    assert "回答第一個 token" in page.locator("article h1").first.inner_text()
    notebook_links = page.locator('article a[href$="7.4.ipynb"]').evaluate_all('els=>els.map(a=>({text:a.innerText,url:a.href}))')
    assert notebook_links
    result["exercise_route"] = {"url": page.url, "heading": page.locator("article h1").first.inner_text(), "notebook_links": notebook_links}
    for name in ["foundations_answer_alignment", "cache", "natural_shared_chat"]:
        source = ROOT / "course/figures" / (name + ".svg")
        page.goto(source.as_uri(), wait_until="load")
        svg = page.locator("svg")
        render = OUT / (name + ".png")
        svg.screenshot(path=str(render))
        result["figures_rendered"].append({"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "render": str(render.relative_to(ROOT)), "render_sha256": hashlib.sha256(render.read_bytes()).hexdigest()})
    result["denominators"] = {"r3_chapter_rows": 23, "chapter_entries_checked": len(chapter_links), "section_pages_checked": len(all_ids), "figures_rendered": 3}
    browser.close()
print(json.dumps(result, ensure_ascii=False, indent=2))
