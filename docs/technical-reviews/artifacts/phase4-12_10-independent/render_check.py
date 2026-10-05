from pathlib import Path
import hashlib,json,re,sys
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
ROOT=Path.cwd();OUT=ROOT/"docs/technical-reviews/artifacts/phase4-12_10-independent"
source=(OUT/"extraction/section.md").read_bytes();html=(OUT/"page.html").read_bytes()
soup=BeautifulSoup(html,"html.parser");main=soup.select_one("main")
source_text=source.decode()
paragraphs=[]
# Compare each current source paragraph and fence to the served main article.
for block in source_text.split("\n\n"):
    block=block.strip()
    if not block or block.startswith(("## ","```","<")):continue
    plain=re.sub(r"\[([^]]+)\]\([^)]+\)",r"\1",block).replace("`","")
    paragraphs.append(plain)
rendered=[p.get_text() for p in main.find_all("p")]
parity=[{"source_paragraph":p,"in_served_main":p in rendered} for p in paragraphs]
assert all(p["in_served_main"] for p in parity),parity
fence=(OUT/"extraction/fence-1.py").read_text()
assert main.select_one("pre code").get_text().rstrip("\n")==fence.rstrip("\n")
assert not main.find_all("img")
results={"source_sha256":hashlib.sha256(source).hexdigest(),"html_sha256":hashlib.sha256(html).hexdigest(),"paragraph_parity":parity,"python_fence_parity":True,"source_figure_count":0,"rendered_main_figure_count":0,"url":"http://127.0.0.1:8765/12.10.html","screenshots":[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox"])
    results["chromium_version"]=browser.version
    for label,width,height in [("desktop",1280,800),("mobile",390,844)]:
        context=browser.new_context(viewport={"width":width,"height":height},device_scale_factor=1)
        page=context.new_page();page.goto(results["url"],wait_until="networkidle",timeout=30000)
        page.screenshot(path=str(OUT/(label+"-viewport.png")))
        page.screenshot(path=str(OUT/(label+"-full.png")),full_page=True)
        bounds=page.locator("main").bounding_box()
        results["screenshots"].append({"label":label,"viewport":{"width":width,"height":height},"main_bounds":bounds,"document_width":page.evaluate("document.documentElement.scrollWidth"),"document_height":page.evaluate("document.documentElement.scrollHeight")})
        context.close()
    browser.close()
(OUT/"render-result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in results.items() if k!="paragraph_parity"},ensure_ascii=False,indent=2))
