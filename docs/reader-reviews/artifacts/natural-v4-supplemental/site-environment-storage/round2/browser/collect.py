from pathlib import Path
import json
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
base="http://127.0.0.1:8787/"
steps=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1280,"height":1000},device_scale_factor=1)
    response=page.goto(base,wait_until="networkidle")
    (out/"index.retrieved.html").write_bytes(response.body())
    (out/"index.rendered.txt").write_text(page.locator("article").inner_text())
    page.screenshot(path=str(out/"index.full.png"),full_page=True)
    print("ENTRY ROOT",page.title(),page.locator("article").inner_text())
    for name in ["參考資料","環境準備"]:
        visible=[e for e in page.get_by_role("link",name=name,exact=True).all() if e.is_visible()]
        if not visible: raise RuntimeError("No visible entry "+name)
        before=page.url
        visible[0].click()
        page.wait_for_load_state("networkidle")
        steps.append({"action":"visible browser click","label":name,"from":before,"to":page.url,"title":page.title()})
    page.screenshot(path=str(out/"environment.entry.png"))
    for key in ["environment","asset-storage"]:
        if key=="asset-storage":
            before=page.url
            page.locator("article").get_by_role("link",name="教材資產存放",exact=True).click()
            page.wait_for_load_state("networkidle")
            steps.append({"action":"visible browser click","label":"教材資產存放","from":before,"to":page.url,"title":page.title()})
        response=page.request.get(page.url)
        (out/(key+".retrieved.html")).write_bytes(response.body())
        article=page.locator("article")
        rendered=article.inner_text()
        (out/(key+".rendered.txt")).write_text(rendered)
        (out/(key+".rendered.dom.html")).write_text(page.content())
        page.screenshot(path=str(out/(key+".full.png")),full_page=True)
        info={"url":page.url,"title":page.title(),"main_text":rendered,"main_links":article.locator("a").evaluate_all("els=>els.map(e=>({label:e.innerText.trim(),href:e.href}))"),"main_images":article.locator("img,svg").evaluate_all("els=>els.map(e=>({tag:e.tagName,src:e.getAttribute('src'),alt:e.getAttribute('alt')}))")}
        (out/(key+".inspection.json")).write_text(json.dumps(info,ensure_ascii=False,indent=2))
        print("COMPLETE ASSIGNED BROWSER READ",key,json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    destinations=[("W.1","first-steps",["intro","W.1"]),("學生操作指引第2節","natural-v4-student",["intro","1.","2.","3."]),("訓練資料入口","training-assets",["all"]),("第20章資料說明","natural-v4-data",["intro","1.","6."]),("教材實作驗證","validation",["intro","先分清要檢查什麼"]),("網站發布","publishing",["intro","1."])]
    for label,key,scope in destinations:
        source="asset-storage.html" if key in ["validation","publishing"] else "environment.html"
        page.goto(base+source,wait_until="networkidle")
        link=page.locator("article").get_by_role("link",name=label,exact=True).first
        before=page.url
        href=link.get_attribute("href")
        link.click()
        page.wait_for_load_state("networkidle")
        sections=page.locator("article").evaluate("""(article, scope) => {
            const results=[]; let current={heading:'intro',nodes:[]};
            for(const node of Array.from(article.children)){
                if(node.tagName==='H2'){results.push(current);current={heading:node.innerText.replace('¶','').trim(),nodes:[]};}
                current.nodes.push(node);
            }
            results.push(current);
            return results.filter(s=>scope.includes('all')||scope.some(x=>s.heading.startsWith(x))).map(s=>({heading:s.heading,text:s.nodes.map(n=>n.innerText).join('\\n'),images:s.nodes.flatMap(n=>Array.from(n.querySelectorAll('img,svg')).map(e=>({tag:e.tagName,src:e.getAttribute('src')})))}));
        }""",scope)
        info={"action":"visible article browser click","label":label,"from":before,"to":page.url,"href":href,"title":page.title(),"read_scope":scope,"read_sections":sections}
        steps.append(info)
        (out/(key+".necessary-sections.json")).write_text(json.dumps(info,ensure_ascii=False,indent=2))
        (out/(key+".rendered.dom.html")).write_text(page.content())
        page.screenshot(path=str(out/(key+".destination.png")))
        print("CURRENT NECESSARY BROWSER READ",json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    (out/"navigation-log.json").write_text(json.dumps(steps,ensure_ascii=False,indent=2))
    browser.close()
