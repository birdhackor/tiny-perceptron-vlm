from pathlib import Path
import json, hashlib, re
from playwright.sync_api import sync_playwright, TimeoutError
base=Path(__file__).parent
out=base/'browser'
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    def destination(name,parent,label,section=None,limit=None,selector=None):
        page.goto('http://127.0.0.1:8786/'+parent,wait_until='networkidle')
        a=page.locator('article a').filter(has_text=label).first if selector is None else page.locator(selector).first
        exact=a.inner_text(); dest=a.get_attribute('href'); response=None
        try:
            with page.expect_navigation(wait_until='domcontentloaded',timeout=20000) as nav:
                a.click(timeout=10000)
            response=nav.value
            page.wait_for_load_state('networkidle',timeout=10000)
            status=response.status if response else None
            article=page.locator('article').first
            if section:
                content=page.evaluate('''(section)=>{const hs=[...document.querySelectorAll('article h2')];const h=hs.find(h=>h.innerText.startsWith(section));if(!h)return 'SECTION NOT FOUND'; const r=document.createRange();r.setStartBefore(h);const n=hs[hs.indexOf(h)+1];if(n)r.setEndBefore(n);else r.setEndAfter(document.querySelector('article').lastElementChild);return r.toString();}''',section)
                page.locator('article h2').filter(has_text=section).first.scroll_into_view_if_needed()
            elif article.count():
                content=article.inner_text()
            else:
                content=page.locator('body').inner_text()
            if limit:
                content='\n'.join(content.splitlines()[:limit])
            (out/(name+'.txt')).write_text(content,encoding='utf-8')
            (out/(name+'.html')).write_text(page.content(),encoding='utf-8')
            page.screenshot(path=str(out/(name+'.png')))
            records.append({'name':name,'from_url':'http://127.0.0.1:8786/'+parent,'actual_visible_label':exact,'link_href':dest,'actual_url':page.url,'status':status,'title':page.title(),'read_scope':section or ('first '+str(limit)+' lines' if limit else 'complete article'),'screenshot':name+'.png','images_on_page':article.locator('img').evaluate_all('(els)=>els.map(x=>({alt:x.alt,src:x.src}))') if article.count() else []})
            print('DESTINATION '+name+' '+page.url+'\n'+content+'\n')
        except Exception as e:
            records.append({'name':name,'from_url':'http://127.0.0.1:8786/'+parent,'actual_visible_label':exact,'link_href':dest,'actual_url':page.url,'error':str(e),'read_scope':'destination unavailable'})
            print('DESTINATION ERROR '+name+' '+str(e))
    destination('chapter19-entry','validation.html','第19章')
    destination('chapter20-entry','validation.html','第20章')
    destination('20-8-selection','validation.html','20.8')
    destination('20-12-asr','validation.html','20.12')
    destination('20-13-capability','validation.html','能力卡')
    destination('W1-setup','validation.html','W.1',section='W.1')
    destination('training-entry','validation.html','完整操作指引',limit=14)
    destination('reading-guide','publishing.html','閱讀指南',limit=66)
    destination('asset-storage','publishing.html','教材資產存放')
    destination('pages-home','publishing.html','GitHub Pages',limit=24)
    page.goto('http://127.0.0.1:8786/build-info.json',wait_until='networkidle')
    text=page.locator('body').inner_text()
    (out/'preview-build-info.txt').write_text(text,encoding='utf-8')
    records.append({'name':'preview-build-info','actual_url':page.url,'read_scope':'current local preview metadata, manually entered URL named in publishing; not a release claim','content':text})
    (out/'navigation-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    browser.close()
