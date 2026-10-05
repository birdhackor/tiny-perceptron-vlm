"""Actual navigation recheck of the immutable numbered 16.1 page."""
import json
import urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1365,'height':1000}, reduced_motion='reduce')
    response = page.goto('http://127.0.0.1:8789/16.1.html', wait_until='networkidle')
    link = page.get_by_role('link', name='T.8的效率量測方法', exact=True)
    assert response.status == 200
    assert link.count() == 1
    before = {'source_url':page.url, 'http_status':response.status,
              'link_text':link.inner_text(), 'href':link.get_attribute('href')}
    assert 'training.html#選讀效率實驗的量測條件' in urllib.parse.unquote(before['href'])
    link.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT/'browser-source-link.png'))
    link.click()
    page.wait_for_load_state('networkidle')
    destination = page.evaluate('''() => {
      const id=decodeURIComponent(location.hash.slice(1));
      const h=document.getElementById(id);
      if (!h) return {id,found:false};
      const chunks=[];let n=h;
      while(n){
        if(n!==h && n.matches && n.matches('h1,h2,h3'))break;
        chunks.push(n.innerText || '');n=n.nextElementSibling;
      }
      return {id,found:true,tag:h.tagName,heading:h.innerText,
              text:chunks.join('\\n'),top:h.getBoundingClientRect().top};
    }''')
    assert destination['found'] and destination['tag']=='H3'
    assert '選讀：效率實驗的量測條件' in destination['heading']
    assert '量九次取中位數' in destination['text']
    assert '未使用的reserved空間' in destination['text']
    assert '另一個獨立程序' in destination['text']
    assert 0 <= destination['top'] < 1000
    page.screenshot(path=str(OUT/'browser-target-method.png'))
    receipt = {'action':'Actually opened the specific numbered 16.1 page and clicked the exact T.8 measurement-method link.',
               'before':before,'after_url':page.url,'destination':destination,
               'browser_version':browser.version,'viewport':{'width':1365,'height':1000},
               'scope':'Navigation on immutable preview160fd47dede5c2453c8a08dd535290ddfd0b915d; no full-book browsing or GPU execution.'}
    (OUT/'browser-navigation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    browser.close()
