import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parent
preview = 'http://127.0.0.1:8789'
anchor = '選讀效率實驗的量測條件'

def section_text(heading):
    return heading.evaluate(r'''h => {
        let n=h, pieces=[];
        do {pieces.push(n.innerText); n=n.nextElementSibling;}
        while (n && n.tagName!=='H2');
        return pieces.join('\n\n');
    }''')

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1000})
    documents=[]
    page.on('response',lambda r: documents.append({'url':r.url,'status':r.status}) if r.request.resource_type=='document' else None)
    initial = page.goto(preview+'/training.html#'+anchor,wait_until='networkidle')
    method = page.locator('[id="'+anchor+'"]')
    assert method.count()==1
    method.scroll_into_view_if_needed()
    initial_url=page.url
    initial_heading=method.inner_text()
    initial_y=method.bounding_box()['y']
    architecture=page.locator('[id="T.8"]')
    assert architecture.count()==1
    (out/'preview-current-T.8-rendered-text.txt').write_text(section_text(architecture)+'\n')
    page.screenshot(path=str(out/'preview-current-method-direct-target.png'))
    link16=page.get_by_role('link',name='16.1',exact=True)
    href16=link16.get_attribute('href')
    link16.click()
    page.wait_for_load_state('networkidle')
    h16=page.locator('[id="16.1"]')
    assert h16.count()==1
    h16.scroll_into_view_if_needed()
    destination16=page.url
    heading16=h16.inner_text()
    (out/'preview-current-16.1-rendered-text.txt').write_text(section_text(h16)+'\n')
    methodlink=page.get_by_role('link',name='T.8的效率量測方法',exact=True)
    method_href=methodlink.get_attribute('href')
    methodlink.scroll_into_view_if_needed()
    page.screenshot(path=str(out/'preview-16.1-current-method-link.png'))
    methodlink.click()
    page.wait_for_load_state('networkidle')
    method=page.locator('[id="'+anchor+'"]')
    assert method.count()==1
    actual_method_url=page.url
    destination_y=method.bounding_box()['y']
    destination_heading=method.inner_text()
    destination_text=section_text(method)
    (out/'preview-method-link-destination-text.txt').write_text(destination_text+'\n')
    page.screenshot(path=str(out/'preview-method-link-actual-destination.png'))
    back=page.get_by_role('link',name='本節入口',exact=True)
    back_href=back.get_attribute('href')
    back.click()
    architecture=page.locator('[id="T.8"]')
    architecture_y=architecture.bounding_box()['y']
    architecture_text=architecture.inner_text()
    pre=architecture.evaluate(r'''h => {
        let n=h.nextElementSibling;
        while(n && n.tagName!=='H2') {
            const pre=n.querySelector('pre');
            if(pre) {pre.setAttribute('data-t8-current-first-pre','true');return true;}
            n=n.nextElementSibling;
        }
        return false;
    }''')
    assert pre
    code=page.locator('[data-t8-current-first-pre="true"] code')
    code.scroll_into_view_if_needed()
    scroll=code.evaluate('''e => {e.scrollLeft=e.scrollWidth;return {
        scrollLeft:e.scrollLeft,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,
        overflowX:getComputedStyle(e).overflowX};}''')
    code.screenshot(path=str(out/'preview-current-command-scrolled-right.png'))
    result={
        'browser':'actual Chromium via existing Playwright','browser_version':browser.version,
        'requested_initial_target':preview+'/training.html#'+anchor,
        'initial_document_status':initial.status,'initial_actual_url':initial_url,
        'initial_method_heading':initial_heading,'initial_heading_y':initial_y,
        'architecture_T.8_heading_retained':architecture.count()==1,
        'T.8_to_16.1_actual_click':{'link_text':'16.1','href':href16,'actual_url':destination16,'actual_heading':heading16},
        '16.1_method_link_actual_click':{'link_text':'T.8的效率量測方法','href':method_href,
            'actual_url':actual_method_url,'destination_heading':destination_heading,'destination_heading_y':destination_y},
        'method_body_back_to_architecture_actual_click':{'link_text':'本節入口','href':back_href,
            'actual_url':page.url,'destination_heading':architecture_text,'destination_heading_y':architecture_y},
        'actual_current_first_command_CODE_horizontal_scroll':scroll,
        'document_responses':documents,
        'screenshots':['preview-current-method-direct-target.png','preview-16.1-current-method-link.png','preview-method-link-actual-destination.png','preview-current-command-scrolled-right.png'],
        'destination_body_saved':'preview-method-link-destination-text.txt',
        'training_download_GPU_commands_executed':False,
    }
    (out/'browser-method-navigation-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    browser.close()
print(json.dumps(result,ensure_ascii=False,indent=2))
