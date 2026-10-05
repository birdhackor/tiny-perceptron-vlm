"""Render current 12.6 page with local requests only."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
root = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', args=['--no-sandbox','--disable-gpu','--disable-background-networking'])
    records=[]
    for width,height,name in [(1280,800,'desktop'),(390,844,'mobile')]:
        page = browser.new_page(viewport={'width':width,'height':height}, device_scale_factor=1)
        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith('http://127.0.0.1:8765/') else route.abort())
        page.goto('http://127.0.0.1:8765/12.6.html', wait_until='domcontentloaded',timeout=20000)
        page.wait_for_timeout(500)
        article = page.locator('article')
        text = article.inner_text()
        assert '相鄰頻率怎麼合成少量頻帶' in text
        assert '不是實際 16 帶的權重值' in text
        img=article.locator('img[src*="multimodal_overlapping_bands"]')
        assert img.count() == 1
        img.wait_for(state='visible')
        page.screenshot(path=str(root/'render'/f'{name}.png'),full_page=True)
        img.screenshot(path=str(root/'render'/f'{name}-figure.png'))
        records.append({'viewport':[width,height], 'image_natural_width':img.evaluate('(x)=>x.naturalWidth'), 'figure_box':img.bounding_box(), 'article_text_file':f'render/{name}-article.txt','screenshot':f'render/{name}.png'})
        (root/'render'/f'{name}-article.txt').write_text(text)
        page.close()
    (root/'render/browser.json').write_text(json.dumps({'chromium':browser.version,'url':'http://127.0.0.1:8765/12.6.html','external_requests':'aborted','render_records':records},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False,indent=2))
    browser.close()
