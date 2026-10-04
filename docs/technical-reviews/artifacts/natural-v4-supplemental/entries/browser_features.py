from pathlib import Path
import json
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
record={'preview':'http://127.0.0.1:8783/','limits':'Actual local Chromium actions; no Colab remote execution.'}
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 context=b.new_context(viewport={'width':1440,'height':1100},permissions=['clipboard-read','clipboard-write'])
 page=context.new_page();page.goto(record['preview']+'readme.html',wait_until='networkidle')
 viewport=page.locator('.diagram-viewport').first
 before=viewport.evaluate('(e)=>({width:e.clientWidth,scrollWidth:e.scrollWidth})')
 page.locator('button.diagram-zoom').click();page.screenshot(path=str(OUT/'diagram-zoom.png'))
 record['zoom']={'clicked':'button.diagram-zoom','aria_expanded':page.locator('button.diagram-zoom').get_attribute('aria-expanded'),'before':before,'after':viewport.evaluate('(e)=>({width:e.clientWidth,scrollWidth:e.scrollWidth})'),'visible_control':page.locator('button.diagram-zoom').inner_text()}
 page.keyboard.press('Escape')
 page.locator('label[for=__palette_1]').click();record['theme']={'scheme':page.locator('body').get_attribute('data-md-color-scheme')};page.screenshot(path=str(OUT/'dark-mode.png'))
 page.locator('label[for=__palette_0]').click()
 page.locator('button.md-search__button').click();search=page.get_by_placeholder('Search');search.fill('文字怎麼變成數字');page.wait_for_timeout(1000)
 record['search']={'query':'文字怎麼變成數字','visible_search_dialog':page.locator('[role=dialog]').inner_text()[:6000]};page.screenshot(path=str(OUT/'search-browser.png'))
 page.keyboard.press('Escape')
 copy=page.locator('button.md-code__button').first;copy.scroll_into_view_if_needed();copy.click()
 record['copy']={'clipboard':page.evaluate('navigator.clipboard.readText()')};assert 'git clone' in record['copy']['clipboard']
 page.screenshot(path=str(OUT/'copy-browser.png'))
 page.emulate_media(reduced_motion='reduce');record['reduced_motion']={'matches':page.evaluate('matchMedia("(prefers-reduced-motion: reduce)").matches')}
 b.close()
(OUT/'browser-features-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Actual zoom, dark/light, search, clipboard copy, and reduced-motion media query checked.')
