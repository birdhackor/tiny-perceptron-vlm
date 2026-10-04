from pathlib import Path
from playwright.sync_api import sync_playwright
import json
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1100})
 page.goto('http://127.0.0.1:8769/training.html',wait_until='networkidle')
 links=page.get_by_role('link',name='教材',exact=True)
 print('visible 教材 count',links.count())
 for a in links.all():
  if a.is_visible(): print('click',a.get_attribute('href'));a.click();break
 page.wait_for_load_state('networkidle')
 print('url',page.url)
 print('visible sidebar',page.locator('.md-sidebar--primary').inner_text())
 print('natural ancestors',page.locator('a[href$="natural-v4-student.html"]').evaluate_all('(els)=>els.map(a=>({label:a.innerText,html:a.closest("li").parentElement.parentElement.outerHTML.slice(0,6000)}))'))
 page.screenshot(path=str(OUT/'guide-nav-before-expand.png'))
 b.close()
