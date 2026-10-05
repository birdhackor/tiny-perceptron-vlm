from pathlib import Path
import json,subprocess
from playwright.sync_api import sync_playwright
root=Path.cwd();base=root/'docs/technical-reviews/artifacts/phase4-19_6-independent'
svg=base/'inputs/course/figures/rewrite-19-selected-text.svg'
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.set_content('<html><head><meta charset="utf-8"></head><body style="margin:0"><div style="max-width:640px;width:100%">'+svg.read_text().replace('<svg ', '<svg style="display:block;width:100%;height:auto" ', 1)+'</div></body></html>')
  page.locator('svg').screenshot(path=str(base/'figures'/f'{name}.png'))
 print(json.dumps({'browser':browser.version,'source':str(svg.relative_to(root)),'render':'inline SVG with responsive width; not entire course page','viewports':[[1280,800],[390,844]],'fonts':subprocess.check_output(['fc-match','Noto Sans CJK TC'],text=True).strip()},ensure_ascii=False))
 browser.close()
