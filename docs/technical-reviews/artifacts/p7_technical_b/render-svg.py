import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for width in [640,360]:
  page=browser.new_page(viewport={'width':width,'height':1000},device_scale_factor=1)
  page.set_content('<style>body{margin:0}svg{display:block;width:100%;height:auto}</style>'+source.read_text())
  name=out/f'{width}.png';page.locator('svg').screenshot(path=str(name));print(name);page.close()
 browser.close()

