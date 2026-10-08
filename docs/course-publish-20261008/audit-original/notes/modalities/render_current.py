import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/workspace/work/tutorial-audit-20261008')
src=root/'freeze/original'/sys.argv[1]
svg=src.read_text()
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox'])
    for w in (640,360):
        page=b.new_page(viewport={'width':w,'height':1600},device_scale_factor=1)
        page.set_content('<style>body{margin:0}svg{width:100%;height:auto;display:block}</style>'+svg)
        page.evaluate('async()=>{await document.fonts.ready}')
        page.locator('svg').screenshot(path=str(root/'renders'/f'{src.stem}-{w}.png'))
        page.close()
    b.close()
