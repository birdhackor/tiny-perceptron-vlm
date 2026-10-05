from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

root=Path.cwd();out=Path(__file__).parent;notes=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1200,'height':850},device_scale_factor=1,reduced_motion='reduce')
    page.set_default_timeout(5000)
    page.goto('http://127.0.0.1:8788/training.html#T.8',wait_until='networkidle')
    heading=page.locator('[id="T.8"]');heading.scroll_into_view_if_needed()
    notes.append({'preview_url':page.url,'heading':heading.inner_text(),'direct_T8_svg_references':[]})
    page.screenshot(path=str(out/'preview-T.8.png'))
    for name in ['flash_allocated_memory','cache','architecture_gqa_cache','normalization','rope']:
        src=root/f'course/figures/{name}.svg'
        page.goto(src.as_uri(),wait_until='load')
        svg=page.locator('svg');svg.screenshot(path=str(out/f'{name}.png'))
        notes.append({'source_path':str(src.relative_to(root)),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'render_path':str((out/f'{name}.png').relative_to(root)),'dimensions':svg.bounding_box()})
    print(json.dumps(notes,ensure_ascii=False))
    browser.close()
