from pathlib import Path
from playwright.sync_api import sync_playwright
import json,hashlib,sys
root=Path.cwd()
out=root/'docs/technical-reviews/artifacts/natural-v4-factual/20.5'
rows=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1200,'height':1200},device_scale_factor=1)
    for name in ['natural-v4-family-crops','natural-v4-training-cat']:
        source=Path('course/figures')/(name+'.svg')
        page.set_content(source.read_text(),wait_until='load')
        page.locator('svg').screenshot(path=str(out/(name+'.png')))
        rows.append({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'render':str(out.relative_to(root)/(name+'.png'))})
    browser.close()
print(json.dumps({'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/20.5/render_figures.py','environment':{'python':sys.version.split()[0],'renderer':'system Chromium /usr/bin/chromium, Playwright','device':'CPU'},'figures':rows},indent=2))
