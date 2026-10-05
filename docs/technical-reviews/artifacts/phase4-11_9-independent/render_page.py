"""Render only the served 11.9 page with the installed system Chromium."""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
        args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking'])
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport=dict(width=width,height=height),device_scale_factor=1)
        response=page.goto('http://127.0.0.1:8765/11.9.html',wait_until='domcontentloaded',timeout=20000)
        page.locator('article img').wait_for(state='visible',timeout=10000)
        page.screenshot(path=str(OUT/f'page-{label}.png'),timeout=20000)
        print(label,response.status,page.title(),page.locator('article img').get_attribute('src'),
              page.locator('article img').bounding_box())
        page.close()
    browser.close()
