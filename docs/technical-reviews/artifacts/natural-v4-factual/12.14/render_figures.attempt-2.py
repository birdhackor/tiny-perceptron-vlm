from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
 page=browser.new_page(viewport={'width':800,'height':1080},device_scale_factor=1)
 for name in ['natural_shared_chat.svg','natural-v4-asr-two-routes.svg']:
  source=ROOT/'course/figures'/name
  page.goto(source.as_uri(),wait_until='load')
  page.locator('svg').screenshot(path=str(ART/name.replace('.svg','.png')))
  print(json.dumps({'source':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'render':str((ART/name.replace('.svg','.png')).relative_to(ROOT)),'chromium':browser.version}))
 browser.close()
