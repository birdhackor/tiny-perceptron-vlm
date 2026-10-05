from pathlib import Path
import json,hashlib,shutil
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[5]
out=Path(__file__).resolve().parent
names=['cache','architecture_online_softmax','flash_allocated_memory','architecture_quantize_grid','ppo_clip','ppo_roles','ppo_dpo_routes','posttrain_signals','rag']
rows=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'],**({'executable_path':shutil.which('chromium')} if shutil.which('chromium') else {}))
 page=browser.new_page(viewport={'width':1200,'height':1000},device_scale_factor=1)
 for name in names:
  svg=root/f'course/figures/{name}.svg';page.set_content(svg.read_text());page.locator('svg').screenshot(path=str(out/(name+'.png')))
  rows.append({'source':str(svg.relative_to(root)),'source_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'render':str((out/(name+'.png')).relative_to(root))})
 page.goto('http://127.0.0.1:8788/glossary.html#G.4');page.locator('#G\\.4').scroll_into_view_if_needed()
 page.screenshot(path=str(out/'preview-G.4.png'),full_page=False)
 browser.close()
(out/'figure-render-receipt.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
