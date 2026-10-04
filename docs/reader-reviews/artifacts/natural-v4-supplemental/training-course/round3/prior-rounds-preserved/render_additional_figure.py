from pathlib import Path
from playwright.sync_api import sync_playwright
import hashlib,json
root=Path('/workspace/tiny-perceptron-vlm');out=root/'docs/reader-reviews/artifacts/natural-v4-supplemental/training-course';source=root/'course/figures/architecture_distillation_temperature.svg';raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest();snapshot=out/(source.stem+'.'+h+'.svg');snapshot.write_bytes(raw)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1800,'height':1400});page.set_content(raw.decode(),wait_until='load');name=source.stem+'.png';page.locator('svg').screenshot(path=str(out/name));bounds=page.locator('svg').bounding_box();b.close()
r={'section':'18.5','source':str(source.relative_to(root)),'original_sha256':h,'source_snapshot':snapshot.name,'render':'existing Playwright Chromium /usr/bin/chromium set_content exact SVG UTF-8 bytes, SVG element screenshot','png':name,'png_sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),'svg_bounds':bounds};(out/'figure-additional-render-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2))
