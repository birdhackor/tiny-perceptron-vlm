from pathlib import Path
import json,hashlib,platform,importlib.metadata
from playwright.sync_api import sync_playwright
root=Path.cwd();prefix=root/'docs/technical-reviews/artifacts/natural-final-fact-20.7-';svg=root/'course/figures/natural_shared_chat.svg';results=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox"]);version=b.version
 for width in [720,358]:
  page=b.new_page(viewport={'width':width,'height':1100},device_scale_factor=1)
  page.set_content('<html><head><style>body{margin:0}svg{width:100%;height:auto;display:block}</style></head><body>'+svg.read_text()+'</body></html>')
  page.evaluate('document.fonts.ready')
  fn=Path(str(prefix)+f'render-{width}.png');page.locator('svg').screenshot(path=str(fn));results.append({'width':width,'path':str(fn.relative_to(root)),'sha256':hashlib.sha256(fn.read_bytes()).hexdigest(),'bounding_box':page.locator('svg').bounding_box()});page.close()
 b.close()
res={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-final-fact-20.7-render.py','environment':{'python':platform.python_version(),'playwright':importlib.metadata.version('playwright'),'chromium':version,'device':'CPU SVG rasterization'},'figure_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'renders':results,'scope':'Prerequisite 20.1/12.14 common diagram; 20.7 has no direct SVG. Visual inspection is a separate recorded act, not implied by screenshot exit.'}
Path(str(prefix)+'render.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
