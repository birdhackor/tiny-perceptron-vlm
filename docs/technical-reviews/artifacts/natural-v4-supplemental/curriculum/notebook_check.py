from pathlib import Path
import hashlib,json,platform
import nbformat,nbclient,ipykernel
from nbclient import NotebookClient
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1,accept_downloads=True);page=context.new_page();page.goto('http://127.0.0.1:8783/1.1.html',wait_until='domcontentloaded')
 link=page.locator('article a',has_text='下載Notebook');print('download candidates',link.count())
 if not link.count():link=page.locator('article a[href$="1.1.ipynb"]').filter(has_text='下載')
 with page.expect_download(timeout=15000) as got:link.first.click()
 download=got.value;path=HERE/'browser.downloaded-1.1.ipynb';download.save_as(str(path));receipt={'action':'actual click download Notebook','page':page.url,'download_url':download.url,'filename':download.suggested_filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'browser':browser.version}
 browser.close()
nb=nbformat.read(path,as_version=4);source_cells=[{'type':c.cell_type,'source':c.source} for c in nb.cells]
NotebookClient(nb,timeout=60,kernel_name='tiny-perceptron',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb,HERE/'notebook.executed-1.1.ipynb')
receipt['fresh_kernel_execution']={'kernel':'tiny-perceptron','code_cells':sum(c.cell_type=='code' for c in nb.cells),'execution_counts':[c.execution_count for c in nb.cells if c.cell_type=='code'],'outputs':[o.get('text',o.get('data',{})) for c in nb.cells if c.cell_type=='code' for o in c.outputs],'source_cells_unchanged':source_cells==[{'type':c.cell_type,'source':c.source} for c in nb.cells]}
receipt['environment']={'python':platform.python_version(),'nbclient':nbclient.__version__,'ipykernel':ipykernel.__version__,'device':'CPU','scope':'one 1.1 notebook; no Colab kernel or wholebook execution'}
(HERE/'notebook.receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
