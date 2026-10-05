import json,hashlib,platform,importlib.metadata,urllib.request,subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path.cwd();out=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-environment-storage'
receipt={'environment':{'python':platform.python_version(),'playwright':importlib.metadata.version('playwright'),'browser':'/usr/bin/chromium'},'preview_url':'http://127.0.0.1:8787/','pages':[],'source_provenance':[]}
b=urllib.request.urlopen(receipt['preview_url']+'build-info.json').read();(out/'preview-build-info.json').write_bytes(b);j=json.loads(b);receipt['build_info']=j
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);receipt['environment']['chromium']=browser.version
 page=browser.new_page(viewport={'width':1400,'height':1000})
 for name in ['environment','asset-storage']:
  url=receipt['preview_url']+name+'.html';response=page.goto(url,wait_until='networkidle');article=page.locator('article').first
  txt=article.inner_text();(out/(name+'-visible.txt')).write_text(txt+'\n');(out/(name+'-preview.html')).write_bytes(response.body())
  article.screenshot(path=str(out/(name+'-browser.png')))
  record={'url':page.url,'status':response.status,'title':page.title(),'visible_text_path':str((out/(name+'-visible.txt')).relative_to(root)),'screenshot_path':str((out/(name+'-browser.png')).relative_to(root)),'article_headings':article.locator('h1,h2').all_inner_texts(),'links':article.locator('a').evaluate_all('(links)=>links.map(a=>({text:a.textContent,href:a.href}))')}
  receipt['pages'].append(record)
  current=(root/('docs/'+name+'.md')).read_bytes();command=['git','show',j['revision']+':docs/'+name+'.md'];run=subprocess.run(command,capture_output=True);(out/(name+'-frozen-source.md')).write_bytes(run.stdout)
  receipt['source_provenance'].append({'source':'docs/'+name+'.md','command':command,'exit_code':run.returncode,'stderr':run.stderr.decode(),'current_sha256':hashlib.sha256(current).hexdigest(),'frozen_sha256':hashlib.sha256(run.stdout).hexdigest(),'byte_equal':current==run.stdout,'current_bytes':len(current),'frozen_bytes':len(run.stdout)})
 # An actual authored next-page link is clicked; destination visible meaning is captured without expanding its review scope.
 page.goto(receipt['preview_url']+'environment.html',wait_until='networkidle');page.locator('article a').filter(has_text='教材資產存放').click();receipt['click']={'from':'environment.html','label':'教材資產存放','url':page.url,'visible_h1':page.locator('article h1').inner_text()}
 browser.close()
(out/'browser-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
