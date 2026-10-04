import json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
p=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/training-course/round2');r=[]
with sync_playwright() as w:
 b=w.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=b.new_page()
 for name in ['course','glossary','chapter-01']:
  response=page.goto('http://127.0.0.1:8770/'+name+'.html',wait_until='networkidle');raw=response.body();local=Path('outputs/site-reader-executed-v4/'+name+'.html').read_bytes();(p/(name+'.navigation.http-response.html')).write_bytes(raw)
  r.append({'page_id':name,'url':page.url,'status':response.status,'h1':page.locator('article h1').all_inner_texts(),'http_response_snapshot':name+'.navigation.http-response.html','http_response_sha256':hashlib.sha256(raw).hexdigest(),'http_bytes':len(raw),'local_build_file':'outputs/site-reader-executed-v4/'+name+'.html','local_sha256':hashlib.sha256(local).hexdigest(),'http_local_exact_byte_match':raw==local,'scope':'raw response preservation after actual click destinations already checked in browser-receipt.json; not full article read'})
 b.close()
(p/'browser-navigation-http-byte-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
