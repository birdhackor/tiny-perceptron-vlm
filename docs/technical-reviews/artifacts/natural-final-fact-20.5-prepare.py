import base64
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/technical-reviews/artifacts'
PREFIX = 'natural-final-fact-20.5-'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
raw = (ROOT / 'course/chapters/20.md').read_text()
section = raw[raw.index('## 20.5'):raw.index('## 20.6')]
(OUT / (PREFIX+'section.md')).write_text(section)
(OUT / (PREFIX+'prerequisites.md')).write_text(raw[raw.index('## 20.3'):raw.index('## 20.5')])
manifest = json.loads((ROOT / 'docs/natural-assistant/manifest.json').read_text())
images = {}
for split in ['test', 'validation']:
    rows = [r for r in manifest['rows'] if r['split']==split and r['task']=='scene']
    unique = list(dict.fromkeys(r['image'] for r in rows))
    sheet = Image.new('RGB',(1200,4*340),'white')
    d = ImageDraw.Draw(sheet)
    for i, name in enumerate(unique):
        p=ROOT/'data/natural'/name
        img=Image.open(p)
        thumb=ImageOps.contain(img,(398,310))
        x=(i%3)*400; y=(i//3)*340
        sheet.paste(thumb,(x,y+25)); d.text((x+5,y+5),Path(name).stem,fill='black')
        images[name]={'sha256':sha(p),'bytes':p.stat().st_size,'size':img.size}
    sheet.save(OUT/(PREFIX+split+'-contact.png'))
    records=json.loads((ROOT/f'docs/natural-assistant/evidence/{"final" if split=="test" else "validation"}/generations.json').read_text())
    text=[]
    for r in rows:
        text.append('\nID '+r['id']+' Q '+r['user']+' REF '+r['answer'])
        for g in records:
            if g['id']==r['id']:
                text.append(g['variant']+' stop='+g['stop_reason']+' tokens='+str(g['generated_tokens'])+'\n'+g['prediction'])
    (OUT/(PREFIX+split+'-raw-read.txt')).write_text('\n'.join(text))
sources = {
    'python-sets': 'https://docs.python.org/3.13/library/stdtypes.html#set-types-set-frozenset',
    'docci-website': 'https://raw.githubusercontent.com/google/docci/c190366529943195d90c141899abb2ec68dc345d/index.html',
    'docci-paper': 'https://arxiv.org/pdf/2404.19753',
    'transformers-stop': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/generation/stopping_criteria.py',
    'transformers-config': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/generation/configuration_utils.py',
    'cc-by': 'https://creativecommons.org/licenses/by/4.0/legalcode.en',
}
fetch=[]
for name,url in sources.items():
    try:
        with urllib.request.urlopen(url,timeout=30) as response: data=response.read()
        p=OUT/(PREFIX+name+('.pdf' if name=='docci-paper' else '.txt'))
        p.write_bytes(data); fetch.append({'id':name,'url':url,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':len(data),'status':'retrieved'})
    except Exception as e: fetch.append({'id':name,'url':url,'status':'failed','error':str(e)})
render=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for name in ['natural_photo_evidence','natural_actual_bike','natural_actual_glass']:
        source=ROOT/'course/figures'/(name+'.svg')
        svg=source.read_text()
        embedded=re.search(r'data:image/jpeg;base64,([^"\s]+)',svg)
        proof=None
        if embedded:
            data=base64.b64decode(embedded[1]); original='test_00729' if name.endswith('bike') else 'test_01103'
            imgpath=ROOT/'data/natural/vision/images'/(original+'.jpg')
            assert data==imgpath.read_bytes()
            proof={'embedded_original_jpeg_equal':True,'sha256':sha(imgpath),'bytes':len(data)}
        html='<!doctype html><meta charset="utf-8"><style>body{margin:0}svg{width:100%;height:auto;display:block}</style>'+svg
        for width in [720,358]:
            page=browser.new_page(viewport={'width':width,'height':1500})
            page.set_content(html); page.evaluate('document.fonts.ready')
            png=OUT/(PREFIX+name+'-'+str(width)+'.png')
            page.locator('svg').screenshot(path=str(png))
            bounds=page.evaluate('''() => {const svg=document.querySelector('svg'); const v=svg.viewBox.baseVal;return [...svg.querySelectorAll('text')].map(t=>{const b=t.getBBox();return {text:t.textContent,x:b.x,y:b.y,width:b.width,height:b.height,inside:b.x>=v.x&&b.y>=v.y&&b.x+b.width<=v.x+v.width&&b.y+b.height<=v.y+v.height}})}''')
            assert all(t['inside'] for t in bounds)
            render.append({'figure':str(source.relative_to(ROOT)),'sha256':sha(source),'width':width,'browser':browser.version,'png':str(png.relative_to(ROOT)),'png_sha256':sha(png),'text_bounds':bounds,'jpeg_proof':proof})
            page.close()
    browser.close()
result={'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'image_inputs':images,'sources':fetch,'renders':render,'python':sys.version}
(OUT/(PREFIX+'prepare.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'section_sha256':result['section_sha256'],'source_status':[(s['id'],s['status']) for s in fetch],'render_count':len(render)},ensure_ascii=False))
