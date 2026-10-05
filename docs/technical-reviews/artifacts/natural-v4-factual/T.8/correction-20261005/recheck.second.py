"""Owner correction recheck: exact byte scope, retained evidence, real preview navigation."""
from pathlib import Path
import hashlib,json,platform,re
from urllib.parse import unquote,urlsplit
from playwright.sync_api import sync_playwright
import torch

ROOT=Path.cwd()
OUT=Path(__file__).parent
OLD=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/T.8'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((OUT/'before-report.json').read_text())
before=(OUT/'before-section.raw.md').read_bytes()
current=(OUT/'current-section.raw.md').read_bytes()
assert current==before.replace('**選讀：效率實驗的量測條件。** '.encode(),'### 選讀：效率實驗的量測條件\n\n'.encode(),1)
before16=(OUT/'before-prerequisite-16.1.md').read_bytes()
current16=(OUT/'current-prerequisite-16.1.md').read_bytes()
assert current16==before16.replace(b'../training.md#T.8',b'../training.md#'+ '選讀效率實驗的量測條件'.encode(),1)
assert digest(OUT/'current-section.raw.md')=='142ff3eb9bbcc919f9229a76ca6043f99737f3b0bca9c04650de49bdb27ebd3e'
assert digest(OUT/'current-prerequisite-16.1.md')=='dcdf00e0de3a37f8f323b9677119b2cfbfab4400a07e6c421ffce4d9ddba8340'
checks={'repository_sources':[],'prior_artifacts':[],'figures':[],'prerequisites':[],'local_original_authorities':[]}
for source in report['sources']:
    if source['kind']=='repository_code':
        actual=digest(ROOT/source['path']);assert actual==source['sha256'],source['path']
        checks['repository_sources'].append({'path':source['path'],'sha256':actual,'byte_hash_unchanged':True})
for artifact in report['artifacts']:
    actual=digest(ROOT/artifact['path']);assert actual==artifact['sha256'],artifact['path']
    checks['prior_artifacts'].append({'path':artifact['path'],'sha256':actual,'byte_hash_unchanged':True})
for path,expected in report['figure_sha256'].items():
    actual=digest(ROOT/path);assert actual==expected,path
    checks['figures'].append({'path':path,'sha256':actual,'byte_hash_unchanged':True,'prior_personal_view_reused':True})
for name,expected in report['prerequisite_sha256'].items():
    assert digest(OLD/name)==expected,name
    id=name.removeprefix('prerequisite-').removesuffix('.md')
    source=ROOT/('course/training.md' if id.startswith('T.') else f'course/chapters/{int(id.split(".")[0]):02d}.md')
    raw=source.read_bytes();start=raw.index(('## '+id+' ').encode());end=raw.find(b'\n## ',start+1);end=len(raw) if end<0 else end+1;section=raw[start:end]
    if id=='16.1':assert section==current16
    else:assert section==(OLD/name).read_bytes(),str(source)+'#'+id
    checks['prerequisites'].append({'source':str(source.relative_to(ROOT))+'#'+id,'old_sha256':expected,'current_sha256':hashlib.sha256(section).hexdigest(),'substance_unchanged':True,'navigation_changed':id=='16.1'})
# Complete originals stay in ignored research. Verify locally retained downloaded bytes against own retrieval SHA.
research=ROOT/'outputs/natural-v4/factual-research/T.8'
receipts=[]
for name in ['authority-retrieval.json','authority-source-fallback.json','authority-markdown-fallback.json']:
    receipts.extend(json.loads((OLD/name).read_text()))
for receipt in receipts:
    if 'sha256' not in receipt:continue
    if 'ignored_research_path' in receipt:p=ROOT/receipt['ignored_research_path']
    else:p=research/(receipt['id']+'.txt')
    if p.exists():assert digest(p)==receipt['sha256']
    checks['local_original_authorities'].append({'url':receipt['url'],'retrieval_sha256':receipt['sha256'],'local_original_still_present':p.exists(),'same_retrieved_bytes':digest(p)==receipt['sha256'] if p.exists() else None})
print('OWN ENVIRONMENT',json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),'cuda_available':torch.cuda.is_available()}))
print('BYTE SCOPE',json.dumps({'T.8':'Exactly one bold-label to h3 conversion; all factual body, code, numbers, links unchanged.','16.1':'Exactly one link fragment change; all factual body/code/numbers unchanged.'}))
print('UNCHANGED EVIDENCE',json.dumps({k:len(v) for k,v in checks.items()}))
(OUT/'reuse-byte-comparison.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1280,'height':960},device_scale_factor=1,reduced_motion='reduce')
    page.set_default_timeout(10000)
    response=page.goto('http://127.0.0.1:8789/chapters/16.html#16.1',wait_until='networkidle')
    assert response.status==200
    page.locator('[id="16.1"]').scroll_into_view_if_needed()
    link=page.get_by_role('link',name='T.8的效率量測方法',exact=True)
    href=link.get_attribute('href');assert unquote(urlsplit(href).fragment)=='選讀效率實驗的量測條件'
    link.scroll_into_view_if_needed();page.screenshot(path=str(OUT/'navigation-before-click.png'))
    link.click();page.wait_for_load_state('networkidle')
    assert unquote(urlsplit(page.url).fragment)=='選讀效率實驗的量測條件'
    target=page.locator('[id="選讀效率實驗的量測條件"]')
    assert target.count()==1 and target.evaluate('(e)=>e.tagName')=='H3'
    assert target.inner_text().replace('¶','').strip()=='選讀：效率實驗的量測條件'
    paragraphs=target.evaluate('(e)=>{let a=[],n=e.nextElementSibling;while(n&&n.tagName!=="H2"&&n.tagName!=="H3"){a.push(n.innerText);n=n.nextElementSibling;}return a;}')
    assert paragraphs[0].startswith('固定配方採L4、PyTorch 2.14.1+cu126、FP32，關閉TF32。')
    combined='\n'.join(paragraphs)
    for phrase in ['六支主要訓練','未使用的reserved空間','1 MiB是2²⁰ bytes','量九次取中位數','略過前三步']:
        assert phrase in combined,phrase
    heading_box=target.bounding_box();assert 0<=heading_box['y']<250,heading_box
    page.screenshot(path=str(OUT/'navigation-destination.png'))
    nav={'start_url':'http://127.0.0.1:8789/chapters/16.html#16.1','href':href,'clicked':True,'final_url':page.url,'target_tag':'H3','target_text':target.inner_text(),'target_viewport_box':heading_box,'destination_paragraphs':paragraphs,'semantic_inspection':'Destination begins immediately at fixed L4/software/FP32/TF32/SFT conditions, then six-branch allocator scope, exclusions, MiB and warmup/synchronization. It correctly answers the 16.1 measurement-method link.'}
    (OUT/'browser-navigation.json').write_text(json.dumps(nav,ensure_ascii=False,indent=2)+'\n')
    print('REAL NAVIGATION',json.dumps({k:v for k,v in nav.items() if k!='destination_paragraphs'},ensure_ascii=False))
    print('DESTINATION RAW RENDERED TEXT','\n'.join(paragraphs[:5]))
    browser.close()
print('FINAL','All correction byte-scope, retained-evidence and actual navigation checks passed; prior CPU/GPU-record/figure evidence reused without claiming new GPU execution.')
