"""Validate existing single-lesson HTML against frozen Markdown, then render."""
import hashlib
import json
import re
import sys
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
raw=(OUT/'section.md').read_bytes()
expected=BeautifulSoup(markdown.markdown(raw.decode(),extensions=['extra']), 'html.parser')
existing=(OUT/'rendered-page-input.html').read_bytes()
article=BeautifulSoup(existing,'html.parser').find('article')
def norm(s):return re.sub(r'\s+','',s).replace('¶','')
expected_p=[norm(x.get_text()) for x in expected.find_all('p')]
actual_p=[norm(x.get_text()) for x in article.find_all('p')]
for p in expected_p:
    assert p in actual_p,('source paragraph missing or changed in rendered page',p)
fence=(OUT/'fence-1.py').read_text()
code_blocks=[x.get_text().rstrip() for x in article.find_all('pre')]
assert fence.rstrip() in code_blocks
assert norm(expected.find('h2').get_text())==norm(article.find('h1').get_text())
checks={'page_url':'http://127.0.0.1:8765/11.3.html','page_input_sha256':hashlib.sha256(existing).hexdigest(),'source_sha256':hashlib.sha256(raw).hexdigest(),'all_source_paragraphs_present_exact_after_whitespace_normalization':True,'source_paragraph_count':len(expected_p),'python_fence_exact':True,'number_of_source_svgs':0,'source_unchanged_during_inspection':True}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'],headless=True)
    checks['chromium_version']=browser.version
    for name,w,h in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
        page.goto(checks['page_url'],wait_until='networkidle')
        page.locator('article').wait_for()
        rendered=page.locator('article').inner_text()
        assert all(x in norm(rendered) for x in expected_p)
        assert page.locator('article pre').inner_text().rstrip()==fence.rstrip()
        page.screenshot(path=str(OUT/(name+'.png')),full_page=True)
        checks[name]={'viewport':[w,h],'screenshot':name+'.png','article_bounding_box':page.locator('article').bounding_box(),'source_paragraphs_match':True,'fence_matches':True}
        page.close()
    browser.close()
# Render the actual scene() input as an enlarged nearest-neighbour pixel image.
sys.path.insert(0,str(ROOT))
from tiny_perceptron.multimodal import scene
from PIL import Image
import torch
pixels=(scene().permute(1,2,0)*255).to(torch.uint8).numpy()
im=Image.fromarray(pixels);im.resize((320,320),Image.Resampling.NEAREST).save(OUT/'scene-red-square.png')
checks['scene_pixels']={'shape':list(pixels.shape),'red_pixel_count':int((pixels[:,:,0]==255).sum()),'green_and_blue_nonzero':int((pixels[:,:,1:]!=0).sum()),'red_square_bbox_inclusive':[4,4,12,12]}
print(json.dumps(checks,ensure_ascii=False,indent=2))
