from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parent
url='http://127.0.0.1:8765/12.7.html'
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 records=[]
 for label,width,height in [('desktop',1280,1000),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  response=page.goto(url,wait_until='networkidle',timeout=30000)
  assert response.status==200
  page.locator('h1').wait_for()
  content=page.content()
  if label=='desktop':
   (root/'rendered-page.html').write_text(content)
   (root/'rendered-page.txt').write_text(BeautifulSoup(content,'html.parser').find('article').get_text('\n',strip=True))
  assert page.locator('h1').inner_text().rstrip(' ¶')=='12.7 功率相差很大，取 log 有什麼效果？'
  for sentence in ['負值只表示原功率小於 1，不表示能量為負。','若波形振幅乘 2，功率才乘 4，不能混用振幅與功率的倍數。','把頻率功率合成 mel 頻帶後再取 log，就得到 log-mel 特徵。']:
   assert sentence in page.locator('body').inner_text(),sentence
  assert 'power = torch.tensor([0.0, 1e-10, 1e-4, 1.0])' in page.locator('body').inner_text()
  assert 'compressed = power.clamp(min=floor).log()' in page.locator('body').inner_text()
  png=root/('page-'+label+'.png');page.screenshot(path=str(png),full_page=True)
  records.append({'viewport':[width,height],'url':url,'page_title':page.title(),'h1':page.locator('h1').inner_text(),'screenshot':png.name,'sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'browser_version':browser.version})
  page.close()
 browser.close()
(root/'render-provenance.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False))
