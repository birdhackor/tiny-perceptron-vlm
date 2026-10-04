from pathlib import Path
from playwright.sync_api import sync_playwright
import hashlib,json
ROOT=Path('/workspace/tiny-perceptron-vlm');OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/training-course/round2';BUILD=ROOT/'outputs/site-reader-executed-v4';BASE='http://127.0.0.1:8770/'
def sha(b):return hashlib.sha256(b).hexdigest()
def receipt_response(response,path,key):
 raw=response.body();local=(BUILD/path).read_bytes();name=key+'.http-response.html';(OUT/name).write_bytes(raw)
 return {'url':response.url,'status':response.status,'http_response_snapshot':name,'http_response_bytes':len(raw),'http_response_sha256':sha(raw),'local_build_file':str((BUILD/path).relative_to(ROOT)),'local_build_sha256':sha(local),'http_local_exact_byte_match':raw==local}
r={'reviewer_task':'/root/v4_review_coordinator/reader_whole_training_course','round':2,'browser':'existing .venv Playwright Chromium','executable_path':'/usr/bin/chromium','reported_build_revision':'a45d1d8 (provided by coordinator; no Git inspected)','base':BASE,'section_clicks':[],'article_tiles':[],'navigation':[],'guides':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1);page.set_default_timeout(6000)
 response=page.goto(BASE+'training.html',wait_until='networkidle');r['training_response']=receipt_response(response,'training.html','training')
 article=page.locator('article');visible=article.inner_text();(OUT/'training.rendered-visible-article.txt').write_text(visible);rendered=page.content().encode();(OUT/'training.rendered-dom.html').write_bytes(rendered)
 r['training_article']={'url':page.url,'h1':page.locator('h1').all_inner_texts(),'visible_text_sha256':sha(visible.encode()),'rendered_dom_sha256':sha(rendered),'paragraph_count':article.locator('p').count(),'table_count':article.locator('table').count(),'code_block_count':article.locator('pre').count(),'svg_img_count':article.locator('img[src$=".svg"]').count(),'visible_text_exact_match_round1':visible==(OUT.parent/'round1-preserved-for-round2/training.browser-article-complete.txt').read_text()}
 output_nodes=article.locator('[class*="output"]')
 outputs=[{'class':x.get_attribute('class'),'visible':x.is_visible(),'text':x.inner_text()} for x in output_nodes.all()]
 r['visible_cpu_output_nodes']=[x for x in outputs if x['visible']]
 (OUT/'training.visible-output-nodes.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
 page.screenshot(path=str(OUT/'training-top.png'))
 for section in ['T.1','T.2','T.3','T.4','T.5','T.6','T.7','T.8','T.9','T.10','T.11']:
  link=page.locator(f'a[href$="#{section}"]').filter(visible=True).first;label=link.inner_text().strip();href=link.get_attribute('href');link.click();heading=page.locator(f'[id="{section}"]');shot=section.replace('.','-')+'.png';page.screenshot(path=str(OUT/shot));r['section_clicks'].append({'section_id':section,'visible_link_label':label,'href':href,'actual_destination':page.url,'heading':heading.inner_text(),'screenshot':shot})
 page.set_viewport_size({'width':1440,'height':2700});page.evaluate('window.scrollTo(0,0)');bounds=article.bounding_box();height=bounds['height'];r['article_height']=height
 for i,off in enumerate(range(0,int(height)+1,2400),1):
  if height-off<=0:break
  shot=f'training-tile-{i:02}.png';page.evaluate('(y)=>window.scrollTo(0,y)',bounds['y']+off-96);page.screenshot(path=str(OUT/shot),clip={'x':bounds['x'],'y':96,'width':bounds['width'],'height':2600});r['article_tiles'].append({'order':i,'article_offset':off,'actual_scroll_y':page.evaluate('window.scrollY'),'viewport_height':2700,'clip_y':96,'clip_height':2600,'screenshot':shot})
 # Natural guides: use the real visible chapter navigation, not directgoto substitution.
 page.set_viewport_size({'width':1440,'height':1100})
 for dest in ['natural-v4-student.html','natural-v4-data.html','natural-v4-training.html']:
  page.goto(BASE+'training.html',wait_until='networkidle');a=page.get_by_role('navigation',name='標籤頁').get_by_role('link',name='教材',exact=True);label=a.inner_text();a.click();page.wait_for_load_state('networkidle');r['navigation'].append({'from':BASE+'training.html','visible_label':label.strip(),'destination':page.url})
  page.locator('label[for="__nav_3_20"]').filter(has_text='第 20 章').first.click();a=page.locator(f'.md-sidebar--primary a[href$="{dest}"]');a.scroll_into_view_if_needed();label=a.inner_text().strip();href=a.get_attribute('href');visible_before=a.is_visible();page.screenshot(path=str(OUT/(dest[:-5]+'-navigation.png')))
  with page.expect_response(lambda x:x.url==BASE+dest and x.request.resource_type=='document') as response_info:a.click()
  page.wait_for_load_state('networkidle');response=response_info.value;rr=receipt_response(response,dest,dest[:-5]);content=page.locator('.md-content').inner_text();(OUT/(dest[:-5]+'.rendered-visible.txt')).write_text(content);(OUT/(dest[:-5]+'.rendered-dom.html')).write_text(page.content());page.screenshot(path=str(OUT/(dest[:-5]+'.png')))
  r['guides'].append({'from':BASE+'chapter-01.html','menu_opened':'第 20 章：做一位能看圖、讀字與聽問題的助理','visible_label':label,'visible_before_click':visible_before,'href':href,'actual_destination':page.url,'h1':page.locator('h1').all_inner_texts(),'opening':content.split('1.')[0],'read_scope':'landing heading and opening purpose, not full guide','response':rr,'screenshot':dest[:-5]+'.png'})
 # Start-reading navigation and actual previous/next destinations.
 page.goto(BASE+'training.html',wait_until='networkidle');a=page.get_by_role('navigation',name='標籤頁').get_by_role('link',name='開始閱讀',exact=True);label=a.inner_text().strip();a.click();page.wait_for_load_state('networkidle');r['navigation'].append({'from':BASE+'training.html','visible_label':label,'destination':page.url,'h1':page.locator('h1').all_inner_texts()});page.screenshot(path=str(OUT/'start-reading-route.png'))
 a=page.locator('.md-sidebar--primary a[href$="training.html"]').filter(visible=True).first;label=a.inner_text().strip();a.click();page.wait_for_load_state('networkidle');r['navigation'].append({'from':BASE+'course.html','visible_label':label,'destination':page.url,'h1':page.locator('h1').all_inner_texts()})
 for cls,key in [('prev','previous'),('next','next')]:
  page.goto(BASE+'training.html',wait_until='networkidle');a=page.locator('.md-footer__link--'+cls);a.scroll_into_view_if_needed();label=a.inner_text().strip();href=a.get_attribute('href');page.screenshot(path=str(OUT/('footer-'+key+'.png')));a.click();page.wait_for_load_state('networkidle');page.screenshot(path=str(OUT/('destination-'+key+'.png')));r['navigation'].append({'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'screenshot':'destination-'+key+'.png'})
 # Explicit prerequisite links required for this page's meaning.
 for needle,key in [('first-steps.html#W.1','W1'),('13.1.html','13-1'),('13.4.html','13-4'),('20.2.html','20-2')]:
  page.goto(BASE+'training.html',wait_until='networkidle');a=page.locator(f'article a[href*="{needle}"]').first;label=a.inner_text().strip();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle');path=page.url.split('/')[-1].split('#')[0]
  raw=page.request.get(BASE+path).body();local=(BUILD/path).read_bytes();(OUT/(key+'.http-response.html')).write_bytes(raw);content=page.locator('article').inner_text();(OUT/(key+'.rendered-visible.txt')).write_text(content);page.screenshot(path=str(OUT/(key+'.png')));r['navigation'].append({'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'http_local_exact_byte_match':raw==local,'http_sha256':sha(raw),'local_sha256':sha(local),'scope':'explicit prerequisite route and heading; necessary text/outputs assessed separately','screenshot':key+'.png'})
 # One real horizontal scroll through the long trainingcommand, preserving the command tail.
 page.goto(BASE+'training.html',wait_until='networkidle');code=page.locator('article pre code').filter(has_text='scripts/train_simple.py --model bigram --seed 42 --device cpu --train').first;code.scroll_into_view_if_needed();page.screenshot(path=str(OUT/'long-command-left.png'));scroll=code.evaluate('(el)=>{el.scrollLeft=el.scrollWidth;return {scrollLeft:el.scrollLeft,scrollWidth:el.scrollWidth,clientWidth:el.clientWidth,text:el.innerText};}');page.screenshot(path=str(OUT/'long-command-right.png'));r['long_command_scroll']=scroll
 browser.close()
(OUT/'browser-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'training':r['training_article'],'training_response':r['training_response'],'tiles':len(r['article_tiles']),'cpu_output_nodes':len(r['visible_cpu_output_nodes']),'guide_labels':[(x['visible_label'],x['actual_destination'],x['h1']) for x in r['guides']],'navigation':r['navigation'],'long_command_scroll':scroll},ensure_ascii=False,indent=2))
