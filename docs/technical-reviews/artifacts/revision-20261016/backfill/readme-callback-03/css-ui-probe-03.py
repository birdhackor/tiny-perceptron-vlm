"""Actual preview17 CSS dependency checks; no training, inference, scoring or remote publication."""
from datetime import datetime,UTC
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[6]
OUT=Path(__file__).parent
SCRATCH=ROOT/'work/readme-backfill-callback-03/attempt03'
SCRATCH.mkdir(parents=True,exist_ok=True)
URL='http://127.0.0.1:8896/readme.html'
def sha(raw):return hashlib.sha256(raw).hexdigest()
old_css=subprocess.run(['git','show','HEAD:course/web/course.css'],capture_output=True,check=True).stdout
current_css=(ROOT/'course/web/course.css').read_bytes()
prior=json.loads((ROOT/'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-02.json').read_bytes())
assert sha(old_css)==next(s['sha256'] for s in prior['sources'] if s['id']=='s_css')
r={'at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/tech_transport_readme_20261016','current_context':'informed','phase':'continuity-phase CSS dependency intake','preview':URL,'environment':{'python':platform.python_version(),'playwright':importlib.metadata.version('playwright'),'browser_executable':'/usr/bin/chromium','device':'headless browser on CPU; no model execution'},'launch_note':'The default Playwright bundled headless-shell binary was absent. Used the already installed /usr/bin/chromium explicitly; no packages, browser installation, dependencies or lockfiles changed.','checked_files':{},'css_diff':''.join(difflib.unified_diff(old_css.decode().splitlines(True),current_css.decode().splitlines(True),fromfile='callback02 identical HEAD CSS',tofile='current course/web/course.css')),'screenshots':{},'checks':[]}
for name in ('README.md','course/chapters/19.md','course/web/course.css','course/web/course.js','zensical.toml','course/figures/rewrite-01-character-ids.svg'):
 r['checked_files'][name]=sha((ROOT/name).read_bytes())
def capture(page,name):
 path=SCRATCH/(name+'.png');page.screenshot(path=str(path),full_page=False)
 r['screenshots'][name]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.read_bytes()),'viewport':page.viewport_size}
def inspect(page):
 return page.evaluate('''() => {
  const box=(selector)=>{const e=document.querySelector(selector);if(!e)return null;const b=e.getBoundingClientRect(),s=getComputedStyle(e);return {x:b.x,y:b.y,width:b.width,height:b.height,right:b.right,bottom:b.bottom,position:s.position,top:s.top,cssRight:s.right,visibility:s.visibility,display:s.display,opacity:s.opacity,marginRight:s.marginRight,zIndex:s.zIndex,pointerEvents:s.pointerEvents}};
  const title=document.querySelector('.md-header__title');
  return {scrollY:window.scrollY,mediaMobile:matchMedia('(max-width:59.984375em)').matches,header:box('.md-header'),title:box('.md-header__title'),sidebar:box('.md-sidebar--secondary'),button:box('.md-sidebar--secondary label.md-sidebar-button'),popup:box('.md-sidebar--secondary .md-sidebar__inner'),search:document.querySelector('#__search').checked,drawer:document.querySelector('#__drawer').checked,toc:document.querySelector('#__toc').checked,bodyWidth:document.body.scrollWidth,viewportWidth:innerWidth,titleText:title.textContent.trim()};
 }''')
def check(name,condition,details):
 r['checks'].append({'name':name,'passed':bool(condition),'details':details})

with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 r['environment']['chromium']=browser.version
 context=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
 page=context.new_page();response=page.goto(URL,wait_until='domcontentloaded',timeout=30000)
 page.locator('.md-sidebar--secondary label.md-sidebar-button').wait_for(state='visible')
 page.wait_for_timeout(250)
 css_url=page.locator('link[href*="course."][href$=".css"]').evaluate('(e)=>e.href')
 css_response=page.request.get(css_url);served=css_response.body()
 r['served_css']={'url':css_url,'status':css_response.status,'sha256':sha(served),'matches_current':served==current_css}
 check('preview17 serves the exact current CSS',response.status==200 and served==current_css,r['served_css'])
 r['mobile_closed_top']=inspect(page);capture(page,'mobile-closed-top')
 state=r['mobile_closed_top'];button=state['button']
 check('390px collapsed TOC is in fixed header at x237 y6',state['sidebar']['position']=='fixed' and abs(button['x']-237)<0.1 and abs(button['y']-6)<0.1 and button['bottom']<=state['header']['bottom']+0.1,state)
 check('mobile title reserves its 2.4rem slot',state['title']['marginRight']=='48px' and state['title']['right']<=button['x'],{'title':state['title'],'button':button})
 page.evaluate('window.scrollTo(0,900)');page.wait_for_timeout(150)
 r['mobile_closed_scrolled']=inspect(page);capture(page,'mobile-closed-scrolled')
 check('collapsed TOC stays in header after scroll',abs(r['mobile_closed_scrolled']['button']['y']-6)<0.1,r['mobile_closed_scrolled'])
 page.locator('.md-sidebar--secondary label.md-sidebar-button').click();page.wait_for_timeout(350)
 r['mobile_toc_open']=inspect(page);capture(page,'mobile-toc-open')
 state=r['mobile_toc_open'];popup=state['popup']
 check('TOC popup opens below header and stays within right viewport',state['toc'] and popup['y']>=state['header']['bottom'] and popup['right']<=390 and popup['x']>=0,{'popup':popup,'header':state['header'],'toc':state['toc']})
 link=page.locator('.md-sidebar--secondary .md-nav__link',has_text='安裝與開啟 Notebook').first
 link.click();page.wait_for_timeout(150)
 r['toc_anchor']={'url':page.url,'installation_heading_present':page.locator('h2',has_text='安裝與開啟 Notebook').count()==1}
 check('TOC anchor reaches README installation section','安裝與開啟' in __import__('urllib.parse',fromlist=['unquote']).unquote(page.url),r['toc_anchor'])
 if page.locator('#__toc').is_checked():page.locator('.md-sidebar--secondary label.md-sidebar-button').click()
 page.locator('.md-header label[for="__search"]').click();page.wait_for_timeout(180)
 r['mobile_search_open']=inspect(page);capture(page,'mobile-search-open')
 check('opening search hides the page TOC',r['mobile_search_open']['search'] and r['mobile_search_open']['sidebar']['visibility']=='hidden',r['mobile_search_open'])
 page.get_by_role('combobox').fill('2928')
 search_result=page.locator('a[href*="readme.html?h=2928"]')
 search_result.wait_for(state='visible',timeout=15000);capture(page,'mobile-search-results')
 r['search_result']={'query':'2928','href':search_result.get_attribute('href'),'text':search_result.inner_text()}
 check('search finds the current README score text','2928' in r['search_result']['text'],r['search_result'])
 search_result.click();page.wait_for_timeout(350)
 r['mobile_search_closed']=inspect(page);capture(page,'mobile-search-closed')
 check('following the README search result closes search and restores page TOC',not page.locator('#__search').is_checked() and r['mobile_search_closed']['sidebar']['visibility']=='visible',r['mobile_search_closed'])
 page.locator('.md-header label[for="__drawer"]').click();page.wait_for_timeout(180)
 r['mobile_drawer_open']=inspect(page);capture(page,'mobile-drawer-open')
 check('opening drawer hides the page TOC',r['mobile_drawer_open']['drawer'] and r['mobile_drawer_open']['sidebar']['visibility']=='hidden',r['mobile_drawer_open'])
 page.mouse.click(360,500);page.wait_for_timeout(180)
 r['mobile_drawer_closed']=inspect(page);capture(page,'mobile-drawer-closed')
 check('clicking outside drawer closes it and restores page TOC',not page.locator('#__drawer').is_checked() and r['mobile_drawer_closed']['sidebar']['visibility']=='visible',r['mobile_drawer_closed'])
 context.close()

 def desktop(use_old):
  ctx=browser.new_context(viewport={'width':1280,'height':900},device_scale_factor=1)
  pg=ctx.new_page()
  if use_old:pg.route('**/stylesheets/course*.css',lambda route:route.fulfill(status=200,content_type='text/css',body=old_css))
  pg.goto(URL,wait_until='domcontentloaded',timeout=30000);pg.wait_for_timeout(250)
  value=inspect(pg)
  if not use_old:capture(pg,'desktop-current')
  ctx.close();return value
 r['desktop_current']=desktop(False);r['desktop_with_only_prior_css']=desktop(True)
 keys=('header','title','sidebar','button','popup','mediaMobile')
 check('1280px desktop geometry unchanged against prior stylesheet',all(r['desktop_current'][k]==r['desktop_with_only_prior_css'][k] for k in keys),{'compared_keys':keys,'current':r['desktop_current'],'prior_css_only':r['desktop_with_only_prior_css']})
 ctx=browser.new_context(viewport={'width':959,'height':900});pg=ctx.new_page();pg.goto(URL,wait_until='domcontentloaded',timeout=30000)
 r['breakpoint']={'width959_mobile':pg.evaluate("matchMedia('(max-width:59.984375em)').matches")}
 pg.set_viewport_size({'width':960,'height':900});r['breakpoint']['width960_mobile']=pg.evaluate("matchMedia('(max-width:59.984375em)').matches")
 check('mobile media cutoff is below 960px',r['breakpoint']['width959_mobile'] and not r['breakpoint']['width960_mobile'],r['breakpoint'])
 ctx.close();browser.close()
r['all_checks_passed']=all(x['passed'] for x in r['checks'])
r['limits']='Actual preview17 README at390x844 and1280x900, plus959/960 media match. Search was closed by following its real README result; Escape did not close it in the preserved prior probe, so Escape support is not claimed. Actual drawer close used viewport x360,y500, inside the overlay (x380 is in reserved scrollbar). Existing HF/model/scoring evidence is not rerun. This does not certify all21 capstone route pages, everyviewport/browser, or formal model tests. Screenshots are ignored work artifacts with hashes, not Git PNG payloads.'
(OUT/'css-ui-results-03.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_checks_passed':r['all_checks_passed'],'checks':r['checks'],'screenshots':r['screenshots']},ensure_ascii=False))
