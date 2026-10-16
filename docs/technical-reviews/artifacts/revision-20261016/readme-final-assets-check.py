import ast,hashlib,json,os,pathlib,platform,re,subprocess,tomllib
from scripts.stage_mathjax import read_asset,stage_assets,validate_manifest
R=pathlib.Path('/workspace/tiny-perceptron-vlm');W=pathlib.Path('/workspace/work/curriculum-20261016/readme-review/final-assets');W.mkdir(exist_ok=True)
P=R/'.venv/bin/python';A=R/'docs/technical-reviews/artifacts/revision-20261016/readme-final-assets-execution.json'
H=lambda b:hashlib.sha256(b).hexdigest();oldrep=json.loads((R/'docs/course-revision-20261016/reviews/technical-readme-callback-01.json').read_text());oldsrc={x['path']:x for x in oldrep['sources'] if 'path' in x};freeze=json.loads((R/'docs/course-revision-20261016/freeze-08.json').read_text());pages={p['page_id']:p for p in freeze['pages']};out={'command':'.venv/bin/python /workspace/work/curriculum-20261016/readme-review/final_assets_check.py','environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu'},'checks':{},'commands':[]}
# Reconstruct old full chapter bytes from the original frozen slices, then compare current changes.
changed={};reconstructed={};related={}
for chap,ids in [('07',['7.17','7.23']),('09',['9.11']),('14',['14.11']),('19',['19.13','19.17','19.18','19.19','19.20'])]:
 path='course/chapters/'+chap+'.md';raw=(R/path).read_text();heads=list(re.finditer(r'^## ([A-Z\d]+\.\d+) .*$',raw,re.M));oldchunks=[(R/pages['chapter-'+chap]['snapshot']).read_text()];rows=[]
 for i,m in enumerate(heads):
  ident=m[1];cur=raw[m.start():heads[i+1].start() if i+1<len(heads) else len(raw)];prev=(R/pages[ident]['snapshot']).read_text();oldchunks.append(prev)
  if ident in ids:related[ident]={'current_sha256':H(cur.encode()),'frozen_sha256':H(prev.encode()),'identical':cur==prev,'current_location':path+'#'+ident}
  if cur!=prev:
   strip=lambda t:re.sub(r'(?ms)^```python\n.*?^```[ \t]*$','<PYTHON>',t)
   oldcode=re.findall(r'(?ms)^```python\n(.*?)^```[ \t]*$',prev);newcode=re.findall(r'(?ms)^```python\n(.*?)^```[ \t]*$',cur)
   rows.append({'id':ident,'prose_equal':strip(prev)==strip(cur),'python_ast_equal':len(oldcode)==len(newcode) and all(ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False) for a,b in zip(oldcode,newcode))})
 oldfull=''.join(oldchunks).encode();reconstructed[path]={'reconstructed_old_sha256':H(oldfull),'prior_recorded_sha256':oldsrc[path]['sha256'],'matches':H(oldfull)==oldsrc[path]['sha256'],'current_sha256':H((R/path).read_bytes())};assert reconstructed[path]['matches'];changed[path]=rows
out['checks']['chapter_reconstruction']=reconstructed;out['checks']['chapter_changed_sections']=changed;out['checks']['related_sections']=related
for ident in ['7.17','7.23','9.11','14.11','19.13','19.18','19.19','19.20']:assert related[ident]['identical']
prev=subprocess.check_output(['git','show','HEAD:pyproject.toml'],cwd=R);current=(R/'pyproject.toml').read_bytes();assert H(prev)==oldsrc['pyproject.toml']['sha256'];t1=tomllib.loads(prev.decode());t2=tomllib.loads(current.decode());assert t1['project']==t2['project'] and t1['dependency-groups']==t2['dependency-groups'] and t1['tool']['uv']==t2['tool']['uv'];out['checks']['pyproject']={'before_sha256':H(prev),'after_sha256':H(current),'project_dependency_groups_uv_contract_equal':True,'new_ruff_excludes':list(set(t2['tool']['ruff']['extend-exclude'])-set(t1['tool']['ruff']['extend-exclude']))}
theme=(R/'zensical.toml').read_text();reversed_theme=theme.replace('"javascripts/mathjax/tex-chtml.js"','"https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"');assert H(reversed_theme.encode())==oldsrc['zensical.toml']['sha256'];out['checks']['theme']={'only_change_since_prior_review':'MathJax script URL changed from CDN to javascripts/mathjax/tex-chtml.js; nav/features/palettes unchanged','current_sha256':H(theme.encode())}
manifest=validate_manifest(json.loads((R/'course/web/mathjax-assets.json').read_bytes()));cache=R/'outputs/mathjax-cache'/manifest['version'];assets=[]
for item in manifest['files']:
 raw=read_asset(cache/item['path'],item);assets.append({'path':item['path'],'bytes':len(raw),'sha256':H(raw)})
# Only copy already-verified assets to scratch; do not run a website/Notebook build or download.
staged=W/'staged-mathjax';stage_assets(staged,(cache,manifest))
for item in manifest['files']:read_asset(staged/item['path'],item)
out['checks']['mathjax_cache_and_stage']={'version':manifest['version'],'revision':manifest['revision'],'manifest_sha256':H((R/'course/web/mathjax-assets.json').read_bytes()),'verified_file_count':len(assets),'verified_bytes':sum(x['bytes'] for x in assets),'destination':str(staged),'network_download_performed':False,'assets':assets}
index=json.loads((R/'course/lesson-index.json').read_text());notebooks=list((R/'notebooks').rglob('*.ipynb'));out['checks']['inventory']={'lesson_index':len(index),'notebooks':len(notebooks)};assert len(index)==len(notebooks)==312
body=(R/'README.md').read_text();assert H(body.encode())==oldrep['source_sha256'];out['checks']['README']={'sha256':H(body.encode()),'body_unchanged':True,'all_local_links_valid':True}
for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',body):
 if target.startswith(('http:','https:')):continue
 path,_,frag=target.partition('#');p=R/path;assert p.exists();assert not frag or re.search(r'^## '+re.escape(frag)+r' ',p.read_text(),re.M)
en=(R/'README_en.md').read_text();assert H(en.encode())==oldrep['checked_files'][0]['sha256'];assert 'a required reasoning-compute unit C' in en and 'pending training and acceptance' in en;out['checks']['README_en']={'sha256':H(en.encode()),'unchanged_since_bilingual_callback':True,'required_C_and_pending_products_retained':True}
for argv,env in [([str(P),'-m','pytest','-q','tests/test_stage_mathjax.py','tests/test_course_export.py'],None),(['uv','sync','--dry-run','--frozen','--extra','cpu','--group','notebook'],{**os.environ,'UV_PROJECT_ENVIRONMENT':str(W/'not-created-env')})]:
 p=subprocess.run(argv,cwd=R,env=env,capture_output=True,text=True,timeout=60);out['commands'].append({'command':subprocess.list2cmdline(argv),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0
out['inspection_notes']={'export_course':'親讀目前prepare_assets在清除舊docs之前執行、stage_assets在複製site腳本後執行、C.1與C.6章目錄分組新增；Notebook/Colab/輸出入口仍同一來源。','chapter19.17':'親讀新增C候選擴充與追蹤/算術資訊充分欄位：先按世界規則標籤、不用B錯答反推；原規則Auto與policy-only閾值、固定B後進階訓練、未驗收成果限定仍在。README只預告三產品/Auto，未聲稱已完成該訓練。','mathjax':'親讀目前stage_mathjax.py、manifest、mathjax.js、zensical設定，實核94份快取及scratch staging全SHA/bytes；不宣稱自己重做作者瀏覽器實測。README連網描述沒有指定外部CDN或宣稱離線必失效；現版由同站HTTP供應MathJax，仍符合網站由MathJax排版的行為，原報告外部CDN的來源描述已過期須更正。','author_evidence':'作者author-change-receipt-09與author-mathjax-recovery-01僅用來定位；AST對比/asset驗證及focused tests是本人本次實執，不把作者desktop/mobile視看算成本人。'}
out['not_repeated']=['模型訓練','舊MoE/Dense或text_foundation再次下載/推論','全部Notebook執行/產生','全站建置','桌面/手機或真Colab瀏覽器操作','其他硬體/OS平台實測'];A.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'result':'pass','chapter_reconstruction':reconstructed,'changed_sections':changed,'mathjax_files':len(assets),'mathjax_bytes':sum(x['bytes'] for x in assets),'commands':[{'command':x['command'],'returncode':x['returncode'],'stdout':x['stdout'][:250]} for x in out['commands']]},ensure_ascii=False))
