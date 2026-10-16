import ast,contextlib,hashlib,io,json,platform,re,subprocess,sys,tomllib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.reading_time import lesson_slices
from scripts.stage_mathjax import validate_manifest,read_asset,stage_assets
R=Path.cwd();W=Path('/workspace/work/curriculum-20261016');F=json.loads((R/'docs/course-revision-20261016/freeze-08.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
changed=[]
for fn in ['course/chapters/07.md','course/chapters/09.md','course/chapters/14.md','course/chapters/19.md']:
 ss=lesson_slices((R/fn).read_text())
 for page in F['pages']:
  if page['source']!=fn or page['kind']!='lesson':continue
  before=(R/page['snapshot']).read_text();after=ss[page['page_id']]
  if before==after:continue
  old=re.findall(r'```python\n(.*?)```',before,re.S);new=re.findall(r'```python\n(.*?)```',after,re.S)
  entry={'id':page['page_id'],'source':fn,'old_snapshot':page['snapshot'],'old_sha256':hashlib.sha256(before.encode()).hexdigest(),'current_sha256':hashlib.sha256(after.encode()).hexdigest(),'python_ast_equal':len(old)==len(new) and all(ast.dump(ast.parse(a))==ast.dump(ast.parse(b)) for a,b in zip(old,new)),'non_python_text_equal':re.sub(r'```python\n.*?```','<PYTHON>',before,flags=re.S)==re.sub(r'```python\n.*?```','<PYTHON>',after,flags=re.S),'executed_examples':[]}
  for k,(a,b) in enumerate(zip(old,new)):
   out=[]
   for version,code in [('freeze08',a),('current',b)]:
    output=io.StringIO()
    with contextlib.redirect_stdout(output):exec(compile(code,f'{page["page_id"]}:{version}', 'exec'),{})
    out.append({'version':version,'stdout':output.getvalue()})
   entry['executed_examples'].append({'block':k,'same_stdout':out[0]['stdout']==out[1]['stdout'],'runs':out})
  changed.append(entry)
settings=tomllib.loads((R/'zensical.toml').read_text())['project']
def navpaths(items):
 for x in items:
  if isinstance(x,str):yield x
  else:
   for v in x.values():
    if isinstance(v,list):yield from navpaths(v)
    else:yield v
nav=list(navpaths(settings['nav']));ids=['7.25','C.1','C.5','8.1','13.17','C.6','C.7','14.1','14.6','14.11','14.18','14.7','14.10','15.1']
manifest=validate_manifest(json.loads((R/'course/web/mathjax-assets.json').read_text()))
cache=R/'outputs/mathjax-cache'/manifest['version'];verified=[]
for item in manifest['files']:
 raw=read_asset(cache/item['path'],item);verified.append({'path':item['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
staged=W/'curriculum-final-assets-callback-01-mathjax'
stage_assets(staged,(cache,manifest))
assert all(sha(staged/item['path'])==item['sha256'] for item in manifest['files'])
check=subprocess.run(['.venv/bin/python','scripts/build_course.py','--check'],capture_output=True,text=True)
counts={str(p.relative_to(R)):len(lesson_slices(p.read_text())) for p in sorted((R/'course/chapters').glob('*.md'))}
front={s:len(lesson_slices((R/s).read_text())) for s in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
# The added example has a deterministic world-rule truth, not a trained-model output.
a,b=2,6;a,b=b,a
with_a8=8
result={'environment':{'python':platform.python_version(),'device':'cpu','platform':platform.platform()},'curriculum_sha256':sha(R/'docs/curriculum.md'),'figure_sha256':sha(R/'course/figures/curriculum_learning_flow.svg'),'changed_sections':changed,'numbered_counts':{'chapter_files':len(counts),'chapter_lessons':sum(counts.values()),'front_sections':sum(front.values()),'all_numbered':sum(counts.values())+sum(front.values()),'notebooks':len(list((R/'notebooks').rglob('*.ipynb')))},'navigation_positions':{i:nav.index(i+'.md') for i in ids},'navigation_sequence_correct':all(nav.index(a+'.md')<nav.index(b+'.md') for a,b in zip(ids,ids[1:])),'chapter19_overview_before_19_1':nav.index('19.13.md')<nav.index('19.1.md'),'notebook_dependencies':tomllib.loads((R/'pyproject.toml').read_text())['dependency-groups']['notebook'],'notebook_sync':{'command':'.venv/bin/python scripts/build_course.py --check','returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr},'mathjax':{'manifest_sha256':sha(R/'course/web/mathjax-assets.json'),'version':manifest['version'],'revision':manifest['revision'],'verified_files':verified,'stage_destination_outside_git':str(staged),'staged_all_sha256_equal':True,'configured_javascript':settings['extra_javascript'],'note':'核對本機既有cache，並用stage_assets寫入獨立work目錄；沒有下載/修改工具、建置全站或宣稱所有瀏覽器公式排版通過。'},'19.17_added_contract_example':{'complete_input':'A=2,B=6;swap;ask B','rule_answer':b,'missing_A_counterexample':{'A2_answer':2,'A8_answer':with_a8,'conclusion':'刪掉A初值後B的答案不能唯一確定；世界規則即可產生資訊不足標籤，不要讓B答題後再反推。'},'classification_candidates':['卡片','地址','App','狀態追蹤','算術','不適用'],'scope':'新增的是訓練與驗證/欄位契約，非C已學會B題型的實測結果。'}}
print(json.dumps(result,ensure_ascii=False,indent=2))
