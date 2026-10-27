"""Post-original, issue-specific historical guard recheck by fresh tool reviewer."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/tokenizer-20261027/tools-callback'
SCRATCH=Path('/workspace/work/tokenizer-dynamic-20261027/tools-callback/historical-01-fixtures')
ORIGINAL='docs/course-review-20261027-tokenizer/reviews/technical-tools-callback/tool-audit-original-01.json'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':path,'sha256':digest((ROOT/path).read_bytes())}
started=datetime.datetime.now(datetime.UTC).isoformat()
original_ref=ref(ORIGINAL)
assert json.loads((ROOT/ORIGINAL).read_text())['created_at'] < started
spec=importlib.util.spec_from_file_location('historical_guard_review',ROOT/'docs/review-tools/incremental_revision.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
cases=[]
def add(issue,name,text,allowed):cases.append((issue,name,text,allowed))

M='AUTHOR-PATCH-MARKUP'
for name,text in {
 'url':'[連結](https://example.invalid/new-token)\n',
 'label':'[new-token](https://example.invalid/)\n',
 'reference':'[連結][x]\n\n[x]: https://example.invalid/new-token\n',
 'image':'![圖](new-token.png)\n',
 'html_image':'<img src="new-token.png">\n',
}.items():add(M,name,text,True)
C='AUTHOR-PATCH-CODEBLOCK'
for name,closing in [('indented_one',' ```'),('indented_four','    ```'),('quoted','> ```')]:
 add(C,name,'```python\n'+closing+'\nprint("new-token")\n```\n',False)
add(C,'plain_inside','```python\nprint("new-token")\n```\n',False)
add(C,'indented_excluded','    print("new-token")\n',True)
add(C,'html_pre_excluded','<pre>new-token</pre>\n',True)
W='AUTHOR-PATCH-FENCE-WHITESPACE'
for label,ch in [('nbsp','\u00a0'),('em_space','\u2003'),('ideographic','\u3000'),('vertical_tab','\x0b')]:
 for fence in ('```','~~~'):
  add(W,label+('_backtick' if fence=='```' else '_tilde'),fence+'python\n'+fence+ch+'\nprint("new-token")\n'+fence+'\n',False)
S='AUTHOR-PATCH-FENCE-STATE'
for name,opening in [('backtick_info','```bad`info'),('invalid_option','```python invalid')]:
 add(S,name,opening+'\n```\nprint("new-token")\n```\n',False)
add(S,'unclosed_eof','new-token\n\n```python\nx=1\n',False)
add(S,'plain_after','```python\nx=1\n```\nnew-token\n',True)
add(S,'plain_after_crlf','```python\r\nx=1\r\n```\r\nnew-token\r\n',True)

results=[]
for issue,name,text,allowed in cases:
 root=SCRATCH/(issue+'-'+name);root.mkdir(parents=True,exist_ok=False)
 source='course/chapters/99.md';(root/source).parent.mkdir(parents=True);(root/source).write_bytes(text.encode())
 patch={'id':'probe','path':source,'old':'old-token','new':'new-token','requested_by':'author','requested_on':'2026-10-27','reason':'Independent historical guard recheck'}
 raw=(json.dumps({'schema_version':1,'kind':r.PATCHES_KIND,'statement':r.PATCHES_STATEMENT,'patches':[patch]},ensure_ascii=False)+'\n').encode()
 (root/'patches.json').write_bytes(raw)
 manifest={'author_patches':{'path':'patches.json','sha256':digest(raw)}}
 try:
  patches=r.author_patches(root,manifest,{'99.1':{'source':source}})
  observed=r.file_versions(root,patches)
  expected={source:{digest(text.replace('new-token','old-token').encode()):('probe',)}}
  accepted=True;passed=allowed and observed==expected;detail=repr(observed)
 except ValueError as e:
  accepted=False;passed=not allowed;detail=str(e)
 results.append({'issue_id':issue,'case':name,'current_text':text,'expected_acceptance':allowed,'accepted':accepted,'passed':passed,'observed':detail})

current=(ROOT/'docs/review-tools/incremental_revision.py').read_text()
old=subprocess.check_output(['git','show','HEAD:docs/review-tools/incremental_revision.py'],cwd=ROOT,text=True)
def functions(text):
 tree=ast.parse(text)
 return {n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
oldfn,newfn=functions(old),functions(current)
guards={name:{'same_as_historical_HEAD':oldfn[name]==newfn[name],'sha256':digest(newfn[name].encode())} for name in ('author_patches','positions','fence_blocks','fenced','undo_patches','file_versions','page_versions')}
assert all(v['same_as_historical_HEAD'] for v in guards.values())
dpath='docs/course-review-20261019/dispositions-continuity-01.json';d=json.loads((ROOT/dpath).read_text())
historical=[]
for i in range(201,205):
 item=d['items'][i]
 for reference in (item['report'],item['verification'][0]['report']):
  assert ref(reference['path'])==reference
 historical.append({'issue_id':item['issue_id'],'disposition_ref':{**ref(dpath),'json_pointer':f'/items/{i}'},'original_report':item['report'],'json_pointer':item['json_pointer'],'historical_fix_verification':item['verification'][0],'current_files_sha256':{p:ref(p)['sha256'] for p in item['affected_files']}})
record={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'started_at':started,'original_review':original_ref,'command':f'{sys.executable} {__file__}','environment':{'python':sys.version,'platform':platform.platform()},'scope':'24 issue-specific source guard probes through author_patches + file_versions; no renderer or full candidate manifest assertion','policy_read':{'AGENTS.md':ref('AGENTS.md'),'docs/technical-review-guide.md':ref('docs/technical-review-guide.md')},'guard_functions':guards,'historical_references':historical,'results':results,'summary':{'passed':sum(x['passed'] for x in results),'total':len(results)}}
assert ref(ORIGINAL)==original_ref
(OUT/'audit-03-historical-probe.py').write_bytes(Path(__file__).read_bytes())
dest=OUT/'audit-03-historical-probes.json';dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'summary':record['summary'],'groups':{issue:{'passed':sum(x['passed'] for x in results if x['issue_id']==issue),'total':sum(x['issue_id']==issue for x in results)} for issue in (M,C,W,S)},'guards_unchanged':True,'artifact':str(dest.relative_to(ROOT))},ensure_ascii=False,indent=2))
raise SystemExit(0 if all(x['passed'] for x in results) else 1)
