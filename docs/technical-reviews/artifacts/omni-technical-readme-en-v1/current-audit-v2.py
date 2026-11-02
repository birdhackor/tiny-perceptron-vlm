"""Same-owner English auxiliary callback against exact canonical source15 slices.

Checks evidence identity and changed premises; never trains, commits or publishes.
"""
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

import torch
from scripts import export_course

ROOT=Path('/workspace/tiny-perceptron-vlm')
HERE=Path(__file__).resolve().parent
CACHE=Path('/workspace/work/omni-technical-readme-en-v1/current-v2')
CACHE.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def ref(p):return {'path':str(p),'sha256':sha(p)}
reportdir=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_readme_en'
original=reportdir/'technical-original-v1.json';previous=reportdir/'technical-current-v1.json'
assert sha(original)=='c415805fae9f88b33e36b5ac230ae2720677a92aa0cb88353dcffc46881dbad4'
assert sha(previous)=='c40cac3fceba042a51b3195ab4e7f342c58ccec765d6e7077276cc4d10699a00'
old=load(previous)
aux=ROOT/'docs/course-revision-20261029-omni/reviews/auxiliary-english-v2/source-freeze.json'
assert sha(aux)=='f842a61963eb3fcda1a47f987924dfed5d3d6656c91b6490681588f37c64a79b'
assert sha(ROOT/'README_en.md')==sha(ROOT/load(aux)['pages'][0]['snapshot'])=='4c5ddc43d3cfb450ee77971c51a9f5bc50146f317e667cfc7f7b8d9cf9fee2e6'
context=ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json'
assert sha(context)=='1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238'
pages={p['page_id']:p for p in load(context)['pages']}
before={p['page_id']:p for p in load(ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json')['pages']}
result={'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_readme_en','review_kind':'same_owner_source_dependency_callback_not_fresh_original','environment':{'python':os.sys.version.split()[0],'torch':str(torch.__version__),'device':'cpu','threads':'2'},'original':ref(original),'previous_current':ref(previous),'auxiliary_freeze':ref(aux),'canonical_context_freeze':ref(context),'claims_rechecked':[c['id'] for c in old['claims']]}
drift=[]
for s in old['sources']:
    if s.get('path') and sha(ROOT/s['path'])!=s['sha256']:drift.append({'id':s['id'],'path':s['path'],'old':s['sha256'],'current':sha(ROOT/s['path'])})
assert [d['id'] for d in drift]==['current_ch12']
assert drift[0]['current']=='3472aa511ad598f041d5f25c4ee28f58269a732112594d53290991f91fa813ce'
result['registered_dependency_drift']=drift
for a in old['artifacts']:assert sha(ROOT/a['path'])==a['sha256']
result['reused_artifact_count']=len(old['artifacts'])
raws={}
def raw(path,expected):
    if str(path) in raws:assert raws[str(path)]==expected;return
    assert sha(path)==expected,str(path)
    raws[str(path)]=expected
audit=load(HERE/'audit-summary.json');supp=load(HERE/'supplement-summary.json')
for c in audit['commands']:raw(c['raw_path'],c['raw_sha256'])
for c in supp['commands']:raw(c['raw_path'],c['sha256'])
for s in supp['external_originals']:
    if s.get('ok'):raw(s['cache_path'],s['sha256'])
for s in supp['additional_originals']:raw(s['path'],s['sha256'])
for s in audit['torch_original_sources']:
    raw(s['distribution_file'],s['distribution_file_sha256']);raw(s['original_excerpt_cache'],s['original_excerpt_sha256'])
p=load(HERE/'final-probes-summary.json')
raw(p['dry_run_guards']['raw_stdout_path'],p['dry_run_guards']['raw_stdout_sha256'])
for s in p['read_only_fetches']:
    if s.get('ok'):raw(s['path'],s['sha256'])
for s in load(HERE/'retrieval-corrections.json')['fetches']:
    if s.get('ok'):raw(s['cache_path'],s['sha256'])
raw(ROOT/'assets/training/selftrained-v2.tar.gz',supp['v2_archive']['archive_sha256'])
result['original_raw_reuse']={'verified':len(raws),'all_sha_match':True,'path_hash_map_digest':hashlib.sha256(json.dumps(raws,sort_keys=True).encode()).hexdigest(),'scope':'Original executions/primary sources retain their original dates and denominators, not relabelled as fresh runs.'}
changed=[i for i in pages if i not in before or pages[i]['source_sha256']!=before[i]['source_sha256']]
assert changed==['5.7','12.2','chapter-10','chapter-14']
result['source13_to_source15_changed_pages']=changed
result['exact_context_slices']=[]
for ident in ['12.2','12.17','12.18','chapter-12','5.7','chapter-10','chapter-14','9.11','9.17','9.18','10.12','14.11','14.19']:
    p=pages[ident];snapshot=ROOT/p['snapshot'];assert sha(snapshot)==p['source_sha256']
    text=(ROOT/p['source']).read_text()
    if ident.startswith('chapter-'):
        m=re.search(r'^## ',text,re.M);actual=text[:m.start()] if m else text
    else:actual=re.search(r'^## '+re.escape(ident)+r' .+?(?=^## |\Z)',text,re.M|re.S).group(0)
    assert actual.encode()==snapshot.read_bytes(),ident
    result['exact_context_slices'].append({'page_id':ident,'source':p['source'],'snapshot':p['snapshot'],'sha256':p['source_sha256'],'same_as_source13':p['source_sha256']==before[ident]['source_sha256'],'matches_current_authored_slice':True})
first=pages['first-steps'];firsttext=(ROOT/first['snapshot']).read_text();w1=re.search(r'^## W\.1 .+?(?=^## |\Z)',firsttext,re.M|re.S).group(0).encode()
w1path=ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v13/sections/W.1.md'
assert w1path.read_bytes()==w1
result['exact_w1_slice']={**ref(w1path),'parent_page_id':'first-steps','parent_snapshot':first['snapshot'],'parent_sha256':first['source_sha256'],'matches_source15_exact_slice':True}
result['premise_decisions']={
 '12.2':'Only added instruction to rename the two print labels when exercising9500/6500Hz. No source code/physical-time rule changed; README cites12.17/18, which are exact unchanged source15 slices. Whole-chapter pin replaced by these narrow primary snapshots.',
 '5.7':'weights_only wording now describes restricted types/functions and reduced arbitrary-code risk. English makes no absolute untrusted-load safety claim; c59/c60/c65 concern inference state versus optimizer/RNG resume and remain supported.',
 'chapter-10':'Names10.9–10.11pairing and adds final10.12timestamps, agreeing with English18-addition route.',
 'chapter-14':'Fixes completed model link4.5→4.6 and adds14.19time/height/width rotation after14.11–18Looped route. English cited sections and required topic remain unchanged.',
 'all70':'Full English page reread; inventory, setup, finite product lineage/results, CPU example and method boundaries remain supported. No reader issue or coherence judgment adopted.'}

# Execute the exact exercise edit, not a helper-only stand-in or a training run.
section=(ROOT/pages['12.2']['snapshot']).read_text();code=re.findall(r'```python\n(.*?)```',section,re.S)[1]
exercise=code.replace('9000.0','9500.0').replace('7000.0','6500.0').replace('9000Hz','9500Hz').replace('7000Hz','6500Hz')
program=CACHE/'12.2-exercise.py'
with program.open('x') as stream:stream.write('import torch\ntorch.set_num_threads(2)\n'+exercise)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT))
cmd=['uv','run','--extra','cpu','python',str(program)]
p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
output=CACHE/'12.2-exercise-output.txt'
with output.open('x') as stream:stream.write(p.stdout+'\n[stderr]\n'+p.stderr)
assert p.returncode==0 and '9500Hz前四點' in p.stdout and '6500Hz前四點' in p.stdout and p.stdout.rstrip().endswith('True')
result['exercise_execution']={'command':cmd,'timeout_seconds':20,'returncode':p.returncode,'input_program':ref(program),'output':ref(output),'key_stdout':p.stdout,'scope':'16sample CPU aliasing/label exercise only, not formal training or a general audio-quality result.'}
index=load(ROOT/'course/lesson-index.json');assert len(index)==331
site=ROOT/'outputs/omni-review-site-technical-fix15'
bundlepath=ROOT/'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix15-cumulative.json'
bundle=load(bundlepath);records={r['lesson']:r for r in bundle['records']};assert len(records)==331
proof=[]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs))
for item in index:
    a=ROOT/item['notebook'];b=site/item['notebook'];record=records[item['id']];executed=Path(record['bundle_path'])
    assert sha(a)==record['notebook_sha256'] and sha(executed)==record['executed_sha256'] and a.read_bytes()==b.read_bytes()
    export_course.checked_notebook(load(a),executed)
    html=(site/(item['id']+'.html')).read_text();links=Links();links.feed(html)
    actions=[s for s in links.links if 'md-button' in s.get('class','').split()]
    assert any(s.get('href')==item['notebook'] and 'download' in s for s in actions)
    assert any(s.get('href')=='https://colab.research.google.com/' for s in actions)
    assert not any('colab.research.google.com/github/' in s.get('href','') for s in actions)
    assert all(s in html for s in ['尚未發布','Upload notebook','File → Upload notebook','雲端環境未經本課實測'])
    proof.append([item['id'],sha(a),sha(executed),sha(b)])
meta=load(site/'build-info.json')
assert meta['colab_mode']=='upload' and not meta['publication_ready'] and not meta['colab_git_objects_verified']
result['preview15']={'site':str(site),'build_info':ref(site/'build-info.json'),'metadata':meta,'bundle':ref(bundlepath),'checked_notebooks':len(proof),'all_download_and_execution_cells_match':True,'all_preview_upload_footers_match':True,'ordered_proof_digest':hashlib.sha256(json.dumps(proof).encode()).hexdigest(),'scope':'Source15cumulative bundle reexecutes5.7only and reuses330exact matching-cell results. This callback performs no331kernel rerun.'}
try:
    with urlopen('http://127.0.0.1:8776/1.1.html',timeout=5) as response:data=response.read()
    q=CACHE/'preview8776-1.1.html'
    with q.open('xb') as stream:stream.write(data)
    result['preview_http']={'url':'http://127.0.0.1:8776/1.1.html','raw':ref(q),'same_as_site_file':data==(site/'1.1.html').read_bytes()}
except Exception as error:result['preview_http']={'error':str(error),'static_preview_checked':True}
result['unchanged_exporter']=ref(ROOT/'scripts/export_course.py')
result['limits']=['English auxiliary only; no new canonical page/main dispositions/reader/seal/coherence/full-tool acceptance.','Original English reader3issues remain in the original auxiliary index; no new technical findings created from them.','Original792fexporter and old whole-chapter12reconstruction are not current semantic evidence; no old-source exemption.','No GPU, paid call, commit, formal training, publication or actual Colab runtime.','No whole-chapter12dependency in current2; canonical15exact snapshots and explicit parent/slice checks bind context.']
out=HERE/'current-audit-v2.json'
with out.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
print(json.dumps({'artifact':ref(out),'canonical_context':ref(context),'changed':changed,'exercise_stdout':p.stdout,'preview_notebooks_checked':len(proof),'reused_raw_files':len(raws)},ensure_ascii=False))
