"""Same-owner bounded callback: current exporter, exact sources and reused evidence.

No commit, publishing, GPU, training, or authored-source mutation. Large details
and command output stay in the ignored /workspace/work cache.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

import torch
from scripts import export_course

ROOT = Path('/workspace/tiny-perceptron-vlm')
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-readme-en-v1/current-v1')
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_bytes())
def ref(path): return {'path': str(path), 'sha256': sha(path)}
original_path = ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_readme_en/technical-original-v1.json'
original = load(original_path)
assert sha(original_path) == 'c415805fae9f88b33e36b5ac230ae2720677a92aa0cb88353dcffc46881dbad4'
freeze_path = ROOT/'docs/course-revision-20261029-omni/reviews/auxiliary-english-v2/source-freeze.json'
freeze = load(freeze_path)
page = freeze['pages'][0]
assert sha(ROOT/page['snapshot']) == sha(ROOT/'README_en.md') == page['source_sha256']
result = {'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_readme_en','environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':'2'},'original_report':ref(original_path),'freeze':ref(freeze_path),'source_sha256':page['source_sha256'],'commands':[]}

result['repository_source_checks'] = []
for source in original['sources']:
    if source['kind'] != 'repository_code': continue
    actual = sha(ROOT/source['path'])
    result['repository_source_checks'].append({'id':source['id'],'path':source['path'],'original_sha256':source['sha256'],'current_sha256':actual,'unchanged':actual==source['sha256']})
drift = [s for s in result['repository_source_checks'] if not s['unchanged']]
assert [s['id'] for s in drift] == ['export']
assert drift[0]['current_sha256'] == '662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf'
result['reused_artifacts'] = []
for artifact in original['artifacts']:
    assert sha(ROOT/artifact['path']) == artifact['sha256']
    result['reused_artifacts'].append({'id':artifact['id'],**ref(ROOT/artifact['path']),'hash_matches_original':True})

# Check original full output / original-source hashes, not just their summaries.
raws = {}
def raw(path, expected):
    if not path: return
    p=Path(path)
    if str(p) in raws: assert raws[str(p)] == expected; return
    assert sha(p) == expected, str(p)
    raws[str(p)] = expected
audit=load(HERE/'audit-summary.json')
supp=load(HERE/'supplement-summary.json')
for command in audit['commands']: raw(command['raw_path'],command['raw_sha256'])
for command in supp['commands']: raw(command['raw_path'],command['sha256'])
for item in supp['external_originals']:
    if item.get('ok'): raw(item['cache_path'],item['sha256'])
for item in supp['additional_originals']: raw(item['path'],item['sha256'])
for item in audit['torch_original_sources']:
    raw(item['distribution_file'],item['distribution_file_sha256'])
    raw(item['original_excerpt_cache'],item['original_excerpt_sha256'])
probes=load(HERE/'final-probes-summary.json')
raw(probes['dry_run_guards']['raw_stdout_path'],probes['dry_run_guards']['raw_stdout_sha256'])
for item in probes['read_only_fetches']:
    if item.get('ok'): raw(item['path'],item['sha256'])
for item in load(HERE/'retrieval-corrections.json')['fetches']:
    if item.get('ok'): raw(item['cache_path'],item['sha256'])
counts=load(HERE/'raw-counts-summary.json');raw(counts['raw_path'],counts['raw_sha256'])
raw(ROOT/'assets/training/selftrained-v2.tar.gz',supp['v2_archive']['archive_sha256'])
result['raw_evidence_reuse']={'all_hashes_match':True,'unique_files':len(raws),'ordered_path_and_hash_digest':hashlib.sha256(json.dumps(raws,sort_keys=True).encode()).hexdigest(),'scope':'Original bounded executions and original sources remain historical executions; they are reused only where their unchanged input/code/denominators remain applicable.'}

index=load(ROOT/'course/lesson-index.json')
result['inventory']={'lesson_sections':len(index),'notebooks':len(list((ROOT/'notebooks').rglob('*.ipynb')))}
assert len(index)==result['inventory']['notebooks']==331
guide_counts={p:len(re.findall(r'^## [RWTG]\.\d+ ',(ROOT/p).read_text(),re.M)) for p in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
result['inventory']['numbered_guides']=guide_counts
result['inventory']['total_numbered']=331+sum(guide_counts.values())
assert sum(guide_counts.values())==26
indirect=[]
for source in audit['inventory']['lesson_source_inputs']:
    actual=sha(ROOT/source['path'])
    if actual!=source['sha256']: indirect.append({'path':source['path'],'original_sha256':source['sha256'],'current_sha256':actual})
result['indirect_chapter_drift']=indirect
result['reread_scope']=['Complete128-line frozen README_en','Complete506-line current exporter','course/README.md routes and W.1 setup/Notebook paths','Current09.md9.11/9.17/9.18 overview and12.md12.17/12.18 mechanism boundaries','All70original claim statements and58source declarations','Source13preview metadata and all331actual notebook/download/footer pairs']
result['source_refs']=[ref(ROOT/p) for p in ['scripts/export_course.py','course/README.md','course/first-steps.md','course/chapters/09.md','course/chapters/12.md','course/lesson-index.json','course/reading-times.json','course/reading-time-routes.json','tests/test_course_export.py']]

env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT))
cwd=CACHE/'command-cwd';cwd.mkdir(exist_ok=True)
if not (cwd/'scripts').exists(): (cwd/'scripts').symlink_to(ROOT/'scripts',target_is_directory=True)
def run(name,command,directory=ROOT,timeout=40):
    p=subprocess.run(command,cwd=directory,env=env,capture_output=True,text=True,timeout=timeout)
    out=CACHE/(name+'.txt')
    with out.open('x') as stream: stream.write(p.stdout+'\n[stderr]\n'+p.stderr)
    entry={'name':name,'command':command,'cwd':str(directory),'timeout_seconds':timeout,'returncode':p.returncode,'raw_output':ref(out),'key_output':p.stdout.strip()[-650:]}
    result['commands'].append(entry);assert p.returncode==0,name
    return p
base=['uv','run','--project',str(ROOT),'--extra','cpu','python']
run('check-env-current',base+['scripts/check_env.py'])
run('prepare-data-current',base+['scripts/prepare_data.py','--kind','toy-text'],cwd)
p=run('dry-run-current',base+['scripts/train.py','--task','text','--data','data/generated/toy-text/train.jsonl'],cwd)
dry=json.loads(p.stdout.strip().splitlines()[-1]);assert dry['mode']=='dry-run-no-weight-update' and dry['device']=='cpu'
assert not (cwd/'checkpoints/text.pt').exists()
result['minimal_workflow']={'actual_report':dry,'checkpoint_created':False,'expected_loss':5.770333290100098,'expected_grad_norm':0.9595053195953369}
run('targeted-export-tests',['uv','run','--extra','cpu','pytest','-q','tests/test_course_export.py','-k','preview_footer_gives_upload_route or footer_claims_outputs_only_when_the_build_carries_them or full_heading_links_reach_the_split_lesson_in_site_and_notebook or unknown_same_page_fragment_is_preserved'])

def git(*args): return subprocess.run(['git',*args],cwd=ROOT,check=True,capture_output=True).stdout
head=git('rev-parse','HEAD').decode().strip();tree=git('rev-parse','HEAD^{tree}').decode().strip()
result['colab_gate']={'head':head,'negative_cases':[]}
for name,revision in [('branch','main'),('short','abc123'),('missing_commit','0'*40),('tree_object',tree)]:
    try: export_course.verify_colab_notebooks(index,revision)
    except ValueError as error: result['colab_gate']['negative_cases'].append({'case':name,'rejected':True,'message':str(error).splitlines()[0]})
    else: raise AssertionError('Unexpected accepted pin '+name)
try: export_course.verify_colab_notebooks(index,head)
except ValueError as error:
    p=CACHE/'head-331-mismatch.txt'
    with p.open('x') as stream: stream.write(str(error)+'\n')
    result['colab_gate']['all331_at_head']={'accepted':False,'raw_error':ref(p),'mismatch_lines':len(str(error).splitlines())-1}
else: result['colab_gate']['all331_at_head']={'accepted':True}
for item in index:
    p=subprocess.run(['git','show',f'{head}:{item["notebook"]}'],cwd=ROOT,capture_output=True)
    if p.returncode==0 and p.stdout==(ROOT/item['notebook']).read_bytes():
        assert export_course.verify_colab_notebooks([item],head) is None
        result['colab_gate']['positive_subset']={'lesson':item['id'],'count':1,'matched_actual_commit_blob':True};break
result['colab_gate']['scope']='Read-only real existing Git objects. No fixture commit or repo mutation. Positive subset is not331-file formal publication approval.'

class Anchors(HTMLParser):
    def __init__(self): super().__init__();self.items=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.items.append(dict(attrs))
site=ROOT/'outputs/omni-review-site-technical-fix13'
bundle_path=ROOT/'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix13-cumulative.json'
bundle=load(bundle_path);byid={r['lesson']:r for r in bundle['records']};assert len(byid)==331
bundle_checks=[];has_outputs=0
for item in index:
    current=ROOT/item['notebook'];published=site/item['notebook'];record=byid[item['id']]
    assert sha(current)==record['notebook_sha256'] and current.read_bytes()==published.read_bytes()
    executed=Path(record['bundle_path']);assert sha(executed)==record['executed_sha256']
    nb=export_course.checked_notebook(load(current),executed)
    s=(site/(item['id']+'.html')).read_text();parser=Anchors();parser.feed(s)
    buttons=[a for a in parser.items if 'md-button' in a.get('class','').split()]
    assert any(a.get('href')==item['notebook'] and 'download' in a for a in buttons)
    assert any(a.get('href')=='https://colab.research.google.com/' for a in buttons)
    assert not any('colab.research.google.com/github/' in a.get('href','') for a in buttons)
    assert all(text in s for text in ['尚未發布','直連尚未提供','Upload notebook','File → Upload notebook','雲端環境未經本課實測'])
    if 'output-label' in s: has_outputs+=1
    bundle_checks.append([item['id'],sha(current),sha(executed),sha(published)])
build=load(site/'build-info.json')
assert build['lessons']==331 and build['colab_mode']=='upload' and build['publication_ready'] is False and build['colab_git_objects_verified'] is False
result['preview331']={'site':str(site),'build_info':ref(site/'build-info.json'),'build_metadata':build,'all_downloads_match_current_bytes':True,'all_executed_copies_match_current_cells':True,'all_preview_upload_instructions_and_home_links':True,'checked':len(bundle_checks),'pages_with_cpu_output_marker':has_outputs,'source_bundle':ref(bundle_path),'ordered_notebook_proof_digest':hashlib.sha256(json.dumps(bundle_checks).encode()).hexdigest(),'scope':'Bundle13 is author cumulative execution evidence:9.15 rerun,330matching-cell reuses. This callback validates hashes/cells and does not claim331new kernel runs.'}
try:
    with urlopen('http://127.0.0.1:8774/1.1.html',timeout=5) as response: raw=response.read()
    p=CACHE/'preview8774-1.1.html'
    with p.open('xb') as stream:stream.write(raw)
    result['preview_http']={'url':'http://127.0.0.1:8774/1.1.html','ok':True,'raw':ref(p),'same_bytes_as_preview_file':raw==(site/'1.1.html').read_bytes()}
except Exception as error: result['preview_http']={'ok':False,'error':str(error),'scope':'Static preview331 audit remains available; no live-site claim inferred.'}
result['claims_recheck']={'all_original_ids':[c['id'] for c in original['claims']],'direct_export_affected':['c9','c10','c36','c43','c44','c66','c67','c69'],'indirect_route_affected':['c13','c21','c22','c25'],'original_issue_count':len(original['issues']),'original_issue_resolution_count':0,'verdict':'pass','notes':'All70claims reread against unchanged English source. Exporter semantics and route changes checked directly; unchanged method/historical input evidence hash-verified. No new source contradiction found. The3reader issues belong to the reader and are not technical original issues.'}
result['limits']=['No actual Colab execution, deployment, GPU or formal training.','No source/checker/model/active/commit changes.','Source13preview reading_time_pages=366 and reading_time_complete=false is not current publication readiness. Raw metadata row count or later reader status is not independently passed here.','W.1 was consulted only for README-linked setup semantics; no full first-steps reader acceptance inferred.','English auxiliary identity retained; no canonical count or section sidecar created.']
out=HERE/'current-audit-v1.json'
with out.open('x') as stream: json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
print(json.dumps({'artifact':ref(out),'drift':drift,'commands':[(r['name'],r['returncode']) for r in result['commands']],'reused_raw_files':len(raws),'preview331':{k:v for k,v in result['preview331'].items() if k not in ['build_metadata']},'colab_gate':result['colab_gate']},ensure_ascii=False))
