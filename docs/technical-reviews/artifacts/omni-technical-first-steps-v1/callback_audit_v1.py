"""Actual same-owner first-steps callback: exact sources, CPU, collection and deliveries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
CANON=ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
SECTION=ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v11.json'
BUNDLE=ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v3.json'
INDEX=ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-issues-v3.json'
PREVIEW=ROOT/'outputs/omni-review-site-technical-fix13'
RAW=ROOT/'outputs/omni-notebooks-technical-fix13-current'
ORIGINAL=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_first_steps/technical-original-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)}
def read(p):return json.loads(p.read_bytes())
def verify(r):
    p=ROOT/r['path'];assert sha(p)==r['sha256'],r['path'];return p
def save(name,obj):
    p=OUT/name
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    return p
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

started=datetime.now(timezone.utc).isoformat()
canon=read(CANON);section=read(SECTION);bundle=read(BUNDLE);index=read(INDEX)
assert sha(BUNDLE)=='d8c8015797dff2e030b211499822873f236d54f5c6025f0ac9e08fce4218adba'
assert sha(INDEX)=='d85a5ca25ab04440b3ca5ba730a5d26c17d8fdc82fb22b56a9a4a0397cd7440f'
page=next(x for x in canon['pages'] if x['page_id']=='first-steps')
snapshot=ROOT/page['snapshot'];body=snapshot.read_text()
assert sha(snapshot)==page['source_sha256']==sha(ROOT/page['source'])
assert body.startswith('# 操作與數學暖身\n')
own_original=read(ORIGINAL)
assert sha(ORIGINAL)=='b306b90afd771bc18937f8d5d370498e7d52185860dad745415c28539f519d73'
preserved=[ref(ORIGINAL)]
for a in own_original['artifacts']:verify(a);preserved.append({'id':a['id'],**ref(ROOT/a['path'])})
oldexport=ROOT/'docs/technical-reviews/artifacts/omni-author-colab-preview-fix01/export-before.py'
assert sha(oldexport)==next(s['sha256'] for s in own_original['sources'] if s['id']=='s_export')
repo_dependencies=[]
for s in own_original['sources']:
    if s['kind']=='repository_code':repo_dependencies.append({'id':s['id'],'path':s['path'],'original_sha256':s['sha256'],'current_sha256':sha(ROOT/s['path'])})

# Read actual owner originals/currents and their small evidence; no copied reports.
entries=[];all_artifacts={};report_receipts=[]
for entry in bundle['sections']:
    for key in ['report','original_report','original_section_freeze']:verify(entry[key])
    if not entry['lesson_id'].startswith('W.'):continue
    original_path=verify(entry['original_report']);current_path=verify(entry['report'])
    original=read(original_path);current=read(current_path)
    assert original['reviewer_task']==current['reviewer_task']
    target=next(x for x in section['pages'] if x['page_id']==entry['lesson_id'])
    assert sha(ROOT/target['snapshot'])==target['source_sha256']==current['source_sha256']
    marker='## '+entry['lesson_id']+' '
    start=body.index(marker);end=body.find('\n## ',start+len(marker));sliced=body[start:] if end<0 else body[start:end+1]
    assert hashlib.sha256(sliced.encode()).hexdigest()==target['source_sha256']
    artifacts_checked=[]
    for label,r in [('original',original),('current',current)]:
        for a in r['artifacts']:
            p=verify(a);artifacts_checked.append({'version':label,'id':a['id'],'kind':a['kind'],**ref(p)})
            if p.suffix=='.json':all_artifacts[str(p)]=read(p)
            else:p.read_text()
        report_receipts.append({'lesson_id':entry['lesson_id'],'version':label,**ref(original_path if label=='original' else current_path),
            'reviewer_task':r['reviewer_task'],'created_at':r.get('created_at'),'claims':len(r['claims']),
            'claim_ids':[c['id'] for c in r['claims']], 'verdict':r['verdict'],'issues':r['issues']})
    entries.append({'lesson_id':entry['lesson_id'],'reviewer_task':current['reviewer_task'],'current_source_sha256':current['source_sha256'],
        'original_collection':entry['original_collection'],'current_collection':entry['current_collection'],
        'current_dispatch_window':entry.get('current_dispatch_window'),'artifacts_checked':artifacts_checked,
        'original_issue_resolutions':current.get('original_issue_resolutions',[]),
        'content_recheck':'Compared complete W.1–W.7 claim inventories with current whole-file reading; independently replayed inline code and exercises, checked originals/receipt contracts. W.7 linked targets are navigation scope, not fresh reviews of all target theories.'})
assert len(entries)==7 and len({e['reviewer_task'] for e in entries})==7
for r in index['reports']:verify(r)
assert len(index['reports'])==35 and len(index['issues'])==5

# Current exact inline source is executed by this owner's preserved bounded collector,
# changing only its selected snapshot and disabling its immutable original-output write.
source=(OUT/'collect_original.py').read_text().split('target = OUT / "cpu-execution-v1.json"')[0]
source=source.replace('SNAP = ROOT / "docs/course-review-20261019/freeze-02/sources/first-steps.md"','SNAP = ROOT / '+repr(page['snapshot']))
ns={'__file__':str(OUT/'collect_original.py'),'__name__':'callback_current_replay'}
exec(compile(source,'callback-selected-current-source','exec'),ns)
replay=ns['results'];replay['command']='uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/callback_audit_v1.py (current-source replay)'
old_cpu=read(OUT/'cpu-execution-v1.json')
assert len(replay['inline_blocks'])==8
assert [x['stdout'] for x in replay['inline_blocks']]==[x['stdout'] for x in old_cpu['inline_blocks']]
error_probe={}
try:{}['狗']
except KeyError as error:
    error_probe={'type':type(error).__name__,'is_str':isinstance(error,str),'message':str(error)}
assert error_probe['type']=='KeyError' and error_probe['is_str'] is False
assert 'error' not in locals()
replay['corrected_exception_binding_probe']=error_probe
save('cpu-current-callback-v1.json',replay)

# Exact current navigation targets and small excerpts, including removal of12.13.
routes=[]
for n,line in enumerate(body.splitlines(),1):
    if not line.startswith('| `tiny_perceptron/') or line.split('`')[1]=='tiny_perceptron/':continue
    path=line.split('`')[1]
    for _,chapter,first,last in re.findall(r'\[([^\]]+)\]\((chapters/[^)#]+)#([^)]+)\)(?:[–-]([A-Z\d]+\.\d+))?',line):
        ids=[first]
        if last:ids=[first.rsplit('.',1)[0]+'.'+str(i) for i in range(int(first.rsplit('.',1)[1]),int(last.rsplit('.',1)[1])+1)]
        chapter_path=ROOT/'course'/chapter;t=chapter_path.read_text()
        for lesson in ids:
            m=re.search(r'^## '+re.escape(lesson)+r' .*?(?=^## |\Z)',t,re.M|re.S);assert m,lesson
            mentions=[v for v in m[0].splitlines() if path in v or path.rstrip('/').split('/')[-1] in v]
            assert mentions,(path,lesson)
            routes.append({'module':path,'lesson_id':lesson,'source':str(chapter_path.relative_to(ROOT)),
                'section_sha256':hashlib.sha256(m[0].encode()).hexdigest(),'supporting_excerpts':mentions[:3]})
assert len(routes)==65
assert not any(x['lesson_id']=='12.13' and x['module'].endswith('natural_concepts.py') for x in routes)

sys.path.insert(0,str(ROOT))
from scripts import export_course
from scripts.check_site import Links
author_path=ROOT/'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix13-cumulative.json'
author=read(author_path);author_rows={r['lesson']:r for r in author['records']}
lesson_index=read(ROOT/'course/lesson-index.json')
notebook_pages={p['page_id']:p for p in canon['pages'] if p.get('notebook')}
rows=[];teaching_sections=0;teaching_cells=0;displayed=0
for item in lesson_index:
    current=ROOT/item['notebook'];raw=RAW/Path(item['notebook']).relative_to('notebooks');download=PREVIEW/item['notebook']
    assert sha(current)==notebook_pages[item['id']]['notebook_sha256']
    original=read(current);executed=export_course.checked_notebook(original,raw)
    ar=author_rows[item['id']]
    assert ar['notebook_sha256']==sha(current) and ar['executed_sha256']==sha(raw)
    assert Path(ar['bundle_path']).resolve()==raw.resolve()
    assert sha(Path(ar['source']))==sha(raw)
    assert current.read_bytes()==download.read_bytes()
    html=(PREVIEW/(item['id']+'.html')).read_text();parsed=Links();parsed.feed(html)
    assert item['notebook'] in parsed.links and 'https://colab.research.google.com/' in parsed.links
    assert not any(x.startswith('https://colab.research.google.com/github/') for x in parsed.links)
    assert 'colab-upload-instructions' in parsed.ids
    assert all(x in html for x in ['尚未發布','Upload notebook','File','雲端環境未經本課實測'])
    cells=[c for c in executed['cells'] if c['cell_type']=='code' and not c.get('metadata',{}).get('course_setup') and 'course_figure' not in c.get('metadata',{})]
    teaching_sections+=bool(cells);teaching_cells+=len(cells)
    shown=sum(bool(c.get('outputs')) for c in cells);displayed+=shown
    if cells:assert '實際執行結果' in html,item['id']
    rows.append({'lesson_id':item['id'],'notebook':item['notebook'],'notebook_sha256':sha(current),'raw_path':str(raw.relative_to(ROOT)),
        'raw_sha256':sha(raw),'teaching_cells':len(cells),'displayed_cells':shown,'author_method':ar['method']})
assert len(rows)==331
def http_check(row):
    with urlopen('http://127.0.0.1:8774/'+row['notebook'],timeout=15) as response:
        data=response.read();assert response.status==200
    assert hashlib.sha256(data).hexdigest()==row['notebook_sha256']
    return {'lesson_id':row['lesson_id'],'status':200,'sha256':hashlib.sha256(data).hexdigest()}
http_rows=list(ThreadPoolExecutor(max_workers=6).map(http_check,rows))

# Test the actual current formal gate using existing Git objects; no commit/build.
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
gate_cases=[]
for name,subset,revision,should_pass in [
    ('same_1.1_existing_blob',[next(x for x in lesson_index if x['id']=='1.1')],head,True),
    ('reject_short_ref',lesson_index[:1],'main',False),
    ('reject_missing_commit',lesson_index[:1],'0'*40,False),
    ('reject_tree_as_commit',lesson_index[:1],tree,False),
    ('reject_current331_old_head',lesson_index,head,False),
    ('reject_missing_blob',[{'notebook':'notebooks/__missing_callback__.ipynb'}],head,False),
    ('reject_tree_as_blob',[{'notebook':'notebooks'}],head,False)]:
    try:export_course.verify_colab_notebooks(subset,revision);accepted=True;message=''
    except ValueError as e:accepted=False;message=str(e)
    assert accepted==should_pass,name
    gate_cases.append({'case':name,'index_count':len(subset),'revision':revision,'accepted':accepted,'message':message})
footer=export_course.section_footer(lesson_index[0],read(ROOT/lesson_index[0]['notebook']),head,True,direct_colab=False)
assert 'Upload notebook' in footer and '尚未發布' in footer and '/github/' not in footer

# Re-run the real intake and each current section checker; receipt is not a seal.
commands=[]
def run(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=45)
    row={'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr};commands.append(row);return p
intake_output=OUT/'numbered-intake-callback-v1.json'
p=run(['uv','run','--extra','cpu','python','scripts/omni_revision/review_setup.py','check-numbered','--manifest',
    'docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v4.json','--intake','--output',str(intake_output.relative_to(ROOT))])
for entry in bundle['sections']:
    if entry['lesson_id'].startswith('W.'):
        run(['uv','run','--extra','cpu','python','docs/course-review-20261019/tools/check_technical_report.py',entry['lesson_id'],entry['report']['path'],
             '--freeze',str(SECTION.relative_to(ROOT)),'--reader-task','/root/rv1019_reader_entry'])
run(['uv','run','--extra','cpu','python','scripts/check_site.py','--site',str(PREVIEW.relative_to(ROOT))])

figures=[]
cache=Path('/workspace/work/omni-technical-first-steps');cache.mkdir(exist_ok=True)
for path,digest in page['figures_sha256'].items():
    assert sha(ROOT/path)==digest
    out=cache/(Path(path).stem+'-callback-v1.png')
    p=subprocess.run(['inkscape',str(ROOT/path),'--export-type=png','--export-filename='+str(out)],capture_output=True,text=True)
    assert p.returncode==0
    figures.append({'source':path,'source_sha256':digest,'render_path':str(out),'render_sha256':sha(out),
        'command':['inkscape',path,'--export-type=png','--export-filename='+str(out)],'render_exit_code':p.returncode})

audit={'started_at':started,'created_at':datetime.now(timezone.utc).isoformat(),
 'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/callback_audit_v1.py',
 'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform()},
 'source_freeze':ref(CANON),'section_freeze':ref(SECTION),'source_snapshot':ref(snapshot),'bundle':ref(BUNDLE),'issue_index':ref(INDEX),
 'preserved_original_and_artifacts':preserved,'original_exporter_snapshot':ref(oldexport),'current_repo_dependencies':repo_dependencies,
 'W_sections':entries,'reports_read':report_receipts,'bundle_all26_reference_hashes_checked':True,'issue_inventory_reports':len(index['reports']),
 'issue_inventory_issues':len(index['issues']),'routes':routes,'replay':ref(OUT/'cpu-current-callback-v1.json'),
 'notebook_author_receipt':ref(author_path),'notebook_rows':rows,'http_rows':http_rows,
 'counts':{'notebooks':len(rows),'HTTP200_same_bytes':len(http_rows),'teaching_sections':teaching_sections,'teaching_cells':teaching_cells,'displayed_cells':displayed,'navigation_targets':len(routes),'embedded_blocks_reexecuted':8},
 'preview_build_info':read(PREVIEW/'build-info.json'),'formal_gate_cases':gate_cases,'preview_footer_generated':footer,'commands':commands,'figure_renders':figures,
 'limits':['Same-owner informed callback after preserved independent original; not a new blind reader.',
 'W.1 is the actual dependency-only reviewer; W.2–W.7 are distinct actual section tasks named in this v3 bundle, not aliases for this collector.',
 '331 current/executed/download identities checked and all331 HTTP downloads checked; no fresh execution of331kernels or full subject-matter reviews.',
 'Manual-upload preview and conditional formal Git gate checked; no upload/login/Connect/cloud cell, commit, model training, GPU or deployment.',
 'Intake only schema_checked_collection_pending; first-steps complete reader wave02 and final seals remain foreign-stage pending.']}
target=save('callback-audit-v1.json',audit)
print(json.dumps({'artifact':ref(target),'counts':audit['counts'],'commands':[(r['exit_code'],r['stdout'][:180]) for r in commands],
 'figures':figures,'result':'Bounded exact current recheck completed; inspect all command receipts before verdict.'},ensure_ascii=False))
