"""Bounded source14 callback; reuse unchanged whole-page science, verify changed deliveries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
REVIEW=ROOT/'docs/course-revision-20261029-omni/reviews'
PREVIEW=ROOT/'outputs/omni-review-site-technical-fix14'
RAW=ROOT/'outputs/omni-notebooks-technical-fix14-current'
CACHE=Path('/workspace/work/omni-technical-first-steps/source14');CACHE.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)}
def read(p):return json.loads(p.read_bytes())
def verify(r):
 p=ROOT/r['path'];assert sha(p)==r['sha256'],r['path'];return p
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

started=datetime.now(timezone.utc).isoformat()
canonical_path=REVIEW/'source-freeze-technical-fix14.json';canonical=read(canonical_path)
section_path=REVIEW/'numbered-freeze-v12.json';section=read(section_path)
oldcanon=read(REVIEW/'source-freeze-technical-fix13.json');oldsec=read(REVIEW/'numbered-freeze-v11.json')
bundle_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v4.json';bundle=read(bundle_path)
issue_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-issues-v4.json';issues=read(issue_path)
assert sha(bundle_path)=='1ddabe842cbb17502ae1fb08783825c16f5669c22b8c294c0e9f83046c8b19da'
assert sha(issue_path)=='187b4b06600c9898a94243d8673697ee9464fbbbbfc2222cb5e969efae9377be'
own_current_path=REVIEW/'technical/technical_first_steps/technical-current-v1.json';own=read(own_current_path)
assert sha(own_current_path)=='cb54d25acd61156497bff2c95d95b85ed4f0d58b9585d8908137d021b4e8a26a'
preserved=[]
for p in [REVIEW/'technical/technical_first_steps/technical-original-v1.json',own_current_path]:preserved.append(ref(p))
for a in own['artifacts']:verify(a)
page=next(x for x in canonical['pages'] if x['page_id']=='first-steps')
oldpage=next(x for x in oldcanon['pages'] if x['page_id']=='first-steps')
assert page['source_sha256']==oldpage['source_sha256']==sha(ROOT/page['source'])==sha(ROOT/page['snapshot'])
assert page['figures_sha256']==oldpage['figures_sha256']
for p,digest in page['figures_sha256'].items():assert sha(ROOT/p)==digest
unchanged_sections=[]
for new in section['pages']:
 if new['page_id'].startswith('W.'):
  old=next(x for x in oldsec['pages'] if x['page_id']==new['page_id'])
  assert new['source_sha256']==old['source_sha256'] and new['figures_sha256']==old['figures_sha256']
  assert sha(ROOT/new['snapshot'])==new['source_sha256']
  unchanged_sections.append({'lesson_id':new['page_id'],'source_sha256':new['source_sha256'],'figures_sha256':new['figures_sha256']})
repo_sources=[]
for s in own['sources']:
 if s['kind']=='repository_code':
  assert sha(ROOT/s['path'])==s['sha256'];repo_sources.append({'id':s['id'],'path':s['path'],'sha256':s['sha256'],'unchanged_since_current1':True})
old_by_id={p['page_id']:p for p in oldcanon['pages']};canonical_delta=[]
for p in canonical['pages']:
 changed={k:{'old':old_by_id[p['page_id']].get(k),'new':p.get(k)} for k in ['source_sha256','figures_sha256','notebook','notebook_sha256'] if old_by_id[p['page_id']].get(k)!=p.get(k)}
 if changed:canonical_delta.append({'page_id':p['page_id'],'changes':changed})
assert [x['page_id'] for x in canonical_delta]==['12.2']

# Verify the real restricted collection, rather than the full raw ledger as canonical.
collection_path=verify(bundle['actual_collection_snapshot']);selected=read(collection_path)
raw_collection_path=verify(selected['raw_collection_snapshot']);raw_collection=read(raw_collection_path)
assert len(bundle['sections'])==26 and len(issues['reports'])==35 and len(issues['issues'])==5
assert len(selected['collection_index'])==45 and all(r in raw_collection['collection_index'] for r in selected['collection_index'])
section_ids={e['lesson_id'] for e in bundle['sections']}
assert all(r['lesson_id'] in section_ids for r in selected['collection_index'])
for r in issues['reports']:verify(r)
changed_reports=[]
oldbundle=read(ROOT/'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v3.json')
for e in bundle['sections']:
 for key in ['report','original_report','original_section_freeze']:verify(e[key])
 old=next(x for x in oldbundle['sections'] if x['lesson_id']==e['lesson_id'])
 if e['report']!=old['report']:
  r=read(ROOT/e['report']['path'])
  for a in r['artifacts']:verify(a)
  changed_reports.append({'lesson_id':e['lesson_id'],'report':e['report'],'reviewer_task':r['reviewer_task'],'created_at':r['created_at'],
     'claims':len(r['claims']),'verdict':r['verdict'],'issues':r['issues'],'read_scope':'Updated dependency claims, all current evidence references and preserved issue chain; no new independent science/blind read claimed.'})
assert {r['lesson_id'] for r in changed_reports}=={'W.1','R.1'}
w1_entry=next(e for e in bundle['sections'] if e['lesson_id']=='W.1')
w1=read(ROOT/w1_entry['report']['path'])
bad_path=REVIEW/'technical/technical_w_1/technical-current-v4.json';bad=read(bad_path)
format_receipt_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-w-1-v1/schema-current-v4.json'
assert read(format_receipt_path)['exit_code']==1
for key in ['claims','sources','issues','checks','review_scope']:assert bad[key]==w1[key],key
w1proof_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-w-1-v1/fix14-current-recheck-v3.json';w1proof=read(w1proof_path)
for key in ['initial_failure','diagnosis','DOM_comparison_source','DOM_negative_controls','second_failure']:
 verify(w1proof['new_test_failure_and_diagnosis'][key])
assert w1['issues'][0]['status']=='resolved' and w1['issues'][0]['id']=='w1-current-dependency-drift'
assert w1['issues'][0]['details']==read(REVIEW/'technical/technical_w_1/technical-current-v1.json')['issues'][0]['details']
r1=read(ROOT/next(e['report']['path'] for e in bundle['sections'] if e['lesson_id']=='R.1'))
assert r1['issues'][0]['status']=='resolved' and r1['issues'][0]['details']==read(REVIEW/'technical/technical_r_1/technical-current-v2.json')['issues'][0]['details']

# Read actual new reviewer-only code and exercise independent positive/negative inputs.
dom_path=ROOT/'docs/technical-reviews/artifacts/omni-technical-w-1-v1/output_dom_v1.py'
assert sha(dom_path)=='55e67c3e848cb5f7aad28da80915cfb16f87ac5bec1271eb64a46691323c4597'
dom=load('bounded_output_dom',dom_path)
sample='<div class="output"><p class="output-label">實際執行結果 · CPU</p><pre>0.64755\n</pre><pre>second\n</pre><img src="data:image/png;base64,abc" alt="chart"></div>'
expected=dom.outputs(sample)
controls=[]
tests=[('wrapper_newlines_only',sample.replace('><','>\n<'),True),
 ('changed_number',sample.replace('0.64755','0.64756'),False),
 ('missing_pre',sample.replace('<pre>second\n</pre>',''),False),
 ('changed_pre_whitespace',sample.replace('0.64755\n','0.64755 \n'),False),
 ('changed_image_src',sample.replace('base64,abc','base64,abcd'),False),
 ('changed_image_alt',sample.replace('alt="chart"','alt="other"'),False),
 ('changed_block_order',sample.replace('<pre>0.64755\n</pre><pre>second\n</pre>','<pre>second\n</pre><pre>0.64755\n</pre>'),False),
 ('missing_label',sample.replace('<p class="output-label">實際執行結果 · CPU</p>',''),False),
 ('unexpected_visible_text',sample.replace('</div>','unexpected</div>'),False)]
for name,value,equal in tests:
 try:observed=dom.outputs(value)==expected;message=''
 except AssertionError as exc:observed=False;message=str(exc)
 assert observed==equal,name
 controls.append({'case':name,'expected_equal':equal,'observed_equal':observed,'rejection':message})

sys.path.insert(0,str(ROOT))
from scripts import export_course
from scripts.reading_time import load_estimates
from scripts.check_site import Links
index=read(ROOT/'course/lesson-index.json')
author_path=ROOT/'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json';author=read(author_path)
assert author['reexecuted_this_fix']==['12.2'] and author['reused_matching_cells']==330
author_rows={r['lesson']:r for r in author['records']}
canon_notebooks={p['page_id']:p for p in canonical['pages'] if p.get('notebook')}
rows=[];raw_string_differences=[]
def joined(x):return ''.join(x) if isinstance(x,list) else x
def desired_blocks(cells):
 result=[]
 for cell in cells:
  parts=[]
  for output in cell.get('outputs',[]):
   kind=output['output_type'];assert kind!='error'
   if kind=='stream':parts.append(['pre',joined(output['text'])])
   elif kind in ['display_data','execute_result']:
    data=output['data']
    if 'image/png' in data:parts.append(['image','data:image/png;base64,'+joined(data['image/png']),'本段程式的實際輸出圖'])
    elif 'text/plain' in data:parts.append(['pre',joined(data['text/plain'])])
  if parts:result.append([['label','實際執行結果 · CPU'],*parts])
 return result
for item in index:
 current=ROOT/item['notebook'];raw=RAW/Path(item['notebook']).relative_to('notebooks');download=PREVIEW/item['notebook']
 assert sha(current)==canon_notebooks[item['id']]['notebook_sha256']
 original=read(current);executed=export_course.checked_notebook(original,raw);ar=author_rows[item['id']]
 assert sha(current)==ar['notebook_sha256'] and sha(raw)==ar['executed_sha256']==sha(Path(ar['source']))
 assert Path(ar['bundle_path']).resolve()==raw.resolve() and current.read_bytes()==download.read_bytes()
 cells=[c for c in executed['cells'] if c['cell_type']=='code' and not c.get('metadata',{}).get('course_setup') and 'course_figure' not in c.get('metadata',{})]
 html=(PREVIEW/(item['id']+'.html')).read_text();parser=Links();parser.feed(html)
 assert item['notebook'] in parser.links and 'https://colab.research.google.com/' in parser.links
 assert not any(x.startswith('https://colab.research.google.com/github/') for x in parser.links)
 assert 'colab-upload-instructions' in parser.ids and '尚未發布' in html and 'Upload notebook' in html
 desired=desired_blocks(cells);observed=dom.outputs(html);assert desired==observed,item['id']
 strings=[export_course.output_html(c.get('outputs',[])) for c in cells]
 if any(value and value not in html for value in strings):raw_string_differences.append({'lesson_id':item['id'],'DOM_content_order_equal':True,'canonical_DOM_sha256':hashlib.sha256(json.dumps(desired,ensure_ascii=False).encode()).hexdigest()})
 rows.append({'lesson_id':item['id'],'notebook':item['notebook'],'source_sha256':sha(current),'raw_sha256':sha(raw),'teaching_cells':len(cells),'displayed_cells':len(desired),'output_DOM_exact':True,'method':ar['method']})
assert len(rows)==331 and {r['lesson_id'] for r in raw_string_differences}=={'15.8','19.6'}
def http_row(row):
 with urlopen('http://127.0.0.1:8775/'+row['notebook'],timeout=15) as response:data=response.read();status=response.status
 assert status==200 and hashlib.sha256(data).hexdigest()==row['source_sha256']
 return {'lesson_id':row['lesson_id'],'status':status,'sha256':hashlib.sha256(data).hexdigest()}
http=list(ThreadPoolExecutor(max_workers=6).map(http_row,rows))
kernel_path=ROOT/'outputs/omni-notebooks-technical-fix14-run/validation-kernel.json'
kernel=read(kernel_path);new_raw=ROOT/'outputs/omni-notebooks-technical-fix14-run/12/12.2.ipynb'
assert sha(new_raw)==sha(RAW/'12/12.2.ipynb')
old12=read(ROOT/'outputs/omni-notebooks-technical-fix13-current/12/12.2.ipynb');new12=read(new_raw)
assert [c['source'] for c in old12['cells'] if c['cell_type']=='code']==[c['source'] for c in new12['cells'] if c['cell_type']=='code']

base_times_path=ROOT/'course/reading-times.json';base_times=read(base_times_path)
preview_times_path=ROOT/'outputs/omni-reading-times-technical-fix14-preview.json';preview_times=read(preview_times_path)
base_rows={r['page_id']:r for r in base_times['pages']};preview_rows={r['page_id']:r for r in preview_times['pages']}
assert set(base_rows)-set(preview_rows)=={'first-steps','12.2'} and len(preview_rows)==367
assert all(value==base_rows[key] for key,value in preview_rows.items())
assert all(preview_times.get(k)==v for k,v in base_times.items() if k!='pages')
assert set(preview_times)-set(base_times)=={'preview_scope'}
assert len(load_estimates(preview_times_path,canonical,False))==367
time_cases=[]
for lesson in ['first-steps','12.2']:
 fixture={**base_times,'pages':[base_rows[lesson]]};path=CACHE/('stale-'+lesson+'.json');path.write_text(json.dumps(fixture))
 try:load_estimates(path,canonical,False);raise AssertionError('stale accepted')
 except ValueError as e:time_cases.append({'case':'reject_stale','page_id':lesson,'message':str(e)})
try:load_estimates(preview_times_path,canonical,True);raise AssertionError('incomplete formal accepted')
except ValueError as e:time_cases.append({'case':'reject_incomplete_formal','message':str(e)})
rendered_time_pages=[p.stem for p in PREVIEW.glob('*.html') if '<p class="reading-time">' in p.read_text()]
assert len(rendered_time_pages)==367 and 'first-steps' not in rendered_time_pages and '12.2' not in rendered_time_pages
info=read(PREVIEW/'build-info.json');assert info['colab_mode']=='upload' and info['publication_ready'] is False and info['reading_time_pages']==367

commands=[]
def run(args):
 p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=45);commands.append({'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
intake_output=OUT/'numbered-intake-source14-v1.json'
run(['uv','run','--extra','cpu','python','scripts/omni_revision/review_setup.py','check-numbered','--manifest',str((REVIEW/'technical-stage/numbered-intake-manifest-v5.json').relative_to(ROOT)),'--intake','--output',str(intake_output.relative_to(ROOT))])
run(['uv','run','--extra','cpu','python','docs/course-review-20261019/tools/check_technical_report.py','W.1',w1_entry['report']['path'],'--freeze',str(section_path.relative_to(ROOT)),'--reader-task','/root/rv1019_reader_entry'])
run(['uv','run','--extra','cpu','python','scripts/check_site.py','--site',str(PREVIEW.relative_to(ROOT))])

audit={'started_at':started,'created_at':datetime.now(timezone.utc).isoformat(),'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/callback_source14_v1.py',
 'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform()},
 'canonical_source_freeze':ref(canonical_path),'section_freeze':ref(section_path),'bundle':ref(bundle_path),'issue_index':ref(issue_path),
 'preserved_own_reports':preserved,'unchanged_first_steps_source':page['source_sha256'],'unchanged_W_sections':unchanged_sections,
 'unchanged_repo_sources':repo_sources,'canonical_delta':canonical_delta,'changed_owner_reports':changed_reports,
 'collection':{'selected':ref(collection_path),'selected_rows':45,'canonical_section_scope':26,'selected_rows_are_verbatim_raw_subset':True,
   'raw_ledger':ref(raw_collection_path),'raw_rows':len(raw_collection['collection_index']),'closed_scope':'Only the26numbered-section intake; not the369canonical-page collection or a complete technical stage/seal.'},
 'W1_format_correction':{'invalid_raw':ref(bad_path),'failure_receipt':ref(format_receipt_path),'effective_current':w1_entry['report'],'claims_sources_issues_checks_scope_unchanged':True},
 'preserved_W1_failures_and_diagnoses':w1proof['new_test_failure_and_diagnosis'],'DOM_parser':ref(dom_path),'own_controls':controls,
 'author_notebook_receipt':ref(author_path),'all331_rows':rows,'HTTP331':http,'serialized_wrapper_only_differences':raw_string_differences,
 'new12_2_kernel':{'validation':ref(kernel_path),'result':kernel,'raw':ref(new_raw),'same_bytes_as_current_bundle':True,'code_cells_same_as_fix13':True},
 'metadata':{'original':ref(base_times_path),'preview':ref(preview_times_path),'author_filter_receipt':ref(ROOT/'docs/course-revision-20261029-omni/evidence/preview-estimates-technical-fix14-v1.json'),
   'retained_rows':367,'all_retained_rows_exactly_same':True,'omitted_stale':['first-steps','12.2'],'only_extra_field':'preview_scope','guards':time_cases,'rendered_current_estimate_pages':len(rendered_time_pages)},
 'preview_build_info':info,'commands':commands,
 'scope':['Informed bounded source14/bundle4 callback; unchanged first-steps/W.1–W.7 content, all figures,102science/software claims and earlier actual visual/CPU evidence are reused explicitly.',
 'Only new reviewer DOM comparator and changed output/delivery/receipt dependencies tested; no full210tool audit or new blind reading claimed.',
 '331current/raw/site/HTTPidentities and exact label/pre-text/image-src/alt/order checked; only wrappers outside blocks normalize whitespace. Not331newkernels.',
 '1new12.2author kernel plus330matching source reuse; first-steps/12.2 fullreader wave02 remains pending.367filtered rows are not new estimates or complete formal publication.',
 'All immutable failure reports and issue chains remain intact. No checker/source edits, Colabcloud, formaltraining, GPU, paid work, commit, build or deployment.']}
target=OUT/'callback-source14-v1.json'
with target.open('x') as f:json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'artifact':ref(target),'counts':{'notebooks':len(rows),'HTTP':len(http),'DOMcontrols':len(controls),'selected_receipts':45,'estimates':367},'wrapper_differences':raw_string_differences,
 'commands':[(x['exit_code'],x['stdout'][:200]) for x in commands]},ensure_ascii=False))
