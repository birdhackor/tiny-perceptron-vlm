"""Owner's metadata-only inspection, preserving necessary revise findings."""
import importlib.util,json,hashlib
from pathlib import Path
root=Path.cwd()
spec=importlib.util.spec_from_file_location('own_review_metadata',root/'docs/review-tools/phase7_review.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
p=root/'docs/technical-reviews/artifacts/p7_technical_c/group-c-initial-report-format-fixed.json';r=json.loads(p.read_text());m=json.loads((root/'docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json').read_text());meta={p['page_id']:p for p in m['pages']};errors=[];trace_checks=[]
for t in r['trace_files']:
 try:
  rows=v.verify_trace(root,t,r['reviewer_task'],'technical','c','phase7');trace_checks.append({'path':t['path'],'sha256':t['sha256'],'events':len(rows),'completed':rows[-1]['event']=='complete'})
 except Exception as e:errors.append({'trace':t['path'],'error':str(e)})
for page in r['pages']:
 try:v.technical_page(root,page,(root/meta[page['page_id']]['snapshot']).read_text())
 except Exception as e:errors.append({'page':page['page_id'],'check':'technical_fields','error':str(e)})
 for visual in page.get('visual_checks',[]):
  try:v.visual_check(root,visual,r['reviewer_task'],figure=visual['figure'])
  except Exception as e:errors.append({'page':page['page_id'],'check':'figure_receipt','error':str(e)})
 try:
  ok=v.visual_check(root,page['page_visual_check'],r['reviewer_task'])
  if not ok and (page['figures_sha256'] or page['page_visual_check'].get('required') is not False):errors.append({'page':page['page_id'],'check':'page_receipt','error':'necessary layout missing'})
 except Exception as e:errors.append({'page':page['page_id'],'check':'page_receipt','error':str(e)})
print(json.dumps({'scope':'Separate own field/hash/trace/receipt metadata inspection; report verdict never mutated; not group preflight/full-stage gate/truth review.','report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'traces':trace_checks,'page_count':len(r['pages']),'errors':errors},ensure_ascii=False,indent=2))
