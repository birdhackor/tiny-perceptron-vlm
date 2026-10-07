"""Read only this owner's new report/evidence; does not approve truth or stage."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'docs/review-tools'))
import phase7_review as review
from scripts import check_technical_reviews as technical
path=R/sys.argv[1]
q=json.loads(path.read_text())
m=json.loads((R/'docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json').read_text())
g=next(g for g in m['groups']if g['group']=='f')
meta={p['page_id']:p for p in m['pages']}
assert [p['page_id']for p in q['pages']]==g['primary_page_ids']
assert q['verdict']=='revise'
rows=review.verify_trace(R,q['trace_files'][0],q['reviewer_task'],'technical','f',m['batch_id'])
results=[]
for p in q['pages']:
 e=[];a=technical._artifacts(R,p['artifacts'],e);s=technical._sources(R,p['sources'],a,e);c=technical._claims(p['claims'],s,a,e)
 for k in technical.CHECKS:
  v=p['checks'].get(k,{});technical._references(v.get('claim_ids'),c,e,k)
  if not technical._text(v.get('details')):e.append('check missing actual details:'+k)
 assert p['source_sha256']==meta[p['page_id']]['source_sha256']
 assert p['figures_sha256']==meta[p['page_id']]['figures_sha256']
 visuals=[]
 for f in p['figures_sha256']:
  checked=[v for v in p['visual_checks']if v.get('figure')==f];fail=[];ok=False
  for v in checked:
   try:ok=review.visual_check(R,v,q['reviewer_task'],figure=f)or ok
   except Exception as x:fail.append(str(x))
  if not ok:e.append('figure has no valid actual receipt:'+f+':'+str(fail))
  visuals.append({'figure':f,'receipt_metadata_matches':ok,'retained_earlier_receipt_errors':fail})
 try:layout=review.visual_check(R,p['page_visual_check'],q['reviewer_task'])
 except Exception as x:layout=False;e.append('layout receipt:'+str(x))
 if not layout and(p['figures_sha256']or p['page_visual_check'].get('required')is not False):e.append('necessary layout unsupported')
 results.append({'page_id':p['page_id'],'schema_errors':e,'actual_layout_receipt_metadata':layout,'figures':visuals,'verdict':p['verdict'],'open_issue_ids':[i['id']for i in p['issues']],'check_statuses':{k:v['status']for k,v in p['checks'].items()}})
out=dict(scope='Only owner report schema/path/hash/actual receipt bindings and own complete trace; not truth verification, issue resolution, collection, full-stage gate or pass.',report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),primary_pages=len(results),trace_events=len(rows),schema_error_count=sum(len(x['schema_errors'])for x in results),results=results)
print(json.dumps(out,ensure_ascii=False))
sys.exit(bool(out['schema_error_count']))
