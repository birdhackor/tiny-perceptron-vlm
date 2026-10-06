"""Correct the recorded end line using actual raw section lines, preserving prior records."""
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report_path=ROOT/'docs/technical-reviews/A.7.json'
receipt_path=HERE/'reinspection-receipt.json'
for source,target in [(report_path,HERE/'report-before-line-scope-fix.opaque.json'),
                      (receipt_path,HERE/'receipt-before-line-scope-fix.opaque.json')]:
 assert not target.exists()
 shutil.copyfile(source,target)
 assert sha(source)==sha(target)
report=json.loads(report_path.read_bytes()); receipt=json.loads(receipt_path.read_bytes())
assert report['reviewer_task']==receipt['reviewer_task']=='/root/phase4_factual_coordinator/factual_a_7'
body=(HERE/'current-section.md').read_bytes()
actual_end=223+len(body.splitlines())-1
assert actual_end==252
receipt['source_bytes_read']='Entire current A.7 raw bytes, lines 223–252, including unchanged paragraphs and fence'
receipt['scope_record_correction']='Initial receipt end line253 was a bookkeeping error; actual raw30-line section beginning223 ends252. Corrected from exact raw splitlines count, without any change to source, claims, evidence or judgment. Before-correction receipt/report preserved opaque.'
extra=[]
for identifier,file,kind,description in [
 ('a7_line_scope_correction_code','fix_read_scope_line.py','code','Executed exact raw-line scope correction; no claim/evidence change'),
 ('a7_pre_scope_fix_report','report-before-line-scope-fix.opaque.json','source_snapshot','Own reinspection report before line-number bookkeeping correction'),
 ('a7_pre_scope_fix_receipt','receipt-before-line-scope-fix.opaque.json','source_snapshot','Own reinspection receipt before line-number bookkeeping correction')]:
 p=HERE/file
 extra.append({'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'kind':kind,'description':description})
receipt['actual_new_evidence_artifacts'] += [{'id':a['id'],'path':a['path'],'sha256':a['sha256']} for a in extra]
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report['read_scope'][0]='Complete current A.7 raw bytes, lines 223–252'
report['artifacts']+=extra
next(a for a in report['artifacts'] if a['id']=='a7_same_owner_reinspection_receipt')['sha256']=sha(receipt_path)
report['reinspection_history'][-1]['receipt_sha256']=sha(receipt_path)
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
saved=json.loads(report_path.read_bytes())
assert saved['source_sha256']=='32397500c886a31262e690ab72dd5ba34e5fc2936e5dbac9bebb9547d23bb58d'
assert saved['reinspection_history'][-1]['receipt_sha256']==sha(receipt_path)
print(json.dumps({'report_sha256':sha(report_path),'source_sha256':saved['source_sha256'],
 'receipt_id':'a7_same_owner_reinspection_receipt','receipt_path':receipt_path.relative_to(ROOT).as_posix(),
 'receipt_sha256':sha(receipt_path),'read_scope':'lines223–252','verdict':saved['verdict']},ensure_ascii=False,indent=2))
