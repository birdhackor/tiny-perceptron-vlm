"""Actual end-of-factual administrative/version/schema gates; no claim adjudication."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,platform
root=Path(__file__).resolve().parents[4]
dir=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
progress_path=root/'docs/course-revision-20261005/factual-review-progress.json'
reader_path=root/'docs/course-revision-20261005/review-progress.json'
p=json.loads(progress_path.read_bytes());r=json.loads(reader_path.read_bytes())
ids=p['section_order'];assert len(ids)==309 and len(set(ids))==309
assert all(p['records'][i]['status']=='pass' and p['records'][i]['report_version_current'] and p['records'][i]['checker_exit_code']==0 for i in ids)
assert all(r['records'][i]['status']=='pass' and not r['records'][i]['format_validation_issues'] for i in ids)
assert all(not p['records'][i].get('dependency_recheck_pending',False) for i in ids)
assert all(x['status']=='pass' for i in ids for x in p['records'][i]['readability_callbacks'])
assert len(p['chapter_acceptances'])==27
assert {i for a in p['chapter_acceptances'].values() for i in a['sections']}==set(ids)
initials=[d for i in ids for d in p['records'][i]['dispatches']]
assert len({d['reviewer_task'] for d in initials})==len(initials)
valid=[d for d in initials if not d['status'].startswith('void_')]
assert len(valid)==309 and all(d['fork_turns']=='none' for d in valid)
receipt={'started_at':datetime.datetime.now(datetime.UTC).isoformat(),'scope':'Current report declarations, actual coordinator dispatch ledger, schema/source/evidence bytes and current reader trace/version checks. These tools do not establish scientific truth, firsthand source inspection or agent dispatch on their own. Actual independent owners and callbacks were separately received by the coordinator.','environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform()},'input_progress_sha256':sha(progress_path.read_bytes()),'input_reader_progress_sha256':sha(reader_path.read_bytes()),'current_sections':len(ids),'actual_recorded_initial_tasks':len(initials),'current_unique_fork_none_owners':len(valid),'retained_void_status_tasks':len(initials)-len(valid),'commands':[],'status':'running'}
commands=[('technical-schema',[sys.executable,'scripts/check_technical_reviews.py']),('reader-schema',[sys.executable,'scripts/check_course_reviews.py']),('whole-round-version',[sys.executable,'docs/review-tools/check_review_round.py','--stage','all','--output',str(dir/'whole-round-version.json')]),('factual-dispatch-artifact-version',[sys.executable,'docs/review-tools/audit_factual_round.py','--output',str(dir/'factual-dispatch-artifact-version.json')])]
for name,cmd in commands:
 started=datetime.datetime.now(datetime.UTC).isoformat();process=subprocess.run(cmd,cwd=root,capture_output=True)
 out=dir/(name+'.stdout.txt');err=dir/(name+'.stderr.txt');out.write_bytes(process.stdout);err.write_bytes(process.stderr)
 receipt['commands'].append({'name':name,'command':cmd,'cwd':str(root),'started_at':started,'completed_at':datetime.datetime.now(datetime.UTC).isoformat(),'exit_code':process.returncode,'stdout_path':str(out.relative_to(root)),'stdout_sha256':sha(process.stdout),'stderr_path':str(err.relative_to(root)),'stderr_sha256':sha(process.stderr)})
 receipt['status']='failed' if process.returncode else 'running';(dir/'gate-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'gate':name,'exit_code':process.returncode,'stdout':process.stdout.decode(errors='replace')},ensure_ascii=False),flush=True)
 if process.returncode:raise SystemExit(process.returncode)
assert progress_path.read_bytes() and sha(progress_path.read_bytes())==receipt['input_progress_sha256']
assert sha(reader_path.read_bytes())==receipt['input_reader_progress_sha256']
receipt.update(status='passed',completed_at=datetime.datetime.now(datetime.UTC).isoformat())
receipt['verification_program_sha256']=sha(Path(__file__).read_bytes())
(dir/'gate-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','sections':309,'gate_count':4},ensure_ascii=False))
