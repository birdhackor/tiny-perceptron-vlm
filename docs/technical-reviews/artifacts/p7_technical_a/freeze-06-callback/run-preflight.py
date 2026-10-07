import json,subprocess,sys,hashlib,datetime,shlex
from pathlib import Path
manifest='docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json'
report='docs/technical-reviews/artifacts/p7_technical_a/group-a-technical-freeze06-callback-20261007.json'
command=['python','docs/course-revision-20261007-phase7/authoring/group-preflight.py','--manifest',manifest,'--report',report]
version=subprocess.check_output(['python','--version'],text=True).strip()
completed=subprocess.run(command,text=True,capture_output=True)
proof={'tool':'tools.exec_command -> saved owner script subprocess','command':shlex.join(command),'argv':command,'environment':{'python_for_actual_preflight':version,'device':'CPU metadata only'},'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'reviewer_task':'/root/p7_technical_a','saved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(Path(report).read_bytes()).hexdigest(),'scope':'Actual single-group metadata preflight, not scientific truth, not full stage gate or collection'}
p=Path('docs/technical-reviews/artifacts/p7_technical_a/freeze-06-callback/group-preflight-actual.json')
with p.open('x',encoding='utf-8') as f:f.write(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(completed.stdout,end='');print(completed.stderr,end='',file=sys.stderr);print('proof',str(p),'sha256',hashlib.sha256(p.read_bytes()).hexdigest());sys.exit(completed.returncode)
