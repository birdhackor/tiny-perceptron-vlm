"""Owner-supplied callback notes, appended only to the real new session."""
import json
import subprocess
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[4]
session="f6f5240dffca4aeca152b6afb8dca93d"
q=json.load(sys.stdin)
for a in q.pop("attach_figures",[]):
    p=R/a["receipt"];r=json.loads(p.read_text())
    q.setdefault("visual_checks",[]).append(dict(figure=a["figure"],source_sha256=r["source_sha256"],status="verified",details=a["details"],observation=r["observation"],receipt=dict(path=a["receipt"],sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest()),artifacts=r["artifacts"]))
folder=R/'outputs/reader-checkpoints/p7_technical_f/recheck04'
folder.mkdir(parents=True,exist_ok=True)
p=folder/(q['page_id']+'-'+str(q['unit_index'])+'.json')
with p.open('x')as f:f.write(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
r=subprocess.run([str(R/'.venv/bin/python'),'docs/review-tools/phase7_review.py','next',session,'--checkpoint',str(p)],cwd=R,text=True,capture_output=True)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr);sys.exit(r.returncode)
