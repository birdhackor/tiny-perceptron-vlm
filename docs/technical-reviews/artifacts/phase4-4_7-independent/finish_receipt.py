"""Persist this review's actual checker result and initial failed audit provenance."""
from pathlib import Path
import hashlib,json,subprocess,sys,time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env=json.loads((BASE/"environment.json").read_text())
initial={"command":".venv/bin/python docs/technical-reviews/artifacts/phase4-4_7-independent/verify.py","exit_code":1,"code_at_execution_path_sha256":sha(BASE/"verify-initial.py"),"code_preserved_as":"verify-initial.py","stdout_sha256":sha(BASE/"verify-initial.stdout.txt"),"stderr_sha256":sha(BASE/"verify-initial.stderr.txt"),"environment":env,"failure":"Audit script expected whole current helper hashes to match old recorded run, but unrelated source updates differ; original Git revision helper snapshots were then read and reviewed contracts checked equal. Not a curriculum defect.","elapsed_seconds":"not recorded for initial failed audit"}
(BASE/"initial-audit-receipt.json").write_text(json.dumps(initial,indent=2)+"\n")
# Rebuild inventory now that this runner and initial receipt exist.
subprocess.run([str(ROOT/".venv/bin/python"),str(BASE/"write_report.py")],cwd=ROOT,check=True)
argv=[str(ROOT/".venv/bin/python"),"scripts/check_technical_reviews.py","--lesson","4.7"]
started=time.perf_counter()
result=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=30)
for stream in ["stdout","stderr"]:
    (BASE/f"checker.{stream}.txt").write_bytes(getattr(result,stream))
receipt={"command_argv":argv,"cwd":str(ROOT),"exit_code":result.returncode,"elapsed_seconds":time.perf_counter()-started,"environment":env,"checker_code_sha256":sha(ROOT/"scripts/check_technical_reviews.py"),"report_sha256":sha(ROOT/"docs/technical-reviews/4.7.json"),"stdout_sha256":sha(BASE/"checker.stdout.txt"),"stderr_sha256":sha(BASE/"checker.stderr.txt"),"stdout":result.stdout.decode(),"stderr":result.stderr.decode()}
(BASE/"checker-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(result.stdout.decode(),end="")
print(result.stderr.decode(),end="",file=sys.stderr)
result.check_returncode()
