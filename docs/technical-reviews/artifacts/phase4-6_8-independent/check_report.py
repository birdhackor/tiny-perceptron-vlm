import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
command=['.venv/bin/python','scripts/check_technical_reviews.py','--lesson','6.8']
result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
(OUT/'checker.stdout.txt').write_text(result.stdout)
(OUT/'checker.stderr.txt').write_text(result.stderr)
receipt={'command_argv':command,'cwd':str(ROOT),'exit_code':result.returncode,'scope':'Own lesson6.8 checker; substantive unresolved issue deliberately retained. Schema gate cannot judge factual truth.','stdout_sha256':hashlib.sha256((OUT/'checker.stdout.txt').read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256((OUT/'checker.stderr.txt').read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256((ROOT/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((ROOT/'docs/technical-reviews/6.8.json').read_bytes()).hexdigest(),'environment':json.loads((OUT/'probe.environment.json').read_text())}
(OUT/'checker-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print(result.stdout)
assert result.returncode==1
expected=['6.8: 審閱要求修訂／尚未核實','6.8: claim c4: 未核實／矛盾主張不能通過，應維持revise','6.8: 尚有未解決問題，不能記為pass','6.8: factual_accuracy: 未通過或不適用的NA']
assert result.stdout.splitlines()==expected,result.stdout
