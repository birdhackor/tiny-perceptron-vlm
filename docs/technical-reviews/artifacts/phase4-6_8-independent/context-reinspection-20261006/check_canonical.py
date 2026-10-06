import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
argv=['.venv/bin/python','scripts/check_technical_reviews.py','--lesson','6.8']
result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,check=False)
(OUT/'checker.stdout.txt').write_text(result.stdout)
(OUT/'checker.stderr.txt').write_text(result.stderr)
receipt={'command_argv':argv,'cwd':str(ROOT),'exit_code':result.returncode,'report_sha256':sha(ROOT/'docs/technical-reviews/6.8.json'),'own_receipt_code_sha256':sha(Path(__file__)),'checker_sha256':sha(ROOT/'scripts/check_technical_reviews.py'),'stdout_sha256':sha(OUT/'checker.stdout.txt'),'stderr_sha256':sha(OUT/'checker.stderr.txt'),'reinspection_receipt_path':(OUT/'context-reinspection-receipt.json').relative_to(ROOT).as_posix(),'reinspection_receipt_sha256':sha(OUT/'context-reinspection-receipt.json'),'environment':json.loads((OUT/'context-input-and-render-receipt.json').read_text())['environment'],'scope':'Singlelesson6.8 checker afterowncanonicalnecessarycontextupdate; priorhistoryretained, no otherreportverdicttakenasevidence.'}
(OUT/'checker-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print(result.stdout)
assert result.returncode==0,result.stdout+result.stderr
