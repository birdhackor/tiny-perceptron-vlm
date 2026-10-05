import hashlib
import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path(__file__).parent
command=[str(ROOT/'.venv/bin/python'),'scripts/check_technical_reviews.py','--lesson','6.1']
completed=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=20)
(OUT/'checker.stdout.txt').write_bytes(completed.stdout)
(OUT/'checker.stderr.txt').write_bytes(completed.stderr)
sha=lambda b:hashlib.sha256(b).hexdigest()
receipt={'command_argv':command,'cwd':str(ROOT),'exit_code':completed.returncode,
 'executed_at':datetime.now(UTC).isoformat(),'environment':{'python':sys.version,'device':'cpu'},
 'source_sha256':json.loads((ROOT/'docs/technical-reviews/6.1.json').read_text())['source_sha256'],
 'report_sha256':sha((ROOT/'docs/technical-reviews/6.1.json').read_bytes()),
 'checker_sha256':sha((ROOT/'scripts/check_technical_reviews.py').read_bytes()),
 'stdout_sha256':sha(completed.stdout),'stderr_sha256':sha(completed.stderr),
 'interpretation':'Intentional revise: unresolved substantive C1 remains; checker verifies metadata, not truth.'}
(OUT/'checker-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(completed.stdout.decode(),end='')
print(json.dumps(receipt,ensure_ascii=False))
