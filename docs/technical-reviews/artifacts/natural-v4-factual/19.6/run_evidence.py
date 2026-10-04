"""Save actual subprocess status, stdout, stderr and script hash for small CPU/render checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[5]
ART=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/19.6'
def run(item):
 stem,filename=item
 script=ART/filename;command=['.venv/bin/python',str(script.relative_to(ROOT))]
 p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=55)
 (ART/(stem+'.stdout.txt')).write_text(p.stdout);(ART/(stem+'.stderr.txt')).write_text(p.stderr)
 receipt={'command':command,'cwd':str(ROOT),'returncode':p.returncode,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()}
 (ART/(stem+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 print(stem,json.dumps(receipt));print(p.stdout);print(p.stderr)
 return p.returncode
with ThreadPoolExecutor(max_workers=2) as pool:
 codes=list(pool.map(run,[('verification','verify_section.py'),('render','render_prerequisites.py')]))
raise SystemExit(int(any(codes)))
