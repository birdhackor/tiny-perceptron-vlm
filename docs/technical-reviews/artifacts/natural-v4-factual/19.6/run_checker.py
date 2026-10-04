"""Invoke the unmodified official per-lesson checker and retain its real result."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[5]
ART=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/19.6'
cmd=['.venv/bin/python','scripts/check_technical_reviews.py','--lesson','19.6']
p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
(ART/'checker.stdout.txt').write_text(p.stdout);(ART/'checker.stderr.txt').write_text(p.stderr)
report=ROOT/'docs/technical-reviews/19.6.json';d=json.loads(report.read_text())
raw=(ROOT/'course/chapters/19.md').read_bytes().decode();start=re.search(r'^## 19\.6 .+$',raw,re.M).start();end=re.search(r'^## 19\.7 .+$',raw,re.M).start()
current_sha=hashlib.sha256(raw[start:end].encode()).hexdigest()
receipt={'command':cmd,'returncode':p.returncode,'reviewer_task':d['reviewer_task'],'verdict':d['verdict'],'source_sha256':current_sha,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'figure_sha256':d['figure_sha256'],'direct_figure_sha256':d['direct_figure_sha256'],'remaining_issues':d['issues'],'checker_sha256':hashlib.sha256((ROOT/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()}
(ART/'checker-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(p.stdout,end='');print(p.stderr,end='');print(json.dumps(receipt,indent=2))
raise SystemExit(p.returncode)
