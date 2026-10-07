import subprocess
command=['.venv/bin/python','docs/course-revision-20261007-phase7/authoring/group-preflight.py','--manifest','docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json','--report','docs/technical-reviews/phase7-freeze06-e-p7_technical_e-recheck.json']
print('command:', ' '.join(command),flush=True)
r=subprocess.run(command,text=True,capture_output=True)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
