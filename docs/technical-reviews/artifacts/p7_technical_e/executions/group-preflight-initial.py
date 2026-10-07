import subprocess,json
cmd=['.venv/bin/python','docs/course-revision-20261007-phase7/authoring/group-preflight.py','--manifest','docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json','--report','docs/technical-reviews/phase7-freeze03-e-p7_technical_e-initial.json']
print('actual_command',json.dumps(cmd),flush=True)
r=subprocess.run(cmd,text=True,capture_output=True);print(r.stdout);print(r.stderr);print('exit_code',r.returncode);raise SystemExit(r.returncode)
