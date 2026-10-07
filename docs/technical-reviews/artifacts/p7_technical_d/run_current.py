import sys,json,subprocess,re
from record import execute,write,SESSION
r=subprocess.run(['.venv/bin/python','docs/course-revision-20261007-phase7/authoring/current-only.py',SESSION],text=True,capture_output=True,check=True)
x=json.loads(r.stdout)
blocks=re.findall(r'```python\n(.*?)```',x['text'],re.S)
if len(blocks)!=1: raise RuntimeError('Current unit must have exactly one Python fence')
a=execute(sys.argv[1],blocks[0])
write('docs/technical-reviews/artifacts/p7_technical_d/'+sys.argv[1]+'-artifact-ref.json',a)
