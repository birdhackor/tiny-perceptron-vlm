from pathlib import Path
import hashlib
import json
import shlex
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.selftrained import modal_runner, train_local_stage

probe=json.loads((BASE/'cpu-probe.json').read_text())
command=next(r['command'] for r in probe['commands'] if r['name']=='weighted44')
args=shlex.split(command)
weighted=Path(args[args.index('--output-dir')+1])
manifest_sha=args[args.index('--manifest-sha256')+1]
results=[]

def rejects(name,fn):
    try:fn()
    except ValueError as e:
        results.append({'case':name,'rejected':True,'exception':str(e)})
    else:raise AssertionError(name)

rejects('native source changes architecture to dense',lambda:modal_runner.native_joint_source_gate(weighted/'best.pt','dense'))
checkpoint=weighted/'best.pt';original=checkpoint.read_bytes()
try:
    checkpoint.write_bytes(original+b'tampered')
    rejects('checkpoint full bytes/SHA changes',lambda:train_local_stage.local_source(checkpoint,manifest_sha,False))
finally:checkpoint.write_bytes(original)
assert hashlib.sha256(checkpoint.read_bytes()).digest()==hashlib.sha256(original).digest()
print(json.dumps(results,ensure_ascii=False,indent=2))
(BASE/'gate-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
