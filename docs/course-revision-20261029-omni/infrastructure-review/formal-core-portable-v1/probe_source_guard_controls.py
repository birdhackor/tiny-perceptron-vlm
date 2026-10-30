"""Fresh import/start/end source guard controls in an isolated copy."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "original-source"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "original"
probe = BASE / "probes" / (LABEL + "-source-guards")
probe.mkdir(parents=True, exist_ok=False)
target = probe / "source"
for path in [SOURCE / "tiny_perceptron/__init__.py", *(SOURCE / "tiny_perceptron/omni_products").glob("*.py")]:
    destination = target / path.relative_to(SOURCE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
child = probe / "child.py"
child.write_text('''import json,sys
from dataclasses import asdict
from pathlib import Path
source,output=Path(sys.argv[1]),Path(sys.argv[2]);sys.path.insert(0,str(source))
import torch
from tiny_perceptron.omni_products import training_runtime as t
from tiny_perceptron.omni_products.sound import SoundCodec,CodecConfig
from tiny_perceptron.omni_products.inference import load_trained
av=source/"tiny_perceptron/omni_products/av.py";original=av.read_bytes();rows=[]
config={"seed":7,"updates":1,"training_seconds":20,"batch_size":4,"learning_rate":0.001,"validate_every":1,"model":asdict(CodecConfig(hop=80,width=16,codes=32))}
for phase in ["stable_fresh","mutated_after_import","mutated_during_update"]:
 av.write_bytes(original);t.deterministic(7);model=SoundCodec(CodecConfig(hop=80,width=16,codes=32))
 if phase=="mutated_after_import":av.write_bytes(original+b"\\n# review-only mutation after import\\n")
 def batch(g):
  if phase=="mutated_during_update":av.write_bytes(original+b"\\n# review-only mutation during update\\n")
  return torch.randn(4,80,generator=g)*0.2
 try:
  r=t.run_updates(model,batch,lambda m,x:m(x),lambda _:0.0,config,output/phase)
  rows.append({"phase":phase,"accepted":True,"complete":r["complete"],"source_unchanged":r["source_unchanged_during_training"],"loader_accepted":load_trained(output/phase,"codec") is not None})
 except Exception as e:
  rows.append({"phase":phase,"accepted":False,"error_type":type(e).__name__,"reason":str(e),"result_file_exists":(output/phase/"training-result.json").exists()})
av.write_bytes(original)
print("REVIEW_RESULT="+json.dumps(rows))
''')
result = subprocess.run([sys.executable, str(child), str(target), str(probe / "output")], text=True, capture_output=True, timeout=45)
(probe / "child.log").write_text(result.stdout + result.stderr)
if result.returncode:
    raise RuntimeError(result.stderr)
rows = json.loads(next(line[len("REVIEW_RESULT="):] for line in result.stdout.splitlines() if line.startswith("REVIEW_RESULT=")))
assert [x["accepted"] for x in rows] == [True, False, False]
value = {"scope": "Three tiny same-process controls after one genuine fresh import; only isolated copies mutate. No formal output/final prediction.", "cases": rows}
(BASE / (LABEL + "-source-guard-controls-result.json")).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(value,ensure_ascii=False,indent=2))
