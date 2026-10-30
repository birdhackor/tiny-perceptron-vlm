"""Fresh-process tiny train/dev continuation provenance counterexample."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "original-source"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "original"
probe = BASE / "probes" / (LABEL + "-resume-provenance")
probe.mkdir(parents=True, exist_ok=False)
for name in ("source-A", "source-B"):
    target = probe / name
    for path in [SOURCE / "tiny_perceptron/__init__.py", *(SOURCE / "tiny_perceptron/omni_products").glob("*.py")]:
        destination = target / path.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
av_b = probe / "source-B/tiny_perceptron/omni_products/av.py"
old = 'return self.head(torch.cat((tokens[:, 0], batch["presence"]), dim=-1))'
new = old + ' * 1.01  # review-only behavior change in isolated copy'
text = av_b.read_text()
assert old in text
av_b.write_text(text.replace(old, new))

child = probe / "child.py"
child.write_text('''import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
source,output,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(source))
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.inference import load_trained
config={"seed":13,"updates":4,"training_seconds":20,"batch_size":2,"learning_rate":0.001,"validate_every":1,"representation":"additive","variants":1,"width":16,"review_scope":"tiny train/dev continuation, not formal"}
value={"mode":mode,"source":str(source),"loaded_source_binding":trainer.LOADED_SOURCE_BINDING}
try:
 if mode=="partial":
  original=trainer.run_updates
  def partial(model,make_batch,*args,**kwargs):
   counter=[0]
   def batch(generator):
    counter[0]+=1
    return make_batch(generator)
   with patch.object(trainer,"time",SimpleNamespace(monotonic=lambda:100.0 if counter[0]>=2 else 0.0)):
    return original(model,batch,*args,**kwargs)
  with patch.object(trainer,"run_updates",side_effect=partial):
   result=trainer.train_av(config,output,"cpu")
 else:
  resume=(Path(sys.argv[4]),sys.argv[5]) if mode=="resumed" else None
  result=trainer.train_av(config,output,"cpu",resume=resume)
 value.update(accepted=True,complete=result["complete"],updates=result["updates"],source_unchanged=result["source_unchanged_during_training"],implementation_sha256=result["implementation_sha256"],latest_parameter_sha256=result["latest_parameter_sha256"])
 if mode!="partial":
  value["load_trained_accepted"]=load_trained(output,"av") is not None
except Exception as error:
 value.update(accepted=False,error_type=type(error).__name__,reason=str(error))
(output.parent/(mode+"-result.json")).write_text(json.dumps(value,indent=2)+"\\n")
print("REVIEW_RESULT="+json.dumps(value))
''')

def run(mode, source, extra=()):
    output = probe / ("partial-output" if mode == "resumed" else mode + "-output")
    command = [sys.executable, str(child), str(source), str(output), mode, *map(str, extra)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=45)
    (probe / (mode + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError({"mode": mode, "returncode": result.returncode, "stderr": result.stderr})
    return json.loads((probe / (mode + "-result.json")).read_bytes())

partial = run("partial", probe / "source-A")
assert partial["accepted"] and partial["updates"] == 2 and not partial["complete"]
checkpoint = probe / "partial-output/latest-continuation.pt"
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
resumed = run("resumed", probe / "source-B", (checkpoint, checkpoint_sha))
direct = run("direct", probe / "source-B")
value = {"scope": "Isolated review copies and three genuine fresh processes; each run uses only tiny AV train/dev banks. No formal output-root/final model predictions.",
         "partial": partial, "resumed": resumed, "direct_new_revision": direct,
         "old_checkpoint_sha256": checkpoint_sha,
         "old_and_new_av_source_differ": partial["loaded_source_binding"]["av.py"] != direct["loaded_source_binding"]["av.py"],
         "old_checkpoint_accepted_under_new_revision": resumed.get("accepted",False),
         "mixed_history_claimed_source_stable": resumed.get("source_unchanged"),
         "mixed_history_is_not_new_revision_fresh_training": resumed.get("latest_parameter_sha256") != direct.get("latest_parameter_sha256"),
         "formal_success_claimed": False}
(BASE / (LABEL + "-resume-provenance-result.json")).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(value,ensure_ascii=False,indent=2))
