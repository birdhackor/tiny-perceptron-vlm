"""Prove the original experiment implementation matches its recorded revision."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
report=json.loads((OUT/"inputs/docs/course-experiments/results/real_text.json").read_text())
records=[]
for rel in ("scripts/course_experiments/text.py","scripts/course_experiments/common.py","tiny_perceptron/data.py"):
    argv=["git","show",report["revision"]+":"+rel]
    raw=subprocess.check_output(argv,cwd=ROOT)
    p=OUT/"original-version"/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    assert digest==report["code_sha256"][rel]
    records.append({"path":rel,"command_argv":argv,"sha256":digest,"match":True})
output={"revision":report["revision"],"files":records,"scope":"Original repository implementation, not a review judgment; historical CUDA run not rerun."}
(OUT/"original-version-provenance.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,indent=2))
