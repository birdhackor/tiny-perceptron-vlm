"""Run this reviewer's unlocked warmup variations and narrow API checks on CPU."""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
result = {"command": ".venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/firststeps-variants.py", "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": platform.machine() + " CPU"}}
animals = ["鳥", "貓", "狗"]
record = {"animal": "貓", "caption": "貓在睡覺"}
before = animals.copy()
record["caption"] = "貓正在喝水"
result["animals_variation"] = {"first": animals[0], "length": len(animals), "loop_outputs": ["看到 " + x for x in animals], "unchanged_by_caption": before == animals}
scores = torch.tensor([[60., 70., 110.], [80., 90., 100.]])
result["scores_variation"] = {"dim1": scores.mean(1).tolist(), "dim0": scores.mean(0).tolist()}
result["log_025"] = {"exact": -math.log(.25), "rounded": round(-math.log(.25), 3)}
recipes = torch.tensor([[2., 1., 0.], [1., 2., 1.]])
weights = torch.tensor([[10., 0.], [20., 1.], [30., 4.]])
result["bias_8"] = (recipes @ weights + torch.tensor([8., 0.])).tolist()
result["learning_rates"] = {}
for rate in [.6, 1.1]:
    w = torch.tensor(1., requires_grad=True)
    (w - 3).square().backward()
    with torch.no_grad():
        w -= rate * w.grad
    result["learning_rates"][str(rate)] = {"w": w.item(), "loss": (w - 3).square().item()}
logits = torch.tensor([[2., 1.]])
target = torch.tensor([0])
prob = logits.softmax(-1)
result["ce_contract"] = {"probability_sum": prob.sum().item(), "raw_logits_ce": F.cross_entropy(logits, target).item(), "negative_log_answer_prob": -prob[0, 0].log().item(), "probabilities_as_logits_ce": F.cross_entropy(prob, target).item()}
cwd = Path.cwd()
result["root_exists"] = [Path("pyproject.toml").exists(), Path("tiny_perceptron").exists()]
os.chdir(cwd / "notebooks")
result["notebooks_exists"] = [Path("pyproject.toml").exists(), Path("tiny_perceptron").exists()]
os.chdir(cwd)
result["exception_types"] = {}
for name, code in {"list_index": "['貓', '狗'][2]", "dictionary_key": "{'animal':'貓'}['absent']", "missing_file": "open('/tmp/p7-technical-a-intentionally-absent-input')", "syntax": "if True", "assertion": "assert False"}.items():
    try:
        exec(code)
    except Exception as error:
        result["exception_types"][name] = type(error).__name__
dry_path = Path(tempfile.mkdtemp(prefix="p7-technical-a-dryrun-", dir="/tmp")) / "dry.pt"
command = [str(cwd / ".venv/bin/python"), "scripts/train_simple.py", "--device", "cpu", "--steps", "1", "--output", str(dry_path)]
dry = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
result["dryrun"] = {"command": command, "returncode": dry.returncode, "stdout": dry.stdout, "stderr": dry.stderr, "checkpoint_created": dry_path.exists(), "source_sha256": hashlib.sha256((cwd / "scripts/train_simple.py").read_bytes()).hexdigest()}
path = Path(__file__).parent / "firststeps-variants-result.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
