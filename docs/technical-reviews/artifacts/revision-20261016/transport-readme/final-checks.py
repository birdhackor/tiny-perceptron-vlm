"""Small CPU checks for the README's explicit training switch and asset verifier."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.training import seed_everything

out = ROOT / "work/transport-readme/train-switch.pt"
assert not out.exists()
args = [sys.executable, "scripts/train.py", "--task", "text", "--data", "work/transport-readme/generated/toy-text/train.jsonl", "--train", "--steps", "1", "--output", str(out)]
p = subprocess.run(args, capture_output=True, text=True, timeout=60)
r = {"at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/tech_transport_readme_20261016", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}, "training_switch": {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}}
assert p.returncode == 0
checkpoint = torch.load(out, map_location="cpu", weights_only=False)
seed_everything(42)
initial = TinyLM(ModelConfig())
r["training_switch"]["checkpoint_exists"] = out.exists()
r["training_switch"]["checkpoint_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
r["training_switch"]["checkpoint_step"] = checkpoint["step"]
r["training_switch"]["changed_state_tensors"] = sum(not torch.equal(value, checkpoint["model"][name]) for name, value in initial.state_dict().items())
args = [sys.executable, "-m", "pytest", "-q", "tests/test_assets.py", "tests/test_env.py"]
p = subprocess.run(args, capture_output=True, text=True, timeout=120)
r["relevant_existing_tests"] = {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
(Path(__file__).parent / "final-checks-results.json").write_text(json.dumps(r, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(r, ensure_ascii=False))
