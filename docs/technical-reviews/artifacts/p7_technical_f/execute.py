"""Capture an actual CPU/read-only command; makes no review verdict."""
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = json.load(sys.stdin)
target = ROOT / "docs/technical-reviews/artifacts/p7_technical_f" / (spec["id"] + ".json")
if target.exists():
    raise SystemExit("Refusing to overwrite execution record")
wrapper_versions = {"python": platform.python_version(), "executable": sys.executable}
versions = {"python": platform.python_version(), "device": "CPU"}
if Path(spec['argv'][0]).name.startswith('python'):
    probe = "import importlib.metadata,json,platform,sys\nv={'python':platform.python_version(),'executable':sys.executable,'device':'CPU'}\nfor n in ('torch','huggingface-hub','safetensors','modal','uv'):\n try:v[n]=importlib.metadata.version(n)\n except importlib.metadata.PackageNotFoundError:pass\nprint(json.dumps(v))"
    observed = subprocess.run([spec['argv'][0], '-c', probe], cwd=ROOT, capture_output=True, text=True, check=True)
    versions = json.loads(observed.stdout)
inputs = {}
for name in spec.get("inputs", []):
    path = ROOT / name
    inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(spec["argv"], cwd=ROOT, capture_output=True, text=True)
record = {"command": spec["argv"], "started_at": started, "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "environment": versions, "wrapper_environment": wrapper_versions, "inputs": inputs, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), **record}, ensure_ascii=False))
sys.exit(result.returncode)
