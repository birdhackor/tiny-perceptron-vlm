"""Check the already read fixed-table unknown-character boundary."""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

base = Path(__file__).parent
source = base / "cpu/1.1/block-00.py"
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(source.read_text(), str(source), "exec"), namespace)
try:
    encoded = [namespace["to_id"][char] for char in "貓看鳥"]
    observed = {"result": encoded}
except Exception as error:
    observed = {"exception": type(error).__name__, "message": str(error)}
record = {"command": ".venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/1-1-unknown.py", "environment": {"python": sys.version.split()[0], "device": "Linux CPU"}, "code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "input": "貓看鳥", "observed": observed, "scope": "reuse the original five-character table without rebuilding it"}
path = base / "1-1-unknown-result.json"
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False))
print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
