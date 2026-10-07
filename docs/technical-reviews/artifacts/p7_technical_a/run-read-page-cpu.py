"""Execute Python fences only after this reviewer has fully read the supplied page."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parents[4]
page_id = sys.argv[1]
manifest = json.loads((root / "docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json").read_text())
meta = next(p for p in manifest["pages"] if p["page_id"] == page_id)
source = root / meta["snapshot"]
body = source.read_text()
folder = Path(__file__).parent / "cpu" / page_id
folder.mkdir(parents=True, exist_ok=True)
namespace = {"__name__": "__main__"}
blocks = []
for index, match in enumerate(re.finditer(r"```python\n(.*?)\n```", body, re.S)):
    code = match[1] + "\n"
    code_path = folder / f"block-{index:02d}.py"
    if code_path.exists():
        raise SystemExit("Refusing to overwrite code evidence")
    code_path.write_text(code)
    out, err = io.StringIO(), io.StringIO()
    failure = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, str(code_path), "exec"), namespace)
        except Exception as error:
            failure = type(error).__name__
            traceback.print_exc()
    item = {"index": index, "code_path": str(code_path.relative_to(root)), "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": out.getvalue(), "stderr": err.getvalue(), "exception": failure}
    blocks.append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
record = {"command": f".venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/run-read-page-cpu.py {page_id}", "cwd": str(root), "page_id": page_id, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "environment": {"python": sys.version.split()[0], "torch": importlib.metadata.version("torch"), "device": platform.machine() + " CPU"}, "blocks": blocks, "scope": "fully read frozen page Python fences in document order with shared namespace; no training beyond the supplied short CPU examples"}
dest = folder / "execution.json"
if dest.exists():
    raise SystemExit("Refusing to overwrite execution evidence")
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print("execution_sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
