"""Run the two frozen W.2 blocks as actual notebook code cells on a CPU kernel."""

from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import re
import sys

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[4]
snapshot = root / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3/sections/W.2.md"
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "e2fd0349c4fe6400594861ef81bab3533323cb0f34dd63855bc7ebe33fae00bd"
fence = chr(96) * 3
blocks = re.findall(fence + r"python\n(.*?)" + fence, raw.decode(), re.S)
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code) for code in blocks])
NotebookClient(nb, timeout=20, kernel_name="python3").execute(cwd=str(root))
expected = ["貓 2\n貓在睡覺\n看到 貓\n看到 狗\n", "16\n找不到 '狗'\n-1\n53 3 5\n"]
actual = ["".join(out.get("text", "") for out in cell.outputs) for cell in nb.cells]
assert actual == expected
assert all(out.output_type != "error" for cell in nb.cells for out in cell.outputs)
print(json.dumps({"created_at": datetime.now(UTC).isoformat(), "executor_python": sys.version,
                  "jupyterlab": version("jupyterlab"), "nbclient": version("nbclient"),
                  "kernel_language_info": dict(nb.metadata.language_info), "device": "cpu",
                  "snapshot_sha256": hashlib.sha256(raw).hexdigest(), "observed_stdout": actual,
                  "expected_stdout": expected, "all_passed": True,
                  "scope": "Actual kernel code-cell execution; no browser keystroke or Colab service test."},
                 ensure_ascii=False, indent=2))
