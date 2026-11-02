"""Recheck fix11 against the exact existing 12.17 CPU code and exercise probe."""

import contextlib
import hashlib
import io
import json
import re
import runpy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
page = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "12.17")
snapshot = ROOT / page["snapshot"]
text = snapshot.read_text()
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(snapshot) == page["source_sha256"]
inline = re.findall(r"```python\n(.*?)\n```", text, re.S)
assert len(inline) == 1
assert inline[0] + "\n" == (HERE / "inline.py").read_text()

# The immutable original probe executes inline.py, which has just been shown
# byte-for-byte equal to fix11's embedded Python block. No helper replaces it.
with contextlib.redirect_stdout(io.StringIO()):
    probe = runpy.run_path(str(HERE / "probe.py"))
fresh = probe["results"]
notebook = ROOT / page["notebook"]
assert digest(notebook) == page["notebook_sha256"]
nb = json.loads(notebook.read_text())
assert "".join(nb["cells"][5]["source"]).rstrip("\n") == inline[0]
output = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-17-v1/current-probe-v1.py",
    "environment": fresh["environment"],
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "snapshot": {"path": page["snapshot"], "sha256": digest(snapshot)},
    "inline": {"path": str((HERE / "inline.py").relative_to(ROOT)),
               "sha256": digest(HERE / "inline.py"), "same_as_current_code": True},
    "probe": {"path": str((HERE / "probe.py").relative_to(ROOT)),
              "sha256": digest(HERE / "probe.py"), "actually_rerun": True},
    "notebook": {"path": page["notebook"], "sha256": digest(notebook),
                 "teaching_cell_matches_current_code": True},
    "revised_source_quote": "本節要讓每條初始聲音特徵只讀自己的 200 ms 區塊，方便對回它觀察的那段波形。",
    "inline_stdout": fresh["inline_stdout"],
    "inputs": fresh["inputs"],
    "base": fresh["base"],
    "exercise": fresh["exercise"],
    "locality": fresh["locality"],
    "input_rejections": fresh["input_rejections"],
    "missing_or_ambiguous": fresh["missing_or_ambiguous"],
}
assert output["revised_source_quote"] in text
print(json.dumps(output, ensure_ascii=False, indent=2))
