"""本角色逐段手寫觀察的保存器；不替讀者產生理解或判定。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

session, payload = sys.argv[1:]
note = json.loads(payload)
state = json.loads((ROOT / "docs/course-revision-20261016/reader-sessions" / session / "state.json").read_bytes())
assert note["unit_index"] == state["next_index"]
path = OUT / "checkpoints" / state["page_id"] / f"unit-{state['next_index']:03}.json"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as stream:
    json.dump(note, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
subprocess.run([sys.executable, str(ROOT / "docs/course-revision-20261016/tools/reader.py"), "next", session, "--checkpoint", str(path)], cwd=ROOT, check=True)
