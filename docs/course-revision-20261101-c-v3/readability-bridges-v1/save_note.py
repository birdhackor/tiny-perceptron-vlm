"""Save this reader's current checkpoint, then ask the official reader for one unit."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
base = Path(__file__).resolve().parent
session, raw = sys.argv[1:]
note = json.loads(raw)
folder = base / "checkpoints" / note.pop("_page")
folder.mkdir(parents=True, exist_ok=True)
path = folder / f"unit-{note['unit_index']:03}.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(note, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
subprocess.run(
    [sys.executable, str(root / "docs/course-revision-20261016/tools/reader.py"),
     "next", session, "--checkpoint", str(path)],
    cwd=root,
    check=True,
)
