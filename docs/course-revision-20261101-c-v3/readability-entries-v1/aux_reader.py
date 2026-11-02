"""三個非 canonical 操作頁的逐段首讀；不計入 native 370 頁覆蓋。"""

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "auxiliary"
spec = importlib.util.spec_from_file_location("reader_api", ROOT / "docs/course-revision-20261016/tools/reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

operation, page_id, *rest = sys.argv[1:]
folder = OUT / page_id
state_path = folder / "state.json"
if operation == "start":
    source = rest[0]
    raw = (ROOT / source).read_bytes()
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "source.md"
    with snapshot.open("xb") as stream:
        stream.write(raw)
    state = {"page_id": page_id, "source": source, "source_sha256": digest(raw), "snapshot": snapshot.relative_to(ROOT).as_posix(), "next_index": 0, "started_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/c_v3_reader_entries", "canonical": False, "native_370_coverage": False}
    write_new(folder / "opening.json", state)
    write_new(state_path, state)
else:
    state = json.loads(state_path.read_bytes())
    raw = (ROOT / state["snapshot"]).read_bytes()
chunks = reader.units(raw.decode("utf-8"))
if operation == "next":
    note = json.loads(rest[0])
    i = state["next_index"]
    assert note["unit_index"] == i
    note.update(page_id=page_id, reviewer_task=state["reviewer_task"], source_sha256=state["source_sha256"], unit_sha256=digest(chunks[i].encode()), saved_at=datetime.now(UTC).isoformat(), canonical=False, native_370_coverage=False)
    write_new(folder / f"note-{i:03}.json", note)
    state["next_index"] += 1
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
i = state["next_index"]
print(json.dumps({"page_id":page_id,"canonical":False,"unit_index":i,"units_total":len(chunks),"source_sha256":state["source_sha256"],"complete":i==len(chunks)}, ensure_ascii=False))
if i < len(chunks):
    print(chunks[i])
