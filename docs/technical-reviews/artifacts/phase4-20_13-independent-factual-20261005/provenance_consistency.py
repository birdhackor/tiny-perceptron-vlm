"""Named primary provenance leaves only; no recursive source metadata output."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
def load(name):
    return json.loads((ROOT / name).read_bytes())
manifest = load("docs/natural-assistant/v4/manifest.json")
transcripts = load("docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/transcripts.json")
sources = {name: load("docs/natural-assistant/v4/data/"+name) for name in ["voice-sources.json","voice-question-sources.json"]}
raw_rows = {}
for name,source in sources.items():
    for row in source["audio_rows"]:
        if row["split"] == "test":
            raw_rows[row["id"]] = (name,row)
assert len(raw_rows) == 22
used=[]
for i,r in enumerate(transcripts):
    name,source = raw_rows[r["id"]]
    assert source["synthetic"] is False
    assert source["sha256"] == r["audio_sha256"]
    assert source["user"] == r["reference_transcript"]
    used.append({"transcript_pointer":"/"+str(i),"source_file":name,"id":r["id"],
                 "source_sha256":source["sha256"],"synthetic":False})
assert Counter(name for name,row in raw_rows.values()) == {"voice-sources.json":18,"voice-question-sources.json":4}
for i,item in enumerate(manifest["sources"]):
    # Read only exact original source path and fingerprint, not metadata values.
    path=item["path"]
    print("MANIFEST_SOURCE_LOCATOR", "/sources/"+str(i)+"/path",path,"metadata_type",type(item.get("metadata")).__name__)
    if path.startswith("docs/natural-assistant/v4/data/"):
        p=ROOT/path
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
print("TEST_RECORDING_SOURCE_COUNTS",json.dumps(dict(Counter(name for name,row in raw_rows.values()))))
print("ALL_TEST_RECORDINGS_GENUINE_PRIMARY_SOURCE_BINDINGS_MATCH")
(HERE/"recording-source-leaf-bindings.json").write_text(json.dumps(used,indent=2)+"\n")
