"""Verify original public small files; README remains opaque, no model download."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "docs/natural-assistant/v4/public-release.json").read_bytes())
records = []
for item in manifest["files"]:
    url = "https://huggingface.co/" + manifest["repo"] + "/resolve/" + manifest["revision"] + "/" + item["path"]
    with urllib.request.urlopen(url, timeout=25) as response:
        raw = response.read(item["bytes"] + 1)
    digest = hashlib.sha256(raw).hexdigest()
    assert (digest,len(raw)) == (item["sha256"],item["bytes"])
    if item["output"] == "release-provenance.json":
        target = HERE / "official/public-release-provenance.json"
        if target.is_file():
            assert target.read_bytes() == raw
        else:
            target.write_bytes(raw)
        provenance = json.loads(raw)
        assert provenance["source"] is None and provenance["selected_variant"] == "base"
        for key in ["base_model", "asr_model", "manifest_sha256", "git_revision", "approval_sha256"]:
            assert provenance[key] == manifest[key]
    records.append({"url":url,"output":item["output"],"bytes":len(raw),"sha256":digest,
                    "expected_match":True,"read_scope":"JSON pins" if item["output"].endswith(".json") else "opaque byte hash/size only; never decoded"})
    print("PUBLIC_FILE_VERIFIED",item["output"],len(raw),digest)
(HERE / "public-metadata-verification.json").write_text(json.dumps(records,indent=2)+"\n")
print("PUBLIC_METADATA_ASSERTIONS_PASSED")
