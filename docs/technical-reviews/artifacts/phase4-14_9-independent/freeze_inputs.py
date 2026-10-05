"""Preserve only the independent review's actual manuscript inputs; no report reads."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
raw = (ROOT / "course/chapters/14.md").read_bytes()
expected = json.loads((OUT / "extraction.json").read_bytes())
actual = hashlib.sha256(raw).hexdigest()
assert actual == expected["source_file_sha256"], "Chapter changed between extraction and frozen copy"
(OUT / "chapter14-frozen-input.md").write_bytes(raw)
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
dependencies = {}
for section in ("14.1", "14.8"):
    i = next(i for i, h in enumerate(headers) if h[0].startswith(f"## {section} ".encode()))
    end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
    body = raw[headers[i].start():end]
    name = f"dependency-{section}.md"
    (OUT / name).write_bytes(body)
    dependencies[section] = {
        "snapshot": name,
        "sha256": hashlib.sha256(body).hexdigest(),
        "source_first_line": raw[:headers[i].start()].count(b"\n") + 1,
    }
metadata = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_14_9",
    "frozen_input": {
        "source": "course/chapters/14.md",
        "snapshot": "chapter14-frozen-input.md",
        "sha256": actual,
        "meaning": "Full original raw UTF-8 chapter at this review's initial extraction; archived for version tracking, not a claim of reviewing the full chapter or introduction.",
    },
    "actual_read_scope": ["course/chapters/14.md#14.9", "course/chapters/14.md#14.1", "course/chapters/14.md#14.8", "course/figures/rewrite-14-9-nearby-angles.svg"],
    "dependencies": dependencies,
    "introduction_reviewed": False,
    "raw_byte_policy": "No newline normalization, stripping, or re-encoding before hashing.",
}
(OUT / "input-provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
