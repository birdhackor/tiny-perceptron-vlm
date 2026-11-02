"""Portable callback binding replay: repository files only, no ignored outputs."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = json.loads((HERE / "callback-v2-check.json").read_bytes())
freeze = record["source_freeze"]
assert sha(ROOT / freeze["path"]) == freeze["sha256"]
page = record["page"]
body = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(body).hexdigest() == page["source_sha256"]
assert record["corrected_line"] in body.decode().splitlines()
codes = re.findall(r"```python\n(.*?)\n```", body.decode(), re.S)
assert [hashlib.sha256(code.encode()).hexdigest() for code in codes] == record["code_block_sha256"]
for kind in ["all_registered_current_implementation_sources_unchanged", "all_original_portable_evidence_artifacts_unchanged"]:
    assert all(record[kind].values())
original = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.22/technical-original-v1.json").read_bytes())
for item in original["sources"]:
    if item["kind"] == "repository_code":
        assert sha(ROOT / item["path"]) == item["sha256"]
for item in original["artifacts"]:
    assert sha(ROOT / item["path"]) == item["sha256"]
for path, digest in page["figures_sha256"].items():
    assert sha(ROOT / path) == digest
assert sha(ROOT / page["notebook"]) == page["notebook_sha256"]
probe = record["corrected_three_entry_behavior_probe"]
assert probe["generated_sound"]["decoder_calls"] == 1
assert not probe["generated_sound"]["legal"] and not probe["generated_sound"]["program_success"]
assert probe["video"]["decoder_calls"] == 1 and not probe["video"]["eos"] and probe["video"]["capped"]
assert probe["sound_chunks"]["decoder_calls"] == probe["sound_chunks"]["chunk_calls"] == 0
assert record["small_projection_replay"]["passed"]
print(json.dumps({"passed": True, "mode": "portable saved evidence and current source binding only", "callback_sha256": sha(HERE / "callback-v2-check.json"), "source_sha256": page["source_sha256"], "code_blocks_unchanged": len(codes), "ignored_outputs_or_external_raw_files_read": False, "scope": "Checks current source/artifact hashes and preserved independent probe results. Does not recreate outside-Git screenshots, fetch author execution receipts, or repeat formal neural inference."}, ensure_ascii=False, indent=2))
