"""T.7 dependency-only byte and provenance checks; no training or model scoring."""
import difflib
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUT = BASE / "current-context-reinspection"
OUT.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(path):
    return path.relative_to(ROOT).as_posix()

initial_report = BASE / "initial-pass-history/T.7.initial-pass.json"
assert sha(initial_report) == "c498f719b46de9018fe7e8b8e62174617672f687ed300e5bdba6ec38988e6318"
current_report = json.loads((ROOT / "docs/technical-reviews/T.7.json").read_text())
assert current_report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_t_7"
assert [x["id"] for x in current_report["claims"]] == [f"c{i}" for i in range(1, 11)]

snapshots = {}
changes = []
for section in ("T.5", "T.6", "T.7"):
    current, _, first = facts.original_section(ROOT / "course/training.md", section)
    initial, _, initial_first = facts.original_section(BASE / "training.frozen-input.md", section)
    saved = OUT / (section + ".current.md")
    saved.write_bytes(current)
    snapshots[section] = {"source": "course/training.md#" + section,
        "current_section_sha256": facts.digest(current), "current_original_first_line": first,
        "current_original_last_line": first + current.count(b"\n") - 1,
        "snapshot": relative(saved), "snapshot_sha256": sha(saved),
        "initial_section_sha256": facts.digest(initial), "initial_original_first_line": initial_first,
        "bytes_unchanged": current == initial,
        "newline_policy": "original UTF-8 bytes; no stripping or normalization"}
    changes.extend(difflib.unified_diff(initial.decode().splitlines(keepends=True),
        current.decode().splitlines(keepends=True), fromfile=section+".initial-frozen",
        tofile=section+".current", n=2))
    if section == "T.7":
        assert current == initial == (BASE / "T.7.original.md").read_bytes()
        assert facts.digest(current) == "ae6d5715da4efcda3138541eb0405408adc2c8eeb2465d49692826c44451e99e"
        old_fences = facts.fences(initial, initial_first)
        new_fences = facts.fences(current, first)
        assert len(old_fences) == len(new_fences) == 4
        assert [x["raw"] for x in old_fences] == [x["raw"] for x in new_fences]
    else:
        # Only manuscript prose changed: these predecessor fence contracts remain identical.
        assert [x["raw"] for x in facts.fences(initial, initial_first)] == [x["raw"] for x in facts.fences(current, first)]

(OUT / "context-only.diff").write_text("".join(changes))
checked_sources = []
for source in current_report["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]
        checked_sources.append({"path":source["path"],"sha256":source["sha256"]})
for artifact in current_report["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"]

proof = {"reviewer_task":current_report["reviewer_task"], "checked_at_utc":datetime.now(UTC).isoformat(),
    "command":".venv/bin/python docs/technical-reviews/artifacts/phase4-t_7-independent-20261005/context-reinspection.py",
    "environment":{"python":platform.python_version(),"executable":sys.executable,
        "cwd":str(ROOT),"device_scope":"text/bytes only; no torch or training execution"},
    "initial_pass_report":{"path":relative(initial_report),"sha256":sha(initial_report)},
    "original_whole_frozen_input":{"path":relative(BASE / "training.frozen-input.md"),
        "sha256":sha(BASE / "training.frozen-input.md"),
        "meaning":"Actual initial frozen whole-file bytes; no claim about current whole Markdown."},
    "personally_reread_current_scope":"course/training.md original lines207–353: T.5,T.6,T.7; no T.8 expansion",
    "sections":snapshots,"repository_contract_hashes_still_match_initial_pass":checked_sources,
    "initial_report_artifact_hashes_still_match":len(current_report["artifacts"]),
    "t7_body_fences_unchanged":True,"t5_t6_fences_unchanged":True,
    "scope":"Narrow dependency reinspection; not independent retraining or new model score."}
(OUT / "context-byte-proof.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(proof,ensure_ascii=False,indent=2))
