"""Additional original/current slice and figure identity checks; no visual claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def bind(ref):
    assert sha(ref["path"]) == ref["sha256"]
    return load(ref["path"])


def section(body, pid):
    start = re.search(r"^## " + re.escape(pid) + r" .+$", body, re.M)
    end = re.search(r"^## ", body[start.end():], re.M)
    return body[start.start():start.end() + end.start()] if end else body[start.start():]


bundle = load("docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v5.json")
canonical = bind(bundle["canonical_source_freeze"])
parents = {p["page_id"]: p for p in canonical["pages"]}
numbered = {p["page_id"]: p for p in bind(bundle["section_freeze"])["pages"]}
figure_refs = set()
for row in bundle["sections"]:
    pid = row["lesson_id"]
    current = bind(row["report"])
    current_page = numbered[pid]
    current_parent = parents[current_page["canonical_page_id"]]
    assert section((ROOT / current_parent["snapshot"]).read_text(), pid) == (ROOT / current_page["snapshot"]).read_text()
    original = bind(row["original_report"])
    old_numbered = bind(row["original_section_freeze"])
    old_page = next(p for p in old_numbered["pages"] if p["page_id"] == pid)
    old_canonical = bind(old_numbered["canonical_source_freeze"])
    old_parent = next(p for p in old_canonical["pages"] if p["page_id"] == old_page["canonical_page_id"])
    assert sha(old_parent["snapshot"]) == old_parent["source_sha256"]
    assert section((ROOT / old_parent["snapshot"]).read_text(), pid) == (ROOT / old_page["snapshot"]).read_text()
    assert original["figure_sha256"] == old_page["figures_sha256"]
    assert current["figure_sha256"] == current_page["figures_sha256"]
    for page, freeze in [(old_page, old_canonical), (current_page, canonical)]:
        for path, digest in page["figures_sha256"].items():
            reference = freeze["figure_snapshots"][path]
            assert reference["sha256"] == digest == sha(reference["path"])
            figure_refs.add((reference["path"], digest))

originals = load("docs/technical-reviews/artifacts/omni-technical-course-v1/original-source-inspection-v1.json")["sources"]
for source in originals:
    assert sha(source["local_original_path"]) == source["sha256"]

old_primary_refs = []


def inspect_refs(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            assert sha(value["path"]) == value["sha256"]
            old_primary_refs.append(value)
        for child in value.values():
            inspect_refs(child)
    elif isinstance(value, list):
        for child in value:
            inspect_refs(child)


account = load("docs/technical-reviews/artifacts/omni-technical-course-assessment-v1/colab-observation-output-v1.json")
upload = load("docs/technical-reviews/artifacts/omni-technical-course-v1/current-browser-output-v1.json")
inspect_refs(account)
inspect_refs(upload)
assert account["observed_primary_text"] == "You must be logged in with a Google Account to continue."
assert account["observed_primary_text"] in (ROOT / account["DOM_text"]["path"]).read_text()
assert upload["official_upload_observation"]["accessible_name"] == "Upload notebook"
assert upload["official_upload_observation"]["visible"] is True
assert not account["cell_executed"] and not account["successful_runtime_connection_verified"]
assert not upload["file_selected_or_uploaded"] and not upload["account_login_performed"] and not upload["cell_executed"]

print(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_course",
    "original_and_current_numbered_slices": 52,
    "original_and_current_section_figure_maps_checked": 52,
    "unique_bound_figure_snapshots_checked": len(figure_refs),
    "unchanged_cached_original_sources_checked": len(originals),
    "unchanged_own_Colab_primary_DOM_and_PNG_refs_checked": len(old_primary_refs),
    "scope": "Source/figure/primary identity checks only. No new image viewing, official Colab navigation, upload, login, cell execution, cloud success, section subject approval or stage seal. Original primary observations are reused at their actual prior dates."
}, ensure_ascii=False, indent=2))
