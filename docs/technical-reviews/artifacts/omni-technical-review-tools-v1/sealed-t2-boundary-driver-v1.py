"""Diagnose retained initial T2 expectation; never overwrite the initial raw result."""

import copy
import hashlib
import importlib.util
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def main():
    helper = load("T2_followup_helper", ROOT / "scripts/omni_revision/review_setup.py")
    api = load("T2_followup_api", ROOT / "docs/review-tools/incremental_revision.py")
    validator = load("T2_followup_validator", ROOT / "scripts/check_technical_reviews.py")
    manifest = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v4/manifest-intake.json").read_bytes())
    bundle = helper.read_ref(manifest["numbered_section_reviews"])
    entry = next(row for row in bundle["sections"] if row["lesson_id"] == "T.2")
    original = helper.read_ref(entry["original_report"])
    auxiliary = helper.read_ref(entry["original_section_freeze"])
    canonical = helper.read_ref(auxiliary["canonical_source_freeze"])
    dependencies = {item["source"] for item in entry["original_source_snapshots"]}
    reconstruction = []
    for source in sorted(dependencies):
        pages = [page for page in canonical["pages"] if page["source"] == source]
        intro = next(page for page in pages if page["selector"] == "before-first-section")
        parts = [intro, *(page for page in pages if page["selector"].startswith("section:"))]
        raw = b"".join(api.artifact(ROOT, {"path": page["snapshot"], "sha256": page["source_sha256"]}) for page in parts)
        source_record = next(item for item in original["sources"] if item.get("path") == source)
        assert sha(raw) == source_record["sha256"]
        reconstruction.append({"source": source, "original_declared_sha256": source_record["sha256"], "reconstructed_sha256": sha(raw),
                               "component_refs": [{"page_id": page["page_id"], "path": page["snapshot"], "sha256": page["source_sha256"]} for page in parts]})
    cases = []
    for case in ("original_without_extra_snapshots_valid_from_canonical", "no_snapshots_and_no_canonical_dependency_pieces", "no_snapshots_and_bad_canonical_component_sha"):
        with tempfile.TemporaryDirectory(prefix="T2-original-source-boundary-") as folder:
            temp = Path(folder)
            tool = load("T2_followup_" + case, ROOT / "scripts/omni_revision/review_setup.py")
            original_local = tool.local
            overlays = {}
            tool.local = lambda name: overlays.get(str(name), original_local(name))
            current_auxiliary = copy.deepcopy(auxiliary)
            changed = copy.deepcopy(canonical)
            reference = entry["original_section_freeze"]

            def overlay(path, value):
                raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
                target = temp / str(len(overlays))
                target.write_bytes(raw)
                overlays[path] = target
                return {"path": path, "sha256": sha(raw)}

            if case == "no_snapshots_and_no_canonical_dependency_pieces":
                changed["pages"] = [page for page in changed["pages"] if page["source"] not in dependencies]
            elif case == "no_snapshots_and_bad_canonical_component_sha":
                next(page for page in changed["pages"] if page["source"] in dependencies)["source_sha256"] = "0" * 64
            if case != "original_without_extra_snapshots_valid_from_canonical":
                current_auxiliary["canonical_source_freeze"] = overlay(auxiliary["canonical_source_freeze"]["path"], changed)
                reference = overlay(entry["original_section_freeze"]["path"], current_auxiliary)
            try:
                tool.check_original(original, reference, "T.2", "training", original["reviewer_task"], validator, api, original_source_snapshots=[])
                accepted, error = True, None
            except (ValueError, KeyError, TypeError, OSError) as caught:
                accepted, error = False, str(caught)
            cases.append({"case": case, "accepted": accepted, "expected_accept": case == "original_without_extra_snapshots_valid_from_canonical",
                          "matches_expectation": accepted == (case == "original_without_extra_snapshots_valid_from_canonical"), "error": error})
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_review_tools",
              "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; actual original bytes and isolated negative overlays"},
              "retained_initial_probe": ref(OUT / "sealed-local-fix-results-v1.json"),
              "diagnosis": "Initial T2_missing_historical_snapshots was incorrectly expected to reject. Both declared old Markdown digests are independently reconstructible from the authenticated ORIGINAL canonical freeze. Extra snapshots are redundant in this case, so acceptance is correct. Initial raw result is preserved unchanged.",
              "original_report": entry["original_report"], "original_section_freeze": entry["original_section_freeze"],
              "original_canonical_freeze": auxiliary["canonical_source_freeze"], "reconstruction": reconstruction,
              "cases": cases, "summary": {"unexpected": [item for item in cases if not item["matches_expectation"]], "correct_guard_behavior": all(item["matches_expectation"] for item in cases)}}
    target = OUT / "sealed-t2-boundary-results-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(target), "summary": output["summary"], "cases": cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
