"""Read-only metadata/replay audit of the actual candidate; no activation or evidence generation."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MANIFEST = "docs/course-revision-20261029-omni/reviews/assembly-reader-01/manifest-reader.json"
sys.path.insert(0, str(ROOT / "scripts"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes())}


def main():
    gate = load("metadata_incremental", ROOT / "docs/review-tools/incremental_revision.py")
    helper = load("metadata_omni", ROOT / "scripts/omni_revision/review_setup.py")
    inventory_tool = load("metadata_composite", ROOT / "docs/review-tools/composite_review.py")
    replay = load("metadata_replay", ROOT / "docs/review-tools/baseline_replay.py")
    builder = load("metadata_builder", ROOT / "scripts/build_course.py")
    preflight = load("metadata_preflight", ROOT / "docs/course-review-20261019/tools/preflight.py")
    reading = load("metadata_reading", ROOT / "scripts/reading_time.py")
    export = load("metadata_export", ROOT / "scripts/export_course.py")
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    frozen = gate.document(ROOT, manifest["source_freeze"])
    inventory = inventory_tool.current_inventory(ROOT)
    live = {p["page_id"]: p for p in inventory["pages"]}
    current_frozen = {p["page_id"]: p for p in frozen["pages"]}
    fields = ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256")
    differences = [{"page_id": pid, "fields": [f for f in fields if row.get(f) != current_frozen[pid].get(f)]}
                   for pid, row in live.items() if pid not in current_frozen or any(row.get(f) != current_frozen[pid].get(f) for f in fields)]
    snapshot_errors = []
    for row in frozen["pages"]:
        try:
            gate.artifact(ROOT, {"path": row["snapshot"], "sha256": row["source_sha256"]})
        except Exception as error:
            snapshot_errors.append({"page_id": row["page_id"], "error": str(error)})
    for ref in frozen["figure_snapshots"].values():
        gate.artifact(ROOT, ref)
    builder_output = io.StringIO()
    with contextlib.redirect_stdout(builder_output):
        builder.build(check=True)
    implementation_drift = [name for name, digest in manifest["implementation"].items()
                            if sha((ROOT / name).read_bytes()) != digest]
    policy = json.loads((ROOT / "docs/course-publish-20261008/review-manifest.json").read_bytes())
    required = set(gate.document(ROOT, policy["current_technical_inputs"])["files_sha256"]) | {
        "docs/review-tools/incremental_revision.py", "docs/review-tools/baseline_replay.py", "docs/review-tools/phase7_review.py",
        "docs/review-tools/composite_review.py", "docs/review-tools/incremental_reader.py", "scripts/check_technical_reviews.py",
        "scripts/build_course.py", "scripts/reading_time.py", "scripts/export_course.py", "docs/course-revision-20261016/tools/reader.py",
        "docs/course-revision-20261016/tools/freeze_sources.py", "tests/test_incremental_revision_review.py"} | gate.HISTORICAL_INPUTS
    supplements = gate.baseline_supplements(ROOT, manifest)
    identities = replay.preserved_identities(ROOT)
    for i, item in enumerate(supplements):
        replay.require_preserved_identity(identities, "independent actual supplement " + str(i), item)
        gate.artifact(ROOT, {"path": item["path"], "sha256": item["sha256"]})
    baseline = replay.replay(ROOT, supplements)
    accepted = helper.read_ref(helper.ACCEPTED_BASELINE)
    accepted_freeze = helper.read_ref(accepted["source_freeze"])
    old_index = [{"id": p["page_id"]} for p in accepted_freeze["pages"] if p["kind"] == "lesson"]
    old_home = next(p for p in accepted_freeze["pages"] if p["page_id"] == "index")
    old_home_raw = (ROOT / old_home["snapshot"]).read_text()
    real_index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    home_tests = {"old_index_count": len(old_index), "current_index_count": len(real_index),
                  "old_index_has_19_21": any(row["id"] == "19.21" for row in old_index),
                  "old_home_matches_accepted_snapshot": export.home_introduction(old_index, True) == old_home_raw,
                  "current_home_matches_candidate_snapshot": export.home_introduction(real_index, True) == (ROOT / current_frozen["index"]["snapshot"]).read_text(),
                  "current_overview_requires_passed_index": "當答案要用來做決定" in export.home_introduction(real_index, True) and "當答案要用來做決定" not in export.home_introduction(old_index, True)}
    metadata = ROOT / "course/reading-times.json"
    proposal = ROOT / "docs/course-revision-20261029-omni/reviews/assembly-reader-01/reading-times-proposal.json"
    estimates = {}
    estimate_error = None
    try:
        estimates = reading.load_estimates(metadata, inventory, require_complete=True)
    except (ValueError, OSError) as error:
        estimate_error = str(error)
    negative_estimates = []
    if estimates:
        for kind in ("stale_source", "stale_figure", "duplicate_page", "missing_page", "false_integer", "human_label"):
            data = json.loads(metadata.read_bytes())
            if kind == "stale_source": data["pages"][0]["source_sha256"] = "0" * 64
            elif kind == "stale_figure": data["pages"][0]["figures_sha256"] = {"fake.svg": "0" * 64}
            elif kind == "duplicate_page": data["pages"].append(copy.deepcopy(data["pages"][0]))
            elif kind == "missing_page": data["pages"].pop()
            elif kind == "false_integer": data["pages"][0]["minutes_min"] = True
            elif kind == "human_label": data["estimate_kind"] = "human_average"
            with tempfile.TemporaryDirectory(prefix="omni-estimate-mutation-") as folder:
                path = Path(folder) / "estimates.json"
                path.write_text(json.dumps(data))
                try:
                    reading.load_estimates(path, inventory, require_complete=True)
                    negative_estimates.append({"case": kind, "accepted": True})
                except ValueError as error:
                    negative_estimates.append({"case": kind, "accepted": False, "error": str(error)})
    assignments = gate.document(ROOT, manifest["assignments"])["tasks"]
    role_map = {row["task"]: row["role"] for row in assignments}
    seals = gate.review_seals(ROOT, manifest["review_waves"], ("reader",))
    try:
        gate.check_tool_audit(ROOT, manifest, role_map, seals)
        old_audit = {"accepted": True}
    except (ValueError, KeyError, OSError) as error:
        old_audit = {"accepted": False, "error": str(error)}
    scope = helper.numbered_scope(accepted_freeze, frozen)
    checked_names = ["docs/review-tools/incremental_revision.py", "docs/review-tools/baseline_replay.py", "docs/review-tools/phase7_review.py",
                     "docs/course-revision-20261016/tools/reader.py", "docs/course-revision-20261016/tools/freeze_sources.py", "tests/test_incremental_revision_review.py",
                     *sorted(gate.HISTORICAL_INPUTS), manifest["baseline_supplements"]["path"],
                     "scripts/omni_revision/review_setup.py", "docs/course-revision-20261029-omni/REVIEW-PROTOCOL.md", ".github/workflows/ci.yml", ".github/workflows/pages.yml",
                     "scripts/check_technical_reviews.py", "scripts/build_course.py", "scripts/reading_time.py", "scripts/export_course.py", "docs/review-tools/composite_review.py",
                     "docs/review-tools/incremental_reader.py", "README_en.md", "course/reading-times.json", "course/lesson-index.json", "course/lesson-contract.json",
                     "docs/course-revision-20261016/baseline-overlay.json", MANIFEST, manifest["source_freeze"]["path"], manifest["assignments"]["path"],
                     "docs/course-revision-20261029-omni/reviews/assembly-reader-01/dependency-W1-packet.json", "docs/course-revision-20261029-omni/reviews/assembly-reader-01/dependency-w-numbered-freeze.json"]
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_review_tools",
              "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(ROOT)),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; read-only metadata and baseline replay"},
              "checked_files": [reference(ROOT / name) for name in dict.fromkeys(checked_names)], "candidate_manifest": reference(ROOT / MANIFEST),
              "implementation": {"count": len(manifest["implementation"]), "required_missing": sorted(required - set(manifest["implementation"])), "drift": implementation_drift, "includes_README_en": "README_en.md" in manifest["implementation"]},
              "source_notebook": {"live_count": len(live), "freeze_count": len(frozen["pages"]), "canonical_tools_id_absent": "__review_tools__" not in live, "differences": differences, "snapshot_errors": snapshot_errors, "figure_snapshot_count": len(frozen["figure_snapshots"]), "builder_output": builder_output.getvalue().strip()},
              "baseline": {"supplements": supplements, "result_status": baseline["status"], "failures": baseline["failures"], "records": len(baseline.get("records", []))},
              "home": home_tests, "estimate_metadata": {"actual": reference(metadata), "proposal": reference(proposal), "same_bytes": metadata.read_bytes() == proposal.read_bytes(), "validated_count": len(estimates), "error": estimate_error, "negative_checks": negative_estimates,
                  "ai_not_human_statement": reading.METHOD, "new_round_owners": sorted({row["reviewer_task"] for row in estimates.values() if "/root/omni_" in row["reviewer_task"]})},
              "scope": {"numbered_parent_sections": {pid: list(rows) for pid, rows in scope.items()}, "section_count": sum(map(len, scope.values())), "W1_is_dependency_only_outside_main_source_delta": "first-steps" not in scope},
              "assignments": {"own_task_present_in_reader_stage_candidate": "/root/omni_technical_coordinator/technical_review_tools" in role_map, "own_task_is_author": "/root/omni_technical_coordinator/technical_review_tools" in manifest["author_tasks"], "reader_author_overlap": sorted({row["task"] for row in assignments if row["role"] == "reader"} & set(manifest["author_tasks"])), "duplicate_tasks": len(assignments) - len(role_map)},
              "preflight_transcription_drift": preflight.drift(), "candidate_carried_old_tool_audit": old_audit,
              "pending": ["This reader-stage manifest has no new technical seal, no actual main numbered bundle, and no new tool-audit assignment/report. Those must be collected later, never presealed.", "W1 actual original/current report, separate issue index, receipt pair, and first-steps collector binding require later same-auditor callback."]}
    with (OUT / "metadata-results-v1.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in {"checked_files", "estimate_metadata", "baseline"}} | {"baseline": output["baseline"], "estimate_metadata": {key: value for key, value in output["estimate_metadata"].items() if key != "new_round_owners"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
