"""Bounded source12 dependency reuse check; standard library only, no kernels/NN."""

import difflib
import hashlib
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FULL = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-12.13/technical-original-v1.json"
HISTORICAL = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-12.13/historical-group-2-appendix-v1.json"
REPAIRS = "docs/course-revision-20261029-omni/reviews/P5-actual/coherence-three-author-repairs-source12-v1.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v12.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def section(raw, lesson):
    start = re.search(rb"^## " + lesson.encode() + rb" ", raw, re.M).start()
    next_heading = re.search(rb"^## ", raw[start + 1 :], re.M)
    return raw[start : start + 1 + next_heading.start()] if next_heading else raw[start:]


def main():
    assert sha(read(FULL)) == "e04fd765b87c31445515034f5ba5b30d09de76cdae160d686755da637e0edae8"
    assert sha(read(HISTORICAL)) == "f2700450cbbf96113e10c2ada51d6d0f40c5cfd3adf9eafcc32c4505854374ee"
    assert sha(read(REPAIRS)) == "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"
    assert sha(read(FREEZE)) == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
    original = json.loads(read(FULL))
    repairs = json.loads(read(REPAIRS))
    frozen = json.loads(read(FREEZE))
    page = next(p for p in frozen["pages"] if p["page_id"] == "12.13")
    assert sha(read(page["snapshot"])) == page["source_sha256"] == original["source_sha256"]
    assert section(read(page["source"]), "12.13") == read(page["snapshot"])
    assert page["figures_sha256"] == original["figure_sha256"]
    for figure, expected in page["figures_sha256"].items():
        assert sha(read(figure)) == expected
    assert sha(read(page["notebook"])) == page["notebook_sha256"] == original["notebook"]["sha256"]
    sources = {s["id"]: s for s in original["sources"]}
    changes = []
    for source_id, changed_lesson in [("s9", "12.17"), ("s13", "19.23")]:
        source = sources[source_id]
        change = next(c for c in repairs["changes"] if c["page_id"] == changed_lesson)
        before, after = read(change["before"]["path"]), read(change["after"]["path"])
        assert sha(before) == change["before"]["sha256"]
        assert sha(after) == change["after"]["sha256"]
        chapter = read(source["path"])
        assert section(chapter, changed_lesson) == after and chapter.count(after) == 1
        reconstructed_before = chapter.replace(after, before, 1)
        assert sha(reconstructed_before) == source["sha256"]
        actual_diff = "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True), fromfile="source11/" + changed_lesson, tofile="source12/" + changed_lesson, n=2))
        assert actual_diff == change["diff"]
        before_code = re.findall(rb"```[^\n]*\n(.*?)```", before, re.S)
        after_code = re.findall(rb"```[^\n]*\n(.*?)```", after, re.S)
        assert before_code == after_code
        needed = []
        if source_id == "s9":
            assert section(chapter, "12.13") == section(reconstructed_before, "12.13")
            for literal in [
                b"raw = waveform[begin : begin + chunk_samples]",
                b"center=False",
                b"support_start = start_ms + begin / rate * 1000",
                b"support_end = start_ms + (begin + len(raw)) / rate * 1000",
                b"energy = raw.square().mean()",
                b"normalized = raw / energy.sqrt().clamp(min=1e-6)",
                b'if token["content"][-1] > 0.01',
            ]:
                assert literal in before and literal in after
            needed = ["12.13 identical section for c4", "12.17 identical raw-block slice, center=False, raw support arithmetic for c13", "Added mean-square sentence agrees with unchanged energy=raw.square().mean(), sqrt normalization and 0.01 threshold"]
        else:
            for lesson in ["19.6", "19.13"]:
                assert section(chapter, lesson) == section(reconstructed_before, lesson)
            needed = ["19.6 unchanged 16 ordered interval paragraph for c12", "19.13 unchanged old-A/product identity mapping for c12", "19.23 rubric clarification is outside these supporting excerpts; all fenced code is unchanged"]
        changes.append({
            "source_id": source_id,
            "path": source["path"],
            "old_report_sha256": source["sha256"],
            "current_sha256": sha(chapter),
            "reconstructed_before_sha256": sha(reconstructed_before),
            "entire_prior_file_reconstructed_exactly": True,
            "changed_lesson": changed_lesson,
            "exact_diff": actual_diff,
            "diff_matches_author_receipt": True,
            "all_fenced_code_in_changed_lesson_identical": True,
            "claim_ids_using_this_source": [c["id"] for c in original["claims"] if any(e["source_id"] == source_id for e in c["evidence"])],
            "support_relations_personally_checked": needed,
        })
    unchanged_repository_sources = []
    for source in original["sources"]:
        if source["kind"] == "repository_code" and source["id"] not in {"s9", "s13"}:
            assert sha(read(source["path"])) == source["sha256"]
            unchanged_repository_sources.append({"source_id": source["id"], "path": source["path"], "sha256": source["sha256"]})
    for artifact in original["artifacts"]:
        assert sha(read(artifact["path"])) == artifact["sha256"]
    print(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": original["reviewer_task"],
        "context": "Same-owner bounded source12 current callback; not a fresh blind whole-page review. Original full claims are reused on an unchanged canonical unit after personally checking the two complete chapter deltas.",
        "requested_collection_phase": "continuity03 (root handles actual receipt); no receipt/seal created by this script",
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-12.13/source12_bounded_dependency_check_v1.py",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU; standard-library file/hash/diff assertions only", "neural_execution": "none", "kernel_rerun": "none"},
        "original_report": {"path": FULL, "sha256": sha(read(FULL))},
        "preserved_historical_appendix1": {"path": HISTORICAL, "sha256": sha(read(HISTORICAL))},
        "source_freeze": {"path": FREEZE, "sha256": sha(read(FREEZE))},
        "author_exact_diff_receipt": {"path": REPAIRS, "sha256": sha(read(REPAIRS))},
        "canonical_unit_unchanged": {"source_sha256": page["source_sha256"], "figure_sha256": page["figures_sha256"], "notebook_sha256": page["notebook_sha256"]},
        "reviewed_dependency_deltas": changes,
        "unchanged_other_repository_sources": unchanged_repository_sources,
        "original_artifacts_digest_verified_without_reexecution": True,
        "result": "All assertions passed. Only s9/s13 complete chapter digests change; their original claim-support relations remain valid. No new technical issue, model execution, full kernel rerun, collection, seal or appendix2 generated.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
