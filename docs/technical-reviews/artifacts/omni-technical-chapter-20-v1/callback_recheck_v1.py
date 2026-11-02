"""Same-reviewer source15 callback; preserve all original review evidence."""

import difflib
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = HERE / "callback-inputs-v1.json"
ORIGINAL = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_chapter_20/technical-original-v1.json"
ORIGINAL_SHA = "94b98d66ad5adaad81affa21e1914bd32cc0bd1827f8c151c772f4e852dcd368"
OLD_FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def current_piece(path, selector):
    raw = path.read_text()
    if selector == "before-first-section":
        return raw.split("\n## ", 1)[0] + "\n"
    section = selector.split(":", 1)[1]
    match = re.search(r"^## " + re.escape(section) + r" .+$", raw, re.M)
    end = re.search(r"^## ", raw[match.end():], re.M)
    return raw[match.start():match.end() + end.start() if end else None]


def main():
    if OUT.exists():
        raise SystemExit("Refuse to overwrite a callback receipt")
    assert sha(ORIGINAL.read_bytes()) == ORIGINAL_SHA
    original = read(ORIGINAL)
    frozen = read(FREEZE)
    prior_freeze = read(OLD_FREEZE)
    pages = {p["page_id"]: p for p in frozen["pages"]}
    prior_pages = {p["page_id"]: p for p in prior_freeze["pages"]}
    old_sources = {s["id"]: s for s in original["sources"]}
    contexts = []
    for identifier, source_id in [("12.17", "s_context_12_17"), ("14.19", "s_context_14_19")]:
        page = pages[identifier]
        snapshot = ROOT / page["snapshot"]
        data = snapshot.read_bytes()
        assert sha(data) == page["source_sha256"]
        assert data == current_piece(ROOT / page["source"], page["selector"]).encode()
        first_five = b"".join(data.splitlines(keepends=True)[:5])
        old_slice = old_sources[source_id]["inspected_slices"][0]
        assert sha(first_five) == old_slice["sha256"]
        assert data == (ROOT / prior_pages[identifier]["snapshot"]).read_bytes()
        contexts.append({
            "lesson_id": identifier, "current_snapshot": pin(snapshot),
            "current_source": page["source"], "current_selector": page["selector"],
            "current_source_slice_matches_worktree": True,
            "full_section_bytes_equal_original_freeze": True,
            "full_text_context_re_read": True,
            "claim_support_slice": {"line_start": 1, "line_end": 5,
                                    "sha256": sha(first_five), "source_quote": first_five.decode()},
            "support_scope": "Only c14's curriculum-placement statement; no new per-section method, program or figure certification.",
        })

    change_records = []
    for chapter, source_id in [(12, "s_context_12_17"), (14, "s_context_14_19")]:
        source = f"course/chapters/{chapter:02}.md"
        pieces = [p for p in prior_freeze["pages"] if p["source"] == source]
        pieces.sort(key=lambda p: 0 if p["page_id"] == f"chapter-{chapter}" else int(p["page_id"].split(".")[1]))
        old = b"".join((ROOT / p["snapshot"]).read_bytes() for p in pieces)
        assert sha(old) == old_sources[source_id]["sha256"]
        now = (ROOT / source).read_bytes()
        diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), now.decode().splitlines(keepends=True),
                                           fromfile="original-fix13/" + source, tofile="current-fix15/" + source, n=1))
        if chapter == 12:
            assert old.decode().replace("把 `7000.0` 改成 `6500.0`，核對最後一行仍印 True。", "把 `7000.0` 改成 `6500.0`，也把兩行print中的頻率名稱改成9500Hz、6500Hz，核對最後一行仍印 True。") == now.decode()
            meaning = "12.2 only adds exercise print-label updates (9500Hz/6500Hz). It does not alter sampling rate, common-time anchors, modality support, or the 12.17 prerequisite passage cited by c14."
        else:
            old_intro = current_piece_from_bytes(old)
            now_intro = current_piece_from_bytes(now)
            assert old[len(old_intro):] == now[len(now_intro):]
            meaning = "Chapter-14 introduction changes its one-layer-model link from 4.5 to 4.6 and explicitly routes readers to 14.19 after Looped sections. It does not change 14.19's time/height/width mechanism; its added route agrees with c14."
        change_records.append({"source": source, "original_whole_sha256": sha(old),
                               "current_whole_sha256_for_diff_only": sha(now), "exact_changed_text": diff,
                               "reviewer_semantic_finding": meaning,
                               "authority_scope": "Old reconstruction authenticates the original pin only. Current support comes from source15-selected snapshots matched to the current worktree."})

    current_intro = pages["chapter-20"]
    assert sha((ROOT / current_intro["snapshot"]).read_bytes()) == original["source_sha256"]
    assert current_piece(ROOT / current_intro["source"], current_intro["selector"]).encode() == (ROOT / current_intro["snapshot"]).read_bytes()
    assert current_intro["figures_sha256"] == original["figure_sha256"] == {}
    assert sha((ROOT / "course/chapters/20.md").read_bytes()) == old_sources["s_ch20"]["sha256"]

    unchanged_sources = []
    for source in original["sources"]:
        if source["kind"] == "repository_code" and source["id"] not in {"s_context_12_17", "s_context_14_19"}:
            path = ROOT / source["path"]
            assert sha(path.read_bytes()) == source["sha256"], source["id"]
            unchanged_sources.append({"id": source["id"], **pin(path)})
        elif source["kind"] in {"paper", "official_docs", "official_source"}:
            path = Path(source["original_cache_path"])
            assert sha(path.read_bytes()) == source["original_sha256"], source["id"]
            unchanged_sources.append({"id": source["id"], **pin(path),
                                      "reused_original_inspection": True})
    reused_artifacts = []
    for artifact in original["artifacts"]:
        path = ROOT / artifact["path"]
        assert sha(path.read_bytes()) == artifact["sha256"]
        reused_artifacts.append({"id": artifact["id"], **pin(path),
                                 "reuse": "Original execution/source receipt retained byte-for-byte; not executed again by this callback."})
    evidence = read(HERE / "evidence-results.json")
    input_pins = []
    input_pins.extend(evidence["chapter19_historical_initialization"]["moe_original_receipts"])
    input_pins.extend(evidence["lora_original_training"]["originals"])
    input_pins.extend(evidence["validation_selection"][key] for key in ["scores", "protocol", "selection", "public_release", "original_validation_result"])
    for row in input_pins:
        assert sha((ROOT / row["path"]).read_bytes()) == row["sha256"]

    findings = []
    for claim in original["claims"]:
        changed = claim["id"] == "c14"
        findings.append({"claim_id": claim["id"], "status": "still_supported",
                         "reason": "Current 12.17 common-time and 14.19 time/height/width passages re-read; their exact cited five-line slices and full sections are unchanged. Unrelated optional edits do not alter the chapter-20 prerequisite-placement statement." if changed else "The chapter-20 statement, its originally inspected repository/original-source bytes and execution artifacts are unchanged; the 12.2/14-introduction edits do not modify this claim's evidence or scope.",
                         "new_execution_of_examples": False})
    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-20-v1/callback_recheck_v1.py",
        "environment": {"python": platform.python_version(), "device": "CPU; text/JSON/hash callback only"},
        "review_kind": "Same reviewer current callback; no new fresh original, reader, whole-tool audit or seal.",
        "original_report": pin(ORIGINAL), "current_source_freeze": pin(FREEZE),
        "chapter20_introduction_and_full_chapter_bytes_unchanged": True,
        "current_contexts": contexts, "changes_and_semantic_findings": change_records,
        "unchanged_original_sources_rechecked": unchanged_sources,
        "reused_evidence": reused_artifacts, "original_primary_input_hashes_rechecked": len(input_pins),
        "claim_rechecks": findings,
        "figure_scope": "chapter-20 introduction has no figures; current c14 support uses prerequisite text only, not their figures.",
        "result": "All 14 chapter-20 introduction claims remain supported. Two broad prerequisite chapter pins are replaced in current report by exact current source15-selected canonical section snapshots. No new issue.",
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"original_sha256": ORIGINAL_SHA, "current_freeze_sha256": pin(FREEZE)["sha256"],
                      "rechecked_claims": len(findings), "reused_artifacts": len(reused_artifacts),
                      "current_contexts": [row["lesson_id"] for row in contexts], "result": output["result"]}, ensure_ascii=False))


def current_piece_from_bytes(raw):
    return raw.split(b"\n## ", 1)[0] + b"\n"


if __name__ == "__main__":
    main()
