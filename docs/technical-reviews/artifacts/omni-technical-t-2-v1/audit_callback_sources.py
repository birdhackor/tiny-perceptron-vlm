"""T.2 same-owner bounded callback: compare source bytes, never run training."""

import difflib
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ORIGINAL = "docs/course-revision-20261029-omni/reviews/technical/technical_t_2/technical-original-v1.json"
OLD_NUMBERED = "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"
NEW_NUMBERED = "docs/course-revision-20261029-omni/reviews/numbered-freeze-v13.json"
NEW_CANONICAL = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    return {"path": path, "sha256": digest((ROOT / path).read_bytes())}


def document(path):
    return json.loads((ROOT / path).read_bytes())


def frozen(page):
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    return raw


def find(freeze, page_id):
    return next(p for p in freeze["pages"] if p["page_id"] == page_id)


def section(raw, page_id):
    text = raw.decode("utf-8")
    first = re.search(r"^## " + re.escape(page_id) + r" .+\n", text, re.M)
    assert first
    following = re.search(r"^## ", text[first.end():], re.M)
    end = first.end() + following.start() if following else len(text)
    return text[first.start():end].encode("utf-8")


def chapter(freeze, source):
    pages = [p for p in freeze["pages"] if p["source"] == source]
    intro = next(p for p in pages if p["selector"] == "before-first-section")
    sections = [p for p in pages if p["selector"].startswith("section:")]
    assert len(pages) == len(sections) + 1
    return b"".join(frozen(p) for p in [intro] + sections), [intro] + sections


def main():
    original = document(ORIGINAL)
    assert reference(ORIGINAL)["sha256"] == "5b8debac5f736ad1f46555cd34b1d6da35bee15bf977f74b8dc3dc26158b85fc"
    old_numbered, new_numbered = document(OLD_NUMBERED), document(NEW_NUMBERED)
    old_canonical_ref = old_numbered["canonical_source_freeze"]
    assert reference(old_canonical_ref["path"]) == old_canonical_ref
    assert new_numbered["canonical_source_freeze"] == reference(NEW_CANONICAL)
    old_canonical, new_canonical = document(old_canonical_ref["path"]), document(NEW_CANONICAL)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": original["reviewer_task"],
        "callback_kind": "same-owner bounded primary-source drift review; not a new fresh review",
        "original_report": reference(ORIGINAL),
        "original_numbered_freeze": reference(OLD_NUMBERED),
        "original_canonical_freeze": old_canonical_ref,
        "current_numbered_freeze": reference(NEW_NUMBERED),
        "current_canonical_freeze": reference(NEW_CANONICAL),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU source-byte comparison only"},
        "whole_chapter_comparisons": [], "current_exact_prerequisites": [],
    }
    for source in ("course/chapters/05.md", "course/chapters/10.md"):
        old_raw, old_pages = chapter(old_canonical, source)
        current_raw, _ = chapter(new_canonical, source)
        original_source = next(s for s in original["sources"] if s.get("path") == source)
        assert digest(old_raw) == original_source["sha256"]
        assert current_raw == (ROOT / source).read_bytes()
        changed = [p["page_id"] for p in old_pages if p["source_sha256"] != find(new_canonical, p["page_id"])["source_sha256"]]
        result["whole_chapter_comparisons"].append({
            "source": source, "original_reconstructed_sha256": digest(old_raw),
            "current_reconstructed_and_live_sha256": digest(current_raw),
            "changed_canonical_slices": changed,
            "complete_unified_diff": "".join(difflib.unified_diff(old_raw.decode().splitlines(keepends=True), current_raw.decode().splitlines(keepends=True), fromfile="original-freeze/" + source, tofile="current/" + source, n=3)),
        })
    assert result["whole_chapter_comparisons"][0]["changed_canonical_slices"] == ["5.7"]
    assert result["whole_chapter_comparisons"][1]["changed_canonical_slices"] == ["chapter-10"]

    for page_id in ("5.1", "10.7"):
        old_page, new_page = find(old_canonical, page_id), find(new_canonical, page_id)
        old_raw, new_raw = frozen(old_page), frozen(new_page)
        live = section((ROOT / new_page["source"]).read_bytes(), page_id)
        assert live == new_raw == old_raw
        result["current_exact_prerequisites"].append({
            "page_id": page_id, "authored_source": new_page["source"],
            "selector": new_page["selector"], "snapshot": reference(new_page["snapshot"]),
            "live_slice_sha256": digest(live), "identical_to_original_slice": True,
            "actual_read": "Current authored section prose and code read in full; equality checked only after reading current content.",
        })

    old_t2, new_t2 = find(old_numbered, "T.2"), find(new_numbered, "T.2")
    current_training_page = find(new_canonical, "training")
    assert frozen(current_training_page) == (ROOT / "course/training.md").read_bytes()
    assert section(frozen(current_training_page), "T.2") == frozen(old_t2) == frozen(new_t2)
    assert old_t2["figures_sha256"] == new_t2["figures_sha256"] == {}
    result["T.2"] = {"current_snapshot": reference(new_t2["snapshot"]), "current_parent_snapshot": reference(current_training_page["snapshot"]), "identical_to_original_slice": True, "figures_sha256": {}, "actual_read": "Current introduction, T.1 context, complete T.2 and transition to T.3 heading were read; callback verdict is T.2 only."}
    result["unchanged_implementation_sources"] = []
    for s in original["sources"]:
        if s["kind"] == "repository_code" and not s["id"].startswith("prereq_"):
            current = reference(s["path"])
            assert current["sha256"] == s["sha256"]
            result["unchanged_implementation_sources"].append(current)
    result["reused_original_artifacts"] = []
    for a in original["artifacts"]:
        current = reference(a["path"])
        assert current["sha256"] == a["sha256"]
        result["reused_original_artifacts"].append({"id": a["id"], **current})
    result["execution_reuse"] = "Original exact CPU commands, instrumentation, fixtures, mask probes and official-source inspection are retained with identical bytes. No model forward/backward, optimizer step, training, GPU, paid job or publishing command was rerun during this callback."
    result["reviewer_difference_judgment"] = "The sole 5.7 edit narrows weights_only loading safety wording; T.2 neither loads a checkpoint in its three commands nor makes a loading-safety claim. Chapter-10 introduction now identifies retrieval sections 10.9–10.11 and video section 10.12; it does not change 10.7 expansion/shift or the forward/backward/quality distinction. These edits do not alter support for T.2."
    reasons = {
        "c1": "Current T.2 keeps numerical passage separate from model ability; source changes do not add quality evidence.",
        "c2": "Exact three CPU commands and task selection are unchanged; original actual command evidence applies.",
        "c3": "toy_documents and prepare_examples are byte-identical; synthetic color/shape input remains supported.",
        "c4": "toy_conversations and render_chat are byte-identical; small-integer addition and assistant targets remain supported.",
        "c5": "scene, modal_loss and MultiModalLM are unchanged; synthetic circle/square behavior remains supported.",
        "c6": "--data parser and loaders are unchanged; three original valid JSONL fixture runs remain applicable.",
        "c7": "steps=1 dry-run and per-sample vision forward followed by one aggregate backward remain unchanged.",
        "c8": "Both optimizer.step and save calls remain gated by --train; original unchanged-weight/call-count evidence is reused.",
        "c9": "Current forward uses unchanged model parameters and returns logits; pinned original official autograd explanation still applies.",
        "c10": "Derivative/.grad meaning is unchanged, including inactive graph leaves restriction; current T.2 retains its parameter-gradient caveat.",
        "c11": "Backward versus optimizer-step distinction is unchanged; 5.7 loading wording is outside this distinction.",
        "c12": "The two stdout JSON lines and history[0] layout remain unchanged.",
        "c13": "Finite loss values are original observed CPU outputs, explicitly reused rather than represented as new execution.",
        "c14": "Pre-clip finite positive norm values are original observations; total-norm reasoning and implementation are unchanged.",
        "c15": "effective_tokens counting remains unchanged; original text=111 and sft=8 observations are reused.",
        "c16": "User/PAD target masking and nonignored denominator are byte-identical; question input may still affect answer indirectly.",
        "c17": "Vision still assigns effective=None while retaining supervised answer labels; original null and target probes apply.",
        "c18": "Reread current 5.1 includes embedding.weight.grad.norm().item() and the signal/quality limit; exact current 5.1 equals original.",
        "c19": "Reread current 10.7 preserves pre-shift labels, expansion then one shift, and the (1,6,8)/73,2 example; exact slice and implementation equal original.",
    }
    assert set(reasons) == {c["id"] for c in original["claims"]}
    result["all_original_claim_dispositions"] = [{"claim_id": c["id"], "status": "still_supported", "reason": reasons[c["id"]]} for c in original["claims"]]
    result["issues_found"] = []
    target = HERE / "callback-source-drift-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"result": "all source comparisons passed", "artifact": reference(target.relative_to(ROOT).as_posix()), "claims_reassessed": len(reasons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
