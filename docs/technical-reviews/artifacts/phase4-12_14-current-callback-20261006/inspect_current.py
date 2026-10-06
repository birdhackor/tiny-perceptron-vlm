"""Same-reviewer narrow callback: versions and dependencies, no CPU/model rerun."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = "/root/phase4_factual_coordinator/factual_12_14"
PRIOR_ROOT = ROOT / "docs/technical-reviews/artifacts/phase4-12_14-independent"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def section(p, identifier):
    raw = p.read_bytes()
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headings) if h[0].startswith(("## " + identifier + " ").encode()))
    start = headings[index].start()
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return raw[start:end]


def necessary_context(text):
    prefix = text.split("```", 1)[0]
    pieces = [prefix]
    for paragraph in text.split("\n\n"):
        if any(w in paragraph for w in ["多輪", "历史", "歷史"]) and paragraph not in prefix:
            pieces.append(paragraph + "\n")
    return "\n".join(pieces)


prior_path = OUT / "prior-12.14-report.opaque.json"
prior = json.loads(prior_path.read_text())
assert prior["reviewer_task"] == TASK
current_section = section(ROOT / "course/chapters/12.md", "12.14")
assert current_section == (OUT / "current-section12_14.md").read_bytes()
assert hashlib.sha256(current_section).hexdigest() == prior["source_sha256"] == "d23cef283cc8fb121c6483f8f4d7dae085fc8875d1987806e59fc2804024465f"
figure = ROOT / "course/figures/rewrite-12-asr-history.svg"
assert sha(figure) == prior["figure_sha256"][figure.relative_to(ROOT).as_posix()] == "32dff30177d59bb231d7bc00c7c866d46087cd2791536eb4556b6e528e1191a4"
context = section(ROOT / "course/chapters/07.md", "7.1")
assert context == (OUT / "current-7_1-raw-section.md").read_bytes()
prior_context = (PRIOR_ROOT / "inputs/prerequisite-7_1.md").read_bytes()
current_slices = necessary_context(context.decode())
assert current_slices == necessary_context(prior_context.decode())
assert current_slices == (OUT / "current-7_1-necessary-slices.md").read_text()
assert "每條用role記user或assistant，用content記原文字" in current_slices
assert "多輪則按消息順序保留角色" in current_slices

fingerprints = []
for artifact in prior["artifacts"]:
    path = ROOT / artifact["path"]
    observed = sha(path)
    assert observed == artifact["sha256"], artifact["id"]
    fingerprints.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed, "unchanged": True})
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        snapshot = ROOT / source["path"]
        current_path = ROOT / snapshot.relative_to(PRIOR_ROOT / "code")
        assert sha(current_path) == source["sha256"], str(current_path)
        fingerprints.append({"id": "current_" + source["id"], "path": current_path.relative_to(ROOT).as_posix(), "sha256": sha(current_path), "unchanged": True})

hf = PRIOR_ROOT / "sources/hf-chat-v4.57.1.md"
assert sha(hf) == "f822eb57b47c467f120358b0088a1b69575b4ae4fe3807fb23ac6073d759dbab"
lines = hf.read_text().splitlines()
assert "role" in lines[77] and "content" in lines[77]
assert "formatted sequence" in lines[83]

environment = {"python": platform.python_version(), "python_executable": sys.executable,
               "device": "CPU (stdlib version/hash inspection only)", "shell": "bash login:false",
               "new_training": False, "new_model_inference": False, "network_retrieval": False,
               "prior_cpu_rerun": False, "prior_render_rerun": False}
(OUT / "current-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")

scope_checks = [
    {"claim_id": "asr_history", "decision": "support retained", "reason": "Own section and SVG identical. Original Whisper audio-to-transcript support and HF role/content-to-sequence support retain the same meaning; neither verifies ASR/chat quality."},
    {"claim_id": "cer_numbers", "decision": "support retained", "reason": "Same four-codepoint reference, one substitution and same fence/helper. Prior exact1/4 and zero-edit CPU cases retained at identical code/stdout/environment hashes; not rerun."},
    {"claim_id": "handwritten_contract", "decision": "support retained", "reason": "Same handwritten calls and helper; equality and None still only measure agreement, with no recipe/requirement judgement. Prior bounded CPU evidence reused."},
    {"claim_id": "separate_judgements", "decision": "support retained", "reason": "Same negation example and independent derivation. No claim that LLMs can never infer a transcription mistake; shared code does not guarantee recovering missing intent."},
    {"claim_id": "history_contract", "decision": "support retained", "reason": "Current7.1 necessary role/content and ordered-turn slices equal prior slices. Official immutable HF original personally re-read at lines23–27,38–44,76–84. Original UI generation-stub contract proof retained, no learned compliance claim."},
    {"claim_id": "base_extension_configuration", "decision": "support retained", "reason": "Original manifest, option-mapping loader and core code have identical current hashes. Only pinned base/configuration fact reused, no performance/download/release-quality rerun."},
]
inspection = {
    "schema_version": 1, "kind": "same_original_reviewer_current_inspection", "reviewer_task": TASK,
    "callback_label": "20261006 narrow technical current callback", "independence_meaning": "Same original technical reviewer; original fresh review remains20261005. This is not a new independent fresh reviewer.",
    "primary": {"source": "course/chapters/12.md#12.14", "source_sha256": hashlib.sha256(current_section).hexdigest(), "current_full_section_personally_read": True, "intro": None},
    "figure": {"path": figure.relative_to(ROOT).as_posix(), "sha256": sha(figure), "unchanged": True, "reuse": "Original personally viewed native640x725 Inkscape render; not rerendered. Original Chromium timeout and website-size limitation remain."},
    "context": {"source": "course/chapters/07.md#7.1", "prior_section_sha256": hashlib.sha256(prior_context).hexdigest(), "current_section_sha256": hashlib.sha256(context).hexdigest(),
                "current_necessary_slices_sha256": hashlib.sha256(current_slices.encode()).hexdigest(), "whole_section_changed": context != prior_context, "necessary_slices_unchanged": True,
                "read_scope": "Opening before first fence plus multi-turn/history paragraph, saved exact necessary slices. Needed claims: role/content list and preserving roles/turn order. Token-ID, shift/loss or SFT details are outside this12.14 callback dependency."},
    "human_inspection": {"own_section": "Personually re-read complete current12.14 including details; no new substantive or visual claim.", "current_context": "Personually read the saved current7.1 necessary slices and checked them against immutable official HF text; current prose is a claim being checked, not authority.", "source_locators": "Re-read HF original lines18–28,34–85. Correct exact Using apply_chat_template locator from broad prior83–94 to76–84; role/content is line78 and formatted sequence is line84. This is report locator refinement, no textbook/content error.", "scope_checks": scope_checks},
    "automated_checks": {"assertions_passed": True, "meaning": "Byte/hash/version and named sentence assertions only; conceptual support assessment above was personally inspected.", "evidence_fingerprints": fingerprints},
    "prior_retention": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": sha(prior_path), "read_permission_scope": "Only this same reviewer's prior claim scopes/evidence/version fields were examined after opaque preservation, as explicitly requested by this callback. No other old reviews read."},
    "reuse": {"authority_sources": "Immutable Whisper v1, Transformers v4.57.1 commit, TorchMetrics v1.8.2 and original Python3.13.16 fragments reused at precise original hashes. No new download; HF relevant text re-read.",
             "cpu": "Original20261005 executed fences/small changes and bounded UI/options proof reused; no repeat. Generation stub and manifest-validation stub remain explicit.", "historical_context": "Other prior prerequisite snapshots remain historical frozen inputs, not asserted as current whole chapters. Only current12.14 and truly necessary current7.1 scope checked."},
    "events": [{"type": "own_report_locator_refinement", "substantive_error": False, "detail": "HF dictionary-role/content locator refined to78 and input contract section76–84; preceding82 rather than83 contains role bullets."}],
    "contamination_event": None, "verdict": "pass", "environment": environment,
}
(OUT / "current-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "primary_source_sha256": inspection["primary"]["source_sha256"],
                  "current_context_sha256": inspection["context"]["current_section_sha256"], "necessary_slice_sha256": inspection["context"]["current_necessary_slices_sha256"],
                  "primary_unchanged": True, "figure_unchanged": True, "context_necessary_slices_unchanged": True,
                  "prior_evidence_fingerprint_checks": len(fingerprints), "new_cpu_or_render_runs": 0,
                  "inspection_path": (OUT / "current-inspection.json").relative_to(ROOT).as_posix(), "inspection_sha256": sha(OUT / "current-inspection.json")}, ensure_ascii=False))
print("CURRENT INSPECTION ASSERTIONS PASSED")
