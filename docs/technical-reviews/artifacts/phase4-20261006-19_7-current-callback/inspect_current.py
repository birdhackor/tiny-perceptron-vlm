"""Original-reviewer callback: verify exact limited change and immutable evidence."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-20261005-19_7-fresh"
TASK = "/root/phase4_factual_coordinator/factual_19_7"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i, h in enumerate(headers) if re.match(rb"^## 19\.7 ", h[0]))
    return raw[headers[i].start():headers[i + 1].start()]


old = (OLD / "inputs/section.md").read_bytes()
new = (ART / "inputs/section-current.md").read_bytes()
assert sha(old) == "713f6d3ddd667feedd5c4053db4641926484e3e94c943df0f9482670e39ca35c"
assert sha(new) == "c8ab7022b13b34697ddecb5b36eabeb1cca228dce415d538d60d21e3e9396c72"
assert section((ROOT / "course/chapters/19.md").read_bytes()) == new
assert old.count("不是隻差".encode()) == new.count("不是只差".encode()) == 1
assert old.replace("不是隻差".encode(), "不是只差".encode()) == new
fences = lambda raw: re.findall(rb"(?ms)^```python\n(.*?)^```\s*$", raw)
assert fences(old) == fences(new) and len(fences(new)) == 1

reuse = json.loads((ART / "fingerprint-reuse-receipt.json").read_bytes())
for row in reuse["live_sources_exactly_unchanged"]:
    assert sha((ROOT / row["path"]).read_bytes()) == row["current_sha256"] == row["prior_sha256"]
for row in reuse["priorproof_immutable_files"]:
    assert sha((ROOT / row["path"]).read_bytes()) == row["sha256"]

original = ROOT / "docs/course-experiments/capstone-evidence/deployment/test-joint.json"
raw = original.read_bytes()
assert sha(raw) == "6c6d027fc14470991445c8c2bda89ece4fe94803c5b88df3309be92369e6b66b"
data = json.loads(raw)
schema = {k: type(v).__name__ for k, v in data.items()}
assert schema["records"] == "list"
selected = []
pointers = []
for index, expected_raw in [(18, "DIRECT:0"), (20, "DIRECT:111")]:
    record = data["records"][index]
    # Only necessary original measurement leaves for the unchanged EOS/content claim.
    trace = record["final_trace"]
    ids = trace["generated_ids"]
    decoded = bytes(i - 8 for i in ids if i >= 8).decode("utf-8", errors="replace")
    assert trace["eos"] is True and trace["stop_reason"] == "eos" and ids[-1] == 2
    assert decoded == trace["raw"] == expected_raw
    assert record["runtime"]["status"] == "ok" and record["runtime"]["result"] == "1"
    assert record["end_to_end_correct"] is False
    row = {"record_pointer": f"/records/{index}", "final_eos": trace["eos"], "stop_reason": trace["stop_reason"], "final_raw": trace["raw"], "decoded_final": decoded, "runtime_status": record["runtime"]["status"], "runtime_result": record["runtime"]["result"], "end_to_end_correct": record["end_to_end_correct"]}
    selected.append(row)
    pointers.extend(f"/records/{index}/{suffix}" for suffix in ["final_trace/eos", "final_trace/stop_reason", "final_trace/generated_ids", "final_trace/raw", "runtime/status", "runtime/result", "end_to_end_correct"])

inspection = {
    "reviewer_task": TASK,
    "callback_kind": "same original independent correctness reviewer, limited current-version reinspection",
    "current_section_sha256": sha(new),
    "current_section_snapshot": str((ART / "inputs/section-current.md").relative_to(ROOT)),
    "current_figure_sha256": "a9c3d469eff7a1377f5e2c53c39583ddab6664c83657e82876e1a9275a82621a",
    "actual_course_read_scope": ["current19.7 full section", "original frozen19.7 section to produce and read the true diff"],
    "intro": None,
    "necessary_context": [],
    "changed_text": {"before": "所以不是隻差一個結束符號。", "after": "所以不是只差一個結束符號。", "independent_assessment": "The single classifier/spelling correction leaves the factual assertion unchanged: both failing final generations reached EOS, so the failure is answer content, not an absent EOS."},
    "current_original_measurement_schema": schema,
    "current_original_measurement_path": str(original.relative_to(ROOT)),
    "current_original_measurement_sha256": sha(raw),
    "current_original_measurement_pointers_read": pointers,
    "current_original_measurements": selected,
    "current_fence_sha256": sha(fences(new)[0]),
    "reused_support_scope": {"original_code_and_CPU": "Same original fence, parser/runtime, same-model user followup, scalar trace decoding/score/family checks and original stage provenance remain supported by the original personally executed evidence. Exact source and evidence fingerprints were checked; no original fence or model execution repeated.", "original_authorities": "Previously personally fetched/inspected OpenAI Model Spec, SDK tool message schema and arXiv DPO v3 bytes are immutable and support the unchanged concept claims; no new fetch.", "figure": "Same SVG SHA and no changed visual assertion; prior intrinsic/mobile Chromium renders and personal viewing remain evidence for this exact image. No unrelated new rendering or whole-page inspection."},
    "website": None,
    "independence": "No peer verdict, repair expectation, root parity or author extra result interpretation used. Prior canonical/proof were saved as opaque exact bytes. This current decision follows my own full current-section read, true frozen diff, raw EOS/content leaves and exact dependency fingerprints.",
    "environment": {"python": sys.version, "implementation": platform.python_implementation(), "device": "CPU; Python hashes/JSON/byte decode only"},
    "events_and_limits": ["Filename locator initially tried nonexistent website/ directory; no HTML read there, then exact19.7.html filename discovery and coordinator locator request.", "Only two necessary historical result-record leaves decoded; no model instantiated, no weights loaded, no GPU/training/download/full recipe/new agent/source edits.", "Initial and current complete-chapter snapshots retain their own frozen-input meaning; neither is relabeled as a latest full-chapter fingerprint."],
    "verdict": "pass",
    "unresolved_issues": [],
}
(ART / "current-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "source_sha256": sha(new), "diff": "隻差 -> 只差 only", "fence_changed": False, "unchanged_sources": len(reuse["live_sources_exactly_unchanged"]), "immutable_priorproof_files": len(reuse["priorproof_immutable_files"]), "necessary_context": [], "selected_EOS_measurements": selected, "independent_verdict": "pass"}, ensure_ascii=False, indent=2))
