"""Same-reviewer complete section recheck after the necessary text repair."""
from pathlib import Path
import hashlib
import json
import platform
import re
import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
initial_report = json.loads((ART / "initial-report.json").read_bytes())
initial_section = (ART / "19.12-initial-source.md").read_bytes()
chapter = (ROOT / "course/chapters/19.md").read_bytes()
heading = re.search(rb"^## 19\.12 .+$", chapter, re.M)
next_heading = re.search(rb"^## ", chapter[heading.end():], re.M)
end = heading.end() + next_heading.start() if next_heading else len(chapter)
current = chapter[heading.start():end]
before = "MoE仍有15項、Dense有12項原數值判準未達，因此".encode()
after = "兩版的真正工具往返都低於95%，因此".encode()
assert initial_section.count(before) == 1
assert initial_section.replace(before, after) == current
assert sha(initial_section) == initial_report["source_sha256"]
assert b".svg)" not in current
hash_checks = []
for item in initial_report["artifacts"] + initial_report["sources"]:
    if "path" not in item:
        continue
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"], item["path"]
    hash_checks.append({"path": item["path"], "sha256": actual, "unchanged": True})
rates = {}
for architecture in ["moe", "dense"]:
    prefix = ROOT / f"docs/selftrained/results/public-raw/{architecture}"
    frozen = json.loads((prefix / "freeze/frozen.json").read_bytes())
    metrics = json.loads((prefix / "test/metrics.json").read_bytes())
    chain = metrics["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
    minimum = frozen["thresholds"]["tool_roundtrip"]
    calculated = chain["numerator"] / chain["denominator"]
    assert minimum == metrics["thresholds"]["tool_roundtrip"] == 0.95
    assert (chain["numerator"], chain["denominator"]) == ((0, 276) if architecture == "moe" else (7, 276))
    assert calculated == chain["rate"] and calculated < minimum
    rates[architecture] = {**chain, "recalculated_rate": calculated, "required_minimum": minimum, "fails_required_condition": True}
(ART / "19.12-after-repair-source.md").write_bytes(current)
out = {
    "reviewer_task": "/root/p6_fact_19_12",
    "complete_current_section_read": True,
    "current_section_sha256": sha(current),
    "initial_section_sha256": sha(initial_section),
    "only_change": {"before": before.decode(), "after": after.decode()},
    "retained_initial_report_sha256": sha((ART / "initial-report.json").read_bytes()),
    "unchanged_evidence": hash_checks,
    "tool_roundtrip": rates,
    "conclusion": "Both architectures fail the independently required original 95% tool-roundtrip condition, so neither passes all capability requirements. Completion is separate from capability acceptance.",
    "historical_count_scope": "Original 15/12 values remain reproducible under the derivative mapping; original full mapping predeclaration remains unsupported. The current section no longer makes that count attribution.",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "new_model_generation": False,
    "new_heldout_evaluation": False,
    "training": False,
    "paid_calls": False,
}
(ART / "repair-recheck-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"current_section_sha256": out["current_section_sha256"], "evidence_hashes_unchanged": len(hash_checks), "tool_roundtrip": rates, "initial_report_preserved": True}, ensure_ascii=False, indent=2))
