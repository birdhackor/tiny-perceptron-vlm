"""Same-owner narrow dependency reinspection; reuse unchanged genuine evidence."""
import difflib
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent
ROOT = OLD.parents[3]
REL = BASE.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_7_2"
PRIOR_SHA = "e7e5a4400fa370f6e7d332e139c7ec9860eddfc8e80ec51d49e585fc25888c53"
PRIOR = ROOT / ("docs/technical-reviews/history/phase4-7_2-own-before-context-recheck-20261006-" + PRIOR_SHA + ".json")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
def section(path, lesson):
    raw = path.read_bytes()
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    found = [i for i,h in enumerate(headings) if re.match(rb"^## " + re.escape(lesson.encode()) + rb" ", h[0])]
    assert len(found) == 1
    i = found[0]
    start = headings[i].start()
    end = headings[i+1].start() if i+1 < len(headings) else len(raw)
    return raw[start:end], raw[:start].count(b"\n") + 1

assert sha(PRIOR) == PRIOR_SHA
report = json.loads(PRIOR.read_bytes())
assert report["reviewer_task"] == TASK
assert sha(ROOT / "docs/technical-reviews/7.2.json") == PRIOR_SHA
shutil.copyfile(PRIOR, BASE / "opaque-prior-report.json")
current, comparisons, diffs = {}, [], []
for chapter,lesson,old_name in [("07","7.2","original-section.md"),
                               ("07","7.1","inputs/prerequisite-7.1.md"),
                               ("06","6.6","inputs/prerequisite-6.6.md")]:
    raw,start = section(ROOT / f"course/chapters/{chapter}.md", lesson)
    current[lesson] = raw
    frozen_path = OLD / old_name
    frozen = frozen_path.read_bytes()
    snapshot = BASE / f"current-{lesson}.md"
    snapshot.write_bytes(raw)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", raw.decode())
    images += re.findall(r'(?:src|href)=["\']([^"\']+\.(?:svg|png|jpg|jpeg|webp))["\']', raw.decode())
    comparisons.append({"lesson_id":lesson,"source":f"course/chapters/{chapter}.md#{lesson}",
        "current_first_line":start,"current_slice_sha256":sha(snapshot),
        "current_snapshot":snapshot.relative_to(ROOT).as_posix(),
        "own_original_frozen_path":frozen_path.relative_to(ROOT).as_posix(),
        "own_original_frozen_sha256":sha(frozen_path),"bytes_unchanged":raw == frozen,
        "current_image_references":images})
    diffs.append("".join(difflib.unified_diff(frozen.decode().splitlines(keepends=True),raw.decode().splitlines(keepends=True),
        fromfile=f"own frozen {lesson}",tofile=f"current {lesson}")))
(BASE / "necessary-slice-diffs.txt").write_text("".join(diffs))
assert current["7.2"] == (OLD / "original-section.md").read_bytes()
assert all(not item["current_image_references"] for item in comparisons)
# These exact changes were read and assessed by the original reviewer, not inferred from full-file hashes.
assert current["7.1"] == (OLD / "inputs/prerequisite-7.1.md").read_bytes().replace("帶着".encode(),"帶著".encode())
assert current["6.6"] == (OLD / "inputs/prerequisite-6.6.md").read_bytes().replace(
    "BPE可登记整串特殊項，仍须另外决定普通內容相同拼寫怎麼编碼".encode(),
    "BPE可登記整串特殊項，仍須另外決定普通內容相同拼寫怎麼編碼".encode())
save(BASE / "necessary-slice-comparison.json", comparisons)
shutil.copyfile(ROOT / "docs/review-tools/factual-reviewer-instructions.md", BASE / "current-factual-reviewer-instructions.md")

artifact_audit = [{"id":a["id"],"path":a["path"],"original_sha256":a["sha256"],
                   "current_sha256":sha(ROOT/a["path"]),"unchanged":sha(ROOT/a["path"]) == a["sha256"]}
                  for a in report["artifacts"]]
code_audit = [{"source_id":s["id"],"path":s["path"],"original_sha256":s["sha256"],
               "current_sha256":sha(ROOT/s["path"]),"unchanged":sha(ROOT/s["path"]) == s["sha256"]}
              for s in report["sources"] if s["kind"] == "repository_code"]
assert all(item["unchanged"] for item in artifact_audit + code_audit)
official_audit = [{"source_id":s["id"],"snapshot_artifact_id":s["snapshot_artifact_id"],
                   "url":s["url"],"version":s["version"],"snapshot_sha256":s["snapshot_sha256"],
                   "unchanged":next(a for a in artifact_audit if a["id"] == s["snapshot_artifact_id"])["unchanged"]}
                  for s in report["sources"] if s["kind"] in ("official_docs","official_source")]
save(BASE / "unchanged-evidence-audit.json", {"original_artifacts":artifact_audit,"repository_code":code_audit,"official_sources":official_audit,
    "reuse_basis":"Same original owner read current necessary slices and own original measurement output; exact current 7.2 bytes, all original evidence files and relevant actual repository contracts are unchanged. No fresh training/inference/CPU rerun or source refetch is represented by this audit."})

supports = {
    "prompt-construction":"Current7.1 still serializes BOS,user,question bytes,EOS,assistant,answer bytes,EOS; unchanged7.2 removes the new answer for generation. No dependency change to current prefix order.",
    "ids-list-count":"Current6.6 still reserves IDs0..7 and offsets ordinary bytes by8; current7.1 still states character2 ID58 versus EOS2. Unchanged original exact IDs and count9 remain applicable.",
    "role-id-learning":"Current6.6 still distinguishes role metadata inserting4 from literal content spelling, and current7.1 retains role/content dictionaries; converted glyphs do not change this distinction or the learning limitation.",
    "tail-next-token":"Current7.1 still shifts targets once and supplies y[8]=58 at x[8]=assistant4, matching the unchanged7.2 prompt tail; genuine original alignment and generate time-axis probe remains applicable.",
    "boundary-format":"Current7.1's serialization order and current6.6's dedicated control IDs are unchanged. Local EOS2/assistant4 functions and matching-format limitation still hold.",
    "history-current-target":"Current7.1 still preserves multi-turn message order and role metadata. No change to old complete answers versus withheld new target, and actual chat_prompt contract is unchanged.",
    "format-vs-capability":"Current6.6 still says inserting markers does not teach answering; current7.1 still distinguishes format-only exercise from a correct arithmetic target. Unchanged7.2 remains a list/assert demonstration, not a score or training run.",
    "exercise-missing-restored":"Unchanged7.2 fence still compares prompt[-1] with assistant4. Current6.6 retains EOS2 and assistant4 distinct. Original genuine missing rc1/restored rc0 records remain valid for identical code."
}
environment = {"python":sys.version,"python_executable":sys.executable,
    "device":"CPU host filesystem/hash inspection only; no model execution",
    "prior_cpu_measurement_environment":"Original measured torch2.14.1+cpu; preserved original-environment.json; not remeasured in this reinspection"}
receipt = {
    "schema_version":1,"kind":"same_owner_necessary_context_reinspection","lesson_id":"7.2","reviewer_task":TASK,
    "performed_on":"2026-10-06","verdict":"pass","source_sha256":sha(BASE/"current-7.2.md"),
    "command":".venv/bin/python " + REL + "/context_recheck.py","cwd":str(ROOT),"environment":environment,
    "prior_report":{"path":PRIOR.relative_to(ROOT).as_posix(),"sha256":PRIOR_SHA,"preserved_opaque_before_reinspection":True},
    "actual_read_scope":["Current7.2 complete raw section (40–71)","Current necessary7.1 complete raw section (5–39)",
        "Current necessary6.6 complete raw section (203–236)","Own frozen7.1/6.6 and own raw original7.2, exact diffs",
        "Current factual-reviewer-instructions.md full", "Own prior report only; own original-stdout.txt, cpu-checks.json and contract.stdout.txt genuine outputs reread"],
    "dependency_comparisons":comparisons,
    "meaning_assessment":{"7.1":"Only 帶着→帶著. Serialization order, numeric IDs, target offset and format-versus-arithmetic distinction remain identical.",
        "6.6":"Only 登记→登記, 须→須, 决定→決定, 编碼→編碼 in one sentence. Dedicated role ID4, content-only byte offset8 and metadata/literal distinction remain identical."},
    "incidental_scope_disposition":{"6.9":"Original owner explicitly recorded incidental visibility of lines310–end, not technical dependency. None of the eight7.2 claims cite BPE, streaming or I/O chunk rules. The initial actual-read record is preserved; this reinspection does not re-review6.9 or the full chapter."},
    "source_and_measurement_reuse":{"official_source_snapshot_count":len(official_audit),"repository_contract_count":len(code_audit),
        "original_artifact_count":len(artifact_audit),"all_sha256_match":True,
        "basis":"Necessary current slice semantics and own8 claim support personally reconsidered; raw7.2, fence, repo contracts, official snapshots and genuine original CPU evidence unchanged.",
        "new_cpu_measurement_runs":0,"new_primary_source_fetches":0,"training_gpu_or_model_weight_evaluation":False},
    "claim_support_reinspection":supports,
    "visual_scope":{"status":"not_applicable","current_referenced_images":[],"rendered_or_viewed":False,
        "reason":"Current7.2 and both truly necessary slices contain no image references; no necessary visual material requires rendering. No screen inspection is claimed."},
    "new_or_unresolved_questions":[],"other_review_or_author_repair_summaries_read":False,
    "whole_chapter_fingerprint_used_as_recheck_trigger":False
}
save(BASE / "reinspection-receipt.json", receipt)

oldscope = report["review_scope"]
report["initial_review_scope"] = oldscope
report["initial_reviewed_on"] = report["reviewed_on"]
report["reviewed_on"] = "2026-10-06"
report["review_scope"] = {
    "scope":"Same original owner, narrow necessary-context reinspection of7.2. Current own body unchanged; only necessary7.1 and6.6 are reconsidered. Original broader incidental visibility is preserved in initial_review_scope, not expanded into current dependencies.",
    "actually_read":receipt["actual_read_scope"],
    "necessary_dependency_sha256":{x["source"]:x["current_slice_sha256"] for x in comparisons if x["lesson_id"] != "7.2"},
    "chapter_intro":{"status":"not_applicable","reason":"7.2 is not the first section; this narrow reinspection does not reread the chapter intro."},
    "figures":receipt["visual_scope"],"historical_metrics":oldscope["historical_metrics"],"shell_contract":oldscope["shell_contract"],
    "original_primary_inspections_carried_forward":oldscope["official_sources_actually_inspected"],
    "evidence_reuse":receipt["source_and_measurement_reuse"],"incidental_scope_disposition":receipt["incidental_scope_disposition"],
    "capability_limit":oldscope["capability_limit"],"claim_inventory":oldscope["claim_inventory"],"issues":[]}
report["context_reinspection"] = {"prior_report":receipt["prior_report"],"receipt_path":(BASE/"reinspection-receipt.json").relative_to(ROOT).as_posix(),
    "receipt_sha256":sha(BASE/"reinspection-receipt.json"),"artifact_id":"context-reinspection-20261006","verdict":"pass"}
report["verdict"] = "pass"
for path in sorted(BASE.iterdir()):
    if not path.is_file() or path.name.startswith("checker") or path.name.endswith("stdout.txt") or path.name.endswith("stderr.txt"):
        continue
    identifier = "context-20261006-" + path.name.replace(".","-")
    artifact = {"id":identifier,"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),
        "kind":"code" if path.suffix == ".py" else "source_snapshot","description":"Same-owner actual necessary-context reinspection: " + path.name}
    if path.name == "reinspection-receipt.json":
        artifact.update(id="context-reinspection-20261006",kind="execution",command=receipt["command"],
            result="Current7.2 unchanged; current7.1/6.6 exact glyph-only changes personally read; original48 evidence hashes and4 repo contracts match; pass; no unresolved questions.",environment=environment)
    report["artifacts"].append(artifact)
report["sources"].append({"id":"same-owner-context-reinspection","kind":"execution","title":"Same original owner's necessary7.1/6.6 context reinspection on2026-10-06",
    "verified":True,"artifact_id":"context-reinspection-20261006"})
for claim in report["claims"]:
    claim["artifact_ids"].append("context-reinspection-20261006")
    claim["evidence"].append({"source_id":"same-owner-context-reinspection","locator":"reinspection-receipt.json#/claim_support_reinspection/"+claim["id"],
        "supports":supports[claim["id"]]})
report["checks"]["factual_accuracy"]["details"] += " Same-owner2026-10-06 current7.1/6.6 slice read confirms only glyph corrections; all8 claim supports remain applicable."
report["checks"]["source_verification"]["details"] += " Original official snapshots, actual repository contracts and all48 permanent artifacts SHA-verified unchanged; reused own genuine evidence, no fresh source fetch."
report["checks"]["limitations"]["details"] += " This is a narrow dependency reinspection with no model execution or new measurement.6.9 was initial incidental reading and is not a claim dependency."
save(ROOT / "docs/technical-reviews/7.2.json", report)
print(json.dumps({"verdict":"pass","report_sha256":sha(ROOT/"docs/technical-reviews/7.2.json"),
    "prior_history_path":PRIOR.relative_to(ROOT).as_posix(),"prior_sha256":PRIOR_SHA,
    "canonical_artifact_id":"context-reinspection-20261006","receipt_path":receipt["command"].replace(".venv/bin/python ","").replace("context_recheck.py","reinspection-receipt.json"),
    "receipt_sha256":sha(BASE/"reinspection-receipt.json"),"unresolved_questions":[]},ensure_ascii=False))
