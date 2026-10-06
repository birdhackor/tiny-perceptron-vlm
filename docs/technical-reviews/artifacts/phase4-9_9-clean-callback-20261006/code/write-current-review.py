"""Same owner's schema update after actual narrow callback, preserving all old raw proof."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
A = Path(__file__).resolve().parents[1]
REL = A.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_9_9_clean"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior_path = A / "prior/9.9-prior-complete-report.opaque.json"
prior_raw = prior_path.read_bytes()
assert sha(prior_path) == "fbfdd78723ac9099d1b77b9718994c8a571e06bd5659e9388c3be762834cdd83"
report = json.loads(prior_raw)
assert report["reviewer_task"] == TASK
inspection = json.loads((A / "execution/actual-inspection.json").read_text())
receipt = json.loads((A / "execution/callback-execution-receipt.json").read_text())
assert inspection["verdict"] == "pass" and not inspection["substantively_changed_claim_ids"]
assert all(c["matches"] for c in inspection["exact_evidence_fingerprint_checks"])
assert receipt["exit_code"] == 0 and receipt["actual_inspection_sha256"] == sha(A / "execution/actual-inspection.json")

def artifact(identifier, path, kind, description, extra=None):
    value = {"id":identifier,"path":f"{REL}/{path}","sha256":sha(A / path),"kind":kind,"description":description}
    if extra:
        value.update(extra)
    report["artifacts"].append(value)

artifact("callback-current-section", "inputs/current-section-9.9.md", "source_snapshot", "Actually read current full9.9 rawUTF8 section at callback; no newline normalization.")
artifact("callback-capture-receipt", "inputs/capture-receipt.json", "source_snapshot", "Actual currentsection/fence fingerprints and clearly labeled callback whole-file frozen observation.")
artifact("callback-wholefile-frozen-bytes", "inputs/current-whole-09-frozen.md", "source_snapshot", "Full currentMarkdown bytes preserved only to reproduce the capturewholefile hash; actual substantive reading/review scope remains9.9 only.")
artifact("callback-actual-diff", "inputs/actual-raw-section.diff", "derivation", "Actual raw diff personally inspected against this same owner's frozen source.")
artifact("callback-current-necessary-context", "inputs/current-needed-context-12.8.md", "source_snapshot", "Actually read necessary current12.8 context; raw-identical to owned frozen input.")
artifact("callback-context-diff", "inputs/necessary-context-12.8.diff", "derivation", "Actual necessary-context diff, empty because exact bytes match.")
artifact("callback-prior-opaque", "prior/9.9-prior-complete-report.opaque.json", "source_snapshot", "Complete raw prior ownreport including issues/allfields, preserved as history only; not newscientific proof.")
artifact("callback-prior-preservation", "prior/opaque-preservation-receipt.json", "source_snapshot", "True prior archive pathname/fullSHA/bytes and preservation method.")
artifact("callback-code", "code/callback-inspect.py", "code", "Actual bounded diff/Unicode/exact fingerprint inspectioncode, including perclaim scope checks.")
artifact("callback-runner-code", "code/run-callback.py", "code", "Actual runner that saved callback command/stdout/stderr/env/hash/status.")
artifact("callback-report-writer", "code/write-current-review.py", "code", "Sameowner report update code; scientificbasis is actualinspection and unchanged originalsource/CPUproof.")
artifact("callback-actual-inspection", "execution/actual-inspection.json", "derivation", "This callback's actual reading, perclaim meaning/scope decision, 37exact checks, unchangedfence/context and honestreuse limits.")
artifact("callback-execution-receipt", "execution/callback-execution-receipt.json", "execution", "Real currentcallback inspection command/env/hashes/status; verifies reuse eligibility, no model/tensorexecution rerun.",
         {"command":" ".join(receipt["command_argv"]),"result":"exit_code=0; bounded30s; all37 fingerprint assertions and exact orthography/same-term diff assertions passed", "environment":receipt["environment"]})
artifact("callback-stdout", "execution/callback-inspect.stdout.txt", "source_snapshot", "Real callbackstdout reporting actual byte/diff/fingerprint results.")
artifact("callback-stderr", "execution/callback-inspect.stderr.txt", "source_snapshot", "Real callbackstderr, empty.")

report["sources"].append({"id":"callback-execution","kind":"execution","title":"Sameoriginalowner narrow currentversion fingerprint/scope callback","verified":True,"artifact_id":"callback-execution-receipt"})
old_first = 316
new_first = inspection["current_section_first_line"]
shift = new_first - old_first
for claim in report["claims"]:
    loc = claim["location"]
    if loc.startswith("09.md:"):
        loc = "course/chapters/09.md:" + re.sub(r"\d+",lambda m:str(int(m[0])+shift),loc.split(":",1)[1])
    claim["location"] = loc
    claim["artifact_ids"].extend(["callback-actual-inspection", "callback-execution-receipt"])
    claim["callback_scope_check"] = inspection["current_claim_scope_checks"][claim["id"]]
    if "verification" in claim:
        claim["verification"]["details"] += " Sameoriginalowner2026-10-06 callback: original actual execution is retained, not newly rerun; identicalfence, unchanged primarysources and allowned proof fingerprints personally checked. See callback-actual-inspection and callback-execution-receipt for actual current check."
report["source_sha256"] = inspection["current_source_sha256"]
report["verdict"] = "pass"
report["reviewed_at"] = "2026-10-06"
report["read_scope_artifact_id"] = "callback-actual-inspection"
report["original_independent_review_date"] = "2026-10-05"
report["reviewer_context_note"] = "Same original fresh independentowner continuing its own narrowcallback; no newfork or claim of a newfresh read."
report["callback"] = {"kind":inspection["callback_kind"],"prior_opaque_path":str(prior_path.relative_to(ROOT)),"prior_opaque_sha256":sha(prior_path),
                      "prior_frozen_source_sha256":inspection["prior_frozen_section_sha256"],"current_source_sha256":inspection["current_source_sha256"],
                      "substantively_changed_claim_ids":[],"actual_inspection_artifact_id":"callback-actual-inspection","execution_receipt_artifact_id":"callback-execution-receipt",
                      "scope":inspection["reuse_decision"],"reuse_limits":inspection["reuse_limits"],"historical_full_file_fingerprint_policy":inspection["source_file_fingerprint_policy"]}
report["review_history"] = [{"kind":"complete priorownerraw report preserved","report_path":str(prior_path.relative_to(ROOT)),"report_sha256":sha(prior_path),
                            "source_sha256":inspection["prior_frozen_section_sha256"],"preserved_complete_issues_and_fields":True}]
report["checks"]["factual_accuracy"]["details"] = "Same originalowner read currentfull9.9 and exactdiff. Only traditionalspellings and probabilityterm synonym changed, without any scientificclaim change. Eightclaims retain exactly the supported scope; no unresolvedsubstantive question. Originalauthority/CPUverification reused after exactfingerprint checks; currentactualinspection is separately recorded."
report["checks"]["numeric_verification"]["details"] += " Currentcallback did not rerun tensor/model arithmetic: unchangedfence and scientificnumbers plus allpriorowned input/code/stdout fingerprints match exactly, so existing executedchecks are explicitly reused."
report["checks"]["source_verification"]["details"] += " 2026-10-06 callback verified37exact fingerprints and currentclaim scope, without refetching/rereading unchanged originalpapers/API sources; originalaccess dates remain historical."
report["checks"]["figure_consistency"]["details"] = "Current9.9 has no directfigure or changedvisualclaim. Necessary12.8 body/SVG are exact-fingerprint identical; prioractual640/390 render/view retained as historicalproof. No current9.9 page rendered in this narrowcallback; no currentvisualverification claimed."
report["checks"]["limitations"]["details"] += " Callbacklimit: only currentrawtext/diff/scope/dependency hashes were newly inspected; prior scientificCPU and visualresults were explicitly reused, not rerun."
(ROOT / "docs/technical-reviews/9.9.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"verdict":report["verdict"],"report_sha256":sha(ROOT / "docs/technical-reviews/9.9.json"),"source_sha256":report["source_sha256"],
                  "actual_inspection_artifact_id":"callback-actual-inspection","execution_receipt_artifact_id":"callback-execution-receipt"},ensure_ascii=False,indent=2))
