"""Write this owner's new canonical only after the actual V4 inspection and figure view."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
REL = A.relative_to(ROOT).as_posix()
OWNER = "/root/phase4_factual_coordinator/factual_13_16"
def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
prior = A / "own-prior-canonical-a3eb7278.json"
prior_raw = prior.read_bytes()
assert sha(prior_raw) == "a3eb727878ab8c6dd9d707cf488f14fd7e9b08322c87e467a141ab652463b44e"
target = ROOT / "docs/technical-reviews/13.16.json"
assert target.read_bytes() == prior_raw
report = json.loads(prior_raw)
assert report["reviewer_task"] == OWNER
facts = json.loads((A / "inspection-facts.json").read_bytes())
assert facts["reviewer_task"] == OWNER
assert facts["necessary_context_figure"]["personally_viewed"] is True
assert facts["necessary_context_figure"]["render_exit_code"] == 0
assert (A / "inspection-stderr.txt").read_bytes() == b""
assert facts["sections"]["13.16"]["sha256"] == report["source_sha256"]
artifact_id = "v4-context-inspection-20261006"
inspect_command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + REL + "/inspect_context.py > " + REL + "/inspection-stdout.txt 2> " + REL + "/inspection-stderr.txt"
render_command = "inkscape " + REL + "/model-roles-current.svg --export-type=png --export-filename=" + REL + "/model-roles-current.png --export-width=640 > " + REL + "/figure-render-stdout.txt 2> " + REL + "/figure-render-stderr.txt"
write_command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + REL + "/write_canonical.py"
commands = {"cwd": str(ROOT), "shell": "bash login:false", "actions": [{"command": inspect_command, "exit_code": 0, "code": REL + "/inspect_context.py", "stdout": REL + "/inspection-stdout.txt", "stderr": REL + "/inspection-stderr.txt", "result": "Current source/context fingerprints, own unchanged evidence/method fingerprints and actual figure-view facts recorded; no model executed."}, {"command": render_command, "exit_code": 0, "source": REL + "/model-roles-current.svg", "output": REL + "/model-roles-current.png", "personally_viewed_with": "tools.view_image", "result": "Actually rendered640×1010 PNG; five roles and reference PPO-start timing readable and consistent."}, {"command": write_command, "result_contract": "Writes new canonical and asserts reviewer/source/new artifact identity; checker must be a later separate command after success."}]}
dump(A / "commands.json", commands)
receipt = {"schema_version": 1, "kind": "same_original_owner_v4_context_reinspection", "reviewer_task": OWNER, "recorded_on": datetime.now(UTC).isoformat(), "verdict": "pass", "source": "course/chapters/13.md#13.16", "source_sha256": report["source_sha256"], "canonical_artifact_id": artifact_id, "own_prior_opaque": {"path": prior.relative_to(ROOT).as_posix(), "sha256": sha(prior_raw), "exact_verbatim_copy_before_inspection": True}, "inspection_facts": {"path": REL + "/inspection-facts.json", "sha256": sha((A / "inspection-facts.json").read_bytes())}, "own_support": {"path": REL + "/own-support.md", "sha256": sha((A / "own-support.md").read_bytes())}, "actual_scope": facts["scope"], "source_and_context": facts["sections"], "necessary_context_figure": facts["necessary_context_figure"], "original_evidence_reused": {"registered_artifact_count": len(facts["unchanged_registered_prior_artifacts"]), "all_exact_hashes_unchanged": True, "current_method_source_hashes_unchanged": True, "support_scope": "Original fence/exercise/bounded counterexample CPU evidence, saved raw measurement audit, personally verified original exact-version papers/docs and prior true renders remain valid. Current reference-after-demonstration timing independently compared to original code at278–299 and original InstructGPT sec3.1/3.5 eq2; no new model performance generated."}, "history_meaning": "Original frozen whole-file hashes and prior reports remain historical snapshot facts; new frozen whole input is separately labeled. Current read_scope covers actual V4 reads13.16/13.12/13.14; earlier13.13/13.15 readings remain only in preserved history.", "claim_support_assessment": {"constraints-do-not-correct-reward": "Changed context narrows reference to the demonstrated PPO starting policy, matching original copy/freeze code and InstructGPT SFT reference. It does not add truth/format acceptance to the reward, so original claim and bounded proof remain valid.", "finite-task-contract": "Independent finite full-request rule and prewritten candidates unchanged, compatible with correctly timed reference.", "independent-task-evaluation": "Independent acceptance still differs from learned score; context change does not alter data or evaluators.", "other_seven_claims": "Own section bytes and registered supports unchanged; no changed mechanism, numerator, denominator, approximation or limitation."}, "limitations": ["Same original owner follow-up, not new fresh reviewer.", "Current required SVG render/view performed; no live-page/site-layout check this round.", "No unrelated chapter reading, model CPU experiment, full recipe, model/weight/data download, paper retrieval, GPU/train/source edit/commit."], "unresolved_questions": [], "inspection_environment": facts["environment"], "inspection_command": inspect_command, "inspection_exit_code": 0, "render_command": render_command, "render_exit_code": 0}
receipt_path = A / "v4-inspection-receipt.json"
dump(receipt_path, receipt)
for path in sorted(A.iterdir()):
    assert path.is_file() and not path.is_symlink()
    identifier = artifact_id if path == receipt_path else "v4-13_16-" + path.name.replace(".", "-")
    kind = "figure_render" if path.suffix == ".png" else "code" if path.suffix == ".py" else "derivation" if path.name in {"v4-inspection-receipt.json", "inspection-facts.json", "own-support.md", "context-diff-from-own-previous.patch"} else "source_snapshot"
    item = {"id": identifier, "kind": kind, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes()), "description": "13.16 same-owner V4 necessary-context inspection: " + path.name}
    if path.name == "inspection-stdout.txt":
        item.update(id="v4-context-inspection-execution", kind="execution", command=inspect_command, result="exit0; three actual current section reads, unchanged original evidence/method hash verification and current required figure render/view recorded; no model execution", environment=facts["environment"])
    if path.name == "figure-render-stdout.txt":
        item.update(id="v4-context-figure-render-execution", kind="execution", command=render_command, result="exit0; actual current context SVG rendered and resulting PNG personally viewed", environment=facts["environment"])
    report["artifacts"].append(item)
figure_path = facts["necessary_context_figure"]["source"]
report["figure_sha256"][figure_path] = facts["necessary_context_figure"]["sha256"]
report["figure_scope"] = {figure_path: "Necessary13.14 role/timing context dependency, not an image referenced directly by13.16; inspected/rendered/viewed this round."}
report["read_scope"]["course/chapters/13.md"] = ["13.16 current complete incl details personally reread; bytes unchanged; " + facts["sections"]["13.16"]["current_locator"], "13.12 current complete incl details personally reread and compared against own prior context; reference timing/new original citation assessed; " + facts["sections"]["13.12"]["current_locator"], "13.14 current complete incl details personally reread and compared; necessary role SVG actually rendered/viewed; " + facts["sections"]["13.14"]["current_locator"]]
report["last_reinspected_on"] = "2026-10-06"
report["reinspection_context"] = "same_original_owner"
report.setdefault("reinspections", []).append({"artifact_id": artifact_id, "receipt_path": receipt_path.relative_to(ROOT).as_posix(), "receipt_sha256": sha(receipt_path.read_bytes()), "verdict": "pass", "context_sections": ["13.12", "13.14"], "context_sha256": {key: facts["sections"][key]["sha256"] for key in ["13.12", "13.14"]}, "context_figure_sha256": {figure_path: facts["necessary_context_figure"]["sha256"]}, "history_path": prior.relative_to(ROOT).as_posix(), "history_sha256": sha(prior_raw), "scope": facts["scope"]})
report["v4_frozen_input"] = facts["frozen_whole_input"]
for claim in report["claims"]:
    if claim["id"] in {"constraints-do-not-correct-reward", "finite-task-contract", "independent-task-evaluation"}: claim["artifact_ids"].append(artifact_id)
    if claim["id"] == "constraints-do-not-correct-reward":
        claim["evidence"].append({"source_id": "instructgpt", "locator": "sec3.1 three-step process;sec3.5 eq2; preserved original text302–319,483–509 narrowly reread in V4", "supports": "Current necessary context correctly places the fixed reference at demonstrated/SFT policy for PPO; this remains regularization rather than a replacement task truth criterion."})
report["checks"]["figure_consistency"] = {"status": "pass", "details": "13.16 itself has no referenced figure. Its necessary current13.14 model-role SVG dependency was copied with exact SHA d7589ab9e44118540111e8f331b555d4d6c4db3f548e7d373bf7d40750eba406, actually rendered with Inkscape and personally viewed. Five role labels/input-output/update timing agree with current context and original implementation; reference is PPO starting policy fixed during PPO. This is a current SVG render/view, not a new live-page layout check; original true13.16 table renders are unchanged.", "claim_ids": ["constraints-do-not-correct-reward", "finite-task-contract"]}
for check in ["factual_accuracy", "source_verification", "limitations"]:
    report["checks"][check]["details"] += " V4 same-owner actual current13.16/13.12/13.14 reread, own input diff and all68 prior evidence/current method exact fingerprint checks complete. Reference-after-demonstration PPO timing independently supported by original copy/freeze code278–299 and original InstructGPT sec3.1/3.5 eq2 narrow reread. Necessary current context figure actually rendered/viewed; no new model CPU/recipe/GPU/train/paper fetch. v4-context-inspection-20261006 records support assessment, exact history and real scope."
report["verdict"] = "pass"
assert receipt["unresolved_questions"] == []
dump(target, report)
new = json.loads(target.read_bytes())
assert new["reviewer_task"] == OWNER and new["source_sha256"] == report["source_sha256"]
assert new["reinspections"][-1]["artifact_id"] == artifact_id
print(json.dumps({"verdict": "pass", "new_canonical_written_and_asserted": True, "report_path": target.relative_to(ROOT).as_posix(), "report_sha256": sha(target.read_bytes()), "source_sha256": report["source_sha256"], "inspection_artifact_id": artifact_id, "inspection_path": receipt_path.relative_to(ROOT).as_posix(), "inspection_sha256": sha(receipt_path.read_bytes()), "prior_opaque_path": prior.relative_to(ROOT).as_posix(), "prior_opaque_sha256": sha(prior_raw), "actual_scope": facts["scope"], "unresolved_questions": []}, ensure_ascii=False))
