"""Write the same technical owner's current report with truthful reuse and new receipts."""
from datetime import datetime,UTC
from hashlib import sha256
from pathlib import Path
import json
import platform
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parents[3]
PREFIX=OUT.relative_to(ROOT).as_posix()
digest=lambda path:sha256(path.read_bytes()).hexdigest()
save=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
initial=json.loads((OUT/"input-and-render-receipt.json").read_text())
prior=ROOT/initial["prior_history_path"]
assert digest(prior)==initial["prior_history_sha256"]
report=json.loads(prior.read_text())
for a in report["artifacts"]: assert digest(ROOT/a["path"])==a["sha256"]
raw=(OUT/"current-section.md").read_bytes()
assert digest(OUT/"current-section.md")==initial["source_sha256"]
expected=(ROOT/"course/chapters/06.md").read_bytes()
assert raw in expected
browser=json.loads((OUT/"browser-render-receipt.json").read_text())
assert browser["exit_code"]==0 and len(browser["renders"])==2
receipt={
 "schema_version":1,"kind":"actual_original_owner_reinspection","reviewer_task":report["reviewer_task"],
 "observed_on":datetime.now(UTC).isoformat(),"source":report["source"],"source_sha256":initial["source_sha256"],
 "prior_source_sha256":initial["prior_source_sha256"],"prior_history_path":initial["prior_history_path"],"prior_history_sha256":initial["prior_history_sha256"],
 "source_change":initial["source_change"],"substantive_claim_changes":[],
 "actual_read_scope":"Complete current6.2 including fence/caption; current SVG; own frozen input/diff; current factual-reviewer instructions; own original probe-results and frozen paper§3.2 pp1717–1718 and apply_bpe.py276–329. No new prerequisites were needed for spelling-only changes.",
 "supports_rechecked":"Seven original material claims still match current text. Exercise independently recalculated: ac4 new occurrences and ca3, ac5 overall>ab3; old outputs[a,b,a,b]/[a,b,ac], token lengths4/3. Existing API and algorithm evidence remains applicable because code/claims/inputs are unchanged.",
 "evidence_reuse":{"explicit":True,"verified_original_artifacts":initial["reused_evidence_hash_checks"],
                  "not_new_runs":"Original fence and2044CPU variations were not rerun. Original official sources were not refetched. Their exact permanent snapshots, outputs and fingerprints are reused after actual scope/support review."},
 "viewed_current_renders":["current-figure-390.png","preview-desktop.png","preview-mobile.png"],
 "visual_findings":"Personally viewed all3new PNGs. Labels3/2/1, row boundaries, merge arrow, output2/3rows andab=one-token annotation agree. Served exercise contains new spelling. Mobile code/output uses horizontal overflow; only initial scroll position is pictured, not every column simultaneously.",
 "environment":{"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"device":"CPU","playwright":browser["environment"]["playwright"],"chromium":browser["environment"]["chromium"],"inkscape":"1.4"},
 "commands":[".venv/bin/python "+PREFIX+"/inspect-current.py","timeout 50s .venv/bin/python "+PREFIX+"/render-preview.py",".venv/bin/python "+PREFIX+"/write-reinspection.py"],
 "command_results":"Input/hash checks and SVG export exit0; preview HTTP200; browser driver exit0; writer assertions passed before canonical save. Source requests were local preview only. Original CPU numerical tests were reused, not rerun.",
 "verdict":"pass","unresolved_questions":[],"source_or_figure_edits_by_reviewer":False,"training_or_model_download_or_gpu_run":False}
save(OUT/"reinspection-receipt.json",receipt)
environment={k:str(v) for k,v in receipt["environment"].items()}
def add(identifier,name,kind,description,command=None,result=None):
 a={"id":identifier,"path":PREFIX+"/"+name,"sha256":digest(OUT/name),"kind":kind,"description":description}
 if kind=="execution": a.update(command=command,result=result,environment=environment)
 report["artifacts"].append(a)
add("reinspection-current-receipt","reinspection-receipt.json","execution","Same original-owner actual reading, support review, verified reuse, current-render viewing and truthful execution scope.",
 ".venv/bin/python "+PREFIX+"/inspect-current.py; timeout 50s .venv/bin/python "+PREFIX+"/render-preview.py; .venv/bin/python "+PREFIX+"/write-reinspection.py",
 "All input/reuse/source/fence/SVG checks pass; new SVG and2browser renders completed and were personally viewed; no substantive claim changes or unresolved questions.")
for identifier,name,kind,description in [
 ("reinspection-current-section","current-section.md","source_snapshot","Actual current6.2 UTF-8 bytes without normalization."),
 ("reinspection-source-diff","source-diff.txt","derivation","Actual comparison with own frozen original: two spelling changes only."),
 ("reinspection-input-receipt","input-and-render-receipt.json","source_snapshot","Current fence/SVG and36own prior artifact fingerprints, prior history and actual render/HTTPreceipt."),
 ("reinspection-current-fence","current-fence.py","code","Current fence bytes, verified identical to original executed fence."),
 ("reinspection-current-svg","current-figure.svg","source_snapshot","Current SVG bytes, verified identical to original checked figure."),
 ("reinspection-current-render","current-figure-390.png","figure_render","New render of actual current SVG, personally viewed."),
 ("reinspection-preview-desktop","preview-desktop.png","figure_render","Actual current preview desktop1280×800 full-page screenshot, personally viewed."),
 ("reinspection-preview-mobile","preview-mobile.png","figure_render","Actual current preview mobile390×844 full-page screenshot, personally viewed; code/output shown at initial horizontal position."),
 ("reinspection-preview-html","current-preview.html","source_snapshot","Actual served current preview HTTP200HTML, local ephemeral preview not an external authority."),
 ("reinspection-browser-receipt","browser-render-receipt.json","source_snapshot","Playwright/Chromium versions, viewport sizes, HTTPstatus and screenshot hashes."),
 ("reinspection-notes","reinspection-notes.md","derivation","Own actual substantive change assessment and continued-source-support notes."),
 ("reinspection-driver","inspect-current.py","code","Executed current-input/hash/freeze/render driver, bounded subprocess and local request."),
 ("reinspection-browser-driver","render-preview.py","code","Executed browser render script, bounded launch/navigation/screenshots."),
 ("reinspection-writer","write-reinspection.py","code","Same original-owner report/receipt writer with reuse and version assertions.")]: add(identifier,name,kind,description)
report["source_sha256"]=initial["source_sha256"]
report["reviewed_on"]=datetime.now(UTC).date().isoformat()
report["scope"]="Same original technical owner personally reread full current6.2 and SVG, compared own frozen input, found spelling-only exercise changes, checked36own original evidence fingerprints and actual source support, recalculated exercise and viewed newly rendered current SVG plus actual desktop/mobile preview. Original fence/probes/sources explicitly reused, not rerun/refetched. No new prerequisite reading or GPU/training/model/data work."
report["reinspection"]={"artifact_id":"reinspection-current-receipt","receipt_path":PREFIX+"/reinspection-receipt.json","receipt_sha256":digest(OUT/"reinspection-receipt.json"),"prior_history_path":initial["prior_history_path"],"prior_history_sha256":initial["prior_history_sha256"],"actual_scope":receipt["actual_read_scope"],"substantive_claim_changes":[],"unresolved_questions":[]}
for claim in report["claims"]:
 claim["artifact_ids"].append("reinspection-current-receipt")
 claim["current_reinspection"]="Personally rechecked current statement and scope; unchanged substantive claim. Its exact original evidence fingerprints match. Original runs/official source inspection remain applicable and are explicitly reused, not presented as new runs/access."
report["checks"]["factual_accuracy"]["details"]+=" Current original-owner reinspection: only two spelling changes; all seven material claims still supported after actual rereading and scope checks."
report["checks"]["numeric_verification"]["details"]+=" Exercise reinspection recomputed ac4new/5overall versusab3 and old sequences[a,b,a,b]/[a,b,ac]; original CPU evidence fingerprints verified and reused without rerun."
report["checks"]["figure_consistency"]["details"]="Personally reread current SVG, verified original hash, rendered actual currentSVG390px and viewed it; also rendered/viewed actual current preview at desktop1280×800 and mobile390×844 with Playwright1.63/Chromium151. Diagram boundaries, counts, arrow and output rows match. Mobile code/output initially clips long lines via horizontal scrolling; no claim all columns fit simultaneously."
report["checks"]["source_verification"]["details"]+=" During current reinspection verified all36own registered artifact hashes and reread original publisher§3.2 plus pinned application code276–329; reuse recorded explicitly, no refetch or unrelated sources."
report["checks"]["limitations"]["details"]="Original one-step character toy and2044bounded CPU checks support their recorded merger scope only. Full trainer boundaries/ties/stops may differ. Pair-free top-level inputs remain outside toy. Current desktop/mobile page and diagram were actually viewed; mobile long code/output lines require horizontal scrolling. Old measurements/sources explicitly reused after support/hash checks, with no new training/model/data/GPU work."
canonical=ROOT/"docs/technical-reviews/6.2.json"
save(canonical,report)
print(json.dumps({"verdict":report["verdict"],"report_path":str(canonical.relative_to(ROOT)),"report_sha256":digest(canonical),"prior_history_path":initial["prior_history_path"],"prior_history_sha256":initial["prior_history_sha256"],"artifact_id":"reinspection-current-receipt","receipt_path":PREFIX+"/reinspection-receipt.json","receipt_sha256":digest(OUT/"reinspection-receipt.json")},ensure_ascii=False,indent=2))
