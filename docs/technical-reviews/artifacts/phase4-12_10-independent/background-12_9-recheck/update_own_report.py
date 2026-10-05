import hashlib,json
from datetime import datetime,UTC
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/"docs/technical-reviews/artifacts/phase4-12_10-independent";OUT=BASE/"background-12_9-recheck"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=OUT/"initial-pass-report-opaque.json";assert sha(prior)=="72d4afc4a4e27e5bce11dc7e057fe8b374547cc1c273cba3ed0b625c65362486"
d=json.loads(prior.read_bytes());assert d["reviewer_task"]=="/root/phase4_factual_coordinator/factual_12_10"
inputs=json.loads((OUT/"input-receipt.json").read_bytes())
for item in inputs["current_inputs"]:
 item.pop("whole_chapter_at_this_read_sha256",None);item.pop("whole_chapter_sha_meaning",None)
 item["byte_policy"]="Complete current section UTF-8 bytes, no normalization"
(OUT/"input-receipt.json").write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+"\n")
checks=json.loads((OUT/"retained-evidence-hash-check.json").read_bytes())
assert checks["repository_code_source_hashes_still_match"] and all(a["matches_initial_report"] for a in checks["artifact_hash_checks"])
for a in d["artifacts"]:assert sha(ROOT/a["path"])==a["sha256"]
for s in d["sources"]:
 if s["kind"]=="repository_code":assert sha(ROOT/s["path"])==s["sha256"]
import sys;sys.path.insert(0,str(ROOT/"docs/review-tools"));from section_facts import original_section
for lesson,expected in [("12.9","0930ed04d140c96d076a3ca8af5d3334f20d684293d2f6eaff8f04830d991149"),("12.10",d["source_sha256"])]:
 body,_,_=original_section(ROOT/"course/chapters/12.md",lesson);assert hashlib.sha256(body).hexdigest()==expected
assert sha(BASE/"inputs/course/chapters/12.md")==d["inspection_record"]["frozen_full_chapter"]["sha256"]
mapping=[
 {"claim_id":"constant-baseline","depends_on_changed_error_attribution":False,"assessment":"Fixed baseline convention and class-count arithmetic are supported directly by unchanged official baseline/accuracy originals; modified prerequisite error description is not evidence for this claim."},
 {"claim_id":"threshold-counts","depends_on_changed_error_attribution":False,"assessment":"Depends only on strict >300 Hz rule unchanged in both current sections and the unchanged 12.10 list/fence/CPU variant. Does not attribute 290/300 failures to duration."},
 {"claim_id":"historical-controls","depends_on_changed_error_attribution":False,"assessment":"Depends directly on saved raw records/samples and original control code. The changed prerequisite's 11/14 remains the same; errors are rows 4,6,7 at 290,300,300 Hz. The two settings couple amplitude and duration; initial claim already described two amplitude/duration settings, with no separate-cause assertion."},
 {"claim_id":"intervention-meaning","depends_on_changed_error_attribution":False,"assessment":"Original-vs-donor label metrics and whole-waveform intervention remain unchanged. Evidence supports this model's response to changing audio; neither prior nor current 12.10/own claim infers independent amplitude or duration causality. Current background limit is consistent."},
 {"claim_id":"family-split","depends_on_changed_error_attribution":False,"assessment":"Related variants kept together is a split policy, not factorial identification. The paired amplitude/duration variants retain the same frequency-family labels and no split overlap. Background correction preserves this rationale."},
 {"claim_id":"original-python","depends_on_changed_error_attribution":False,"assessment":"12.10's seven-line scalar arithmetic fence is byte-identical and standalone; no dependency on the 12.9 gradient demo or changed description of errors."}
]
receipt={"artifact_id":"background-12_9-recheck-receipt","reviewer_task":d["reviewer_task"],"review_type":"Same independent reviewer recheck after prerequisite source change","completed_at_utc":datetime.now(UTC).isoformat(),"actual_read":[{"source":"course/chapters/12.md#12.9","locator":"Complete section, source lines 265-307; changed paragraph at source line 295","snapshot_path":(OUT/"current-12.9.md").relative_to(ROOT).as_posix(),"sha256":sha(OUT/"current-12.9.md")},{"source":"course/chapters/12.md#12.10","locator":"Complete section, source lines 308 through end of section","snapshot_path":(OUT/"current-12.10.md").relative_to(ROOT).as_posix(),"sha256":sha(OUT/"current-12.10.md")}],"previous_pass_report":{"path":prior.relative_to(ROOT).as_posix(),"sha256":sha(prior),"copied_before_substantive_recheck":True,"not_used_as_external_authority":True},"retained_initial_frozen_input":d["inspection_record"]["frozen_full_chapter"],"source_change_scope":"Only prerequisite 12.9 historical-error description: boundary/duration attribution replaced with three error frequencies and coupled amplitude/duration caveat. Current 12.10 bytes are unchanged.","claim_dependency_review":mapping,"raw_pointer_reread":["/results/test/samples","/results/data/splits/test/records"],"recorded_error_rows":checks["recorded_error_rows"],"recorded_amplitude_duration_pairs":checks["recorded_amplitude_duration_pairs"],"retained_evidence":"All initial formal artifact hashes and repository code/raw JSON source hashes rechecked equal; original CPU, official-source and screenshot evidence remains applicable. No unrelated rerun.","no_training_or_model_execution":True,"no_other_reviewer_or_author_summary_read":True,"verdict":"pass","scope":"Current own 12.10 only; this receipt does not independently certify the complete 12.9 lesson."}
(OUT/"personal-recheck-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
new_arts=[("background-12_9-recheck-receipt","personal-recheck-receipt.json","source_snapshot","Personal complete current-background and own-section reread with all six dependency assessments"),("background-12_9-current-source","current-12.9.md","source_snapshot","Exact complete current prerequisite source read by this reviewer"),("own12_10-recheck-source","current-12.10.md","source_snapshot","Exact unchanged complete own section reread by this reviewer"),("background-recheck-inputs","input-receipt.json","source_snapshot","Opaque initial PASS backup and actual current-section source hashes/locators"),("background-retained-evidence","retained-evidence-hash-check.json","source_snapshot","Preserved evidence version checks and narrow reread of original wrong rows and coupled settings"),("background-own-source-diff","own-prerequisite-source-diff.txt","source_snapshot","Diff of this reviewer's initial prerequisite source bytes versus current prerequisite source"),("background-recheck-code","update_own_report.py","code","Exact code for preserving and updating this reviewer's report after personal semantic recheck")]
for ident,file,kind,description in new_arts:
 path=OUT/file;d["artifacts"].append({"id":ident,"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"kind":kind,"description":description})
for a in d["artifacts"]:
 if a["id"]=="prereq12_9":a["description"]="Original first-read prerequisite 12.9 bytes, retained as historical frozen input; current background reread is the separate background-12_9-current-source artifact"
 if a["id"]=="inspection":a["description"]="Initial independent inspection record, retained unchanged; subsequent personal background recheck is separately recorded"
 if a["id"]=="read-inventory":a["description"]="Initial actual code/source read ranges; current background reread has its separate receipt"
for c in d["claims"]:
 c["artifact_ids"].append("background-12_9-recheck-receipt")
 if c["id"]=="historical-controls":c["scope"]+=" Amplitude and duration are coupled settings; no independent amplitude/duration effect is inferred."
 if c["id"]=="intervention-meaning":c["scope"]+=" No independent duration or amplitude causal effect is inferred from coupled settings."
 if c["id"]=="family-split":c["scope"]+=" The two amplitude/duration conditions are paired settings, not independently controlled factors."
d["checks"]["factual_accuracy"]["details"]+=" Personally reread complete current 12.9 and unchanged 12.10 after prerequisite edit; all six dependency assessments preserve support, documented in background-12_9-recheck-receipt."
d["checks"]["limitations"]["details"]+=" Current prerequisite makes amplitude/duration confounding explicit; own claims already avoid independent-factor attribution."
d["review_history"]=[{"event":"initial_independent_pass","report_path":prior.relative_to(ROOT).as_posix(),"report_sha256":sha(prior),"own_source_sha256":d["source_sha256"],"initial_frozen_full_chapter":d["inspection_record"]["frozen_full_chapter"]},{"event":"personal_background_recheck","artifact_id":"background-12_9-recheck-receipt","receipt_path":(OUT/"personal-recheck-receipt.json").relative_to(ROOT).as_posix(),"receipt_sha256":sha(OUT/"personal-recheck-receipt.json"),"current_background_section_sha256":sha(OUT/"current-12.9.md"),"own_source_sha256_unchanged":d["source_sha256"],"verdict":"pass"}]
d["inspection_record"]["background_rechecks"]=[{"artifact_id":"background-12_9-recheck-receipt","receipt_path":(OUT/"personal-recheck-receipt.json").relative_to(ROOT).as_posix(),"sha256":sha(OUT/"personal-recheck-receipt.json")}]
raw=json.dumps(d,ensure_ascii=False,indent=2)+"\n"
(ROOT/"docs/technical-reviews/12.10.json").write_text(raw);(OUT/"rechecked-full-report.json").write_text(raw)
print(json.dumps({"formal_artifact_id":"background-12_9-recheck-receipt","receipt_sha256":sha(OUT/"personal-recheck-receipt.json"),"prior_report_sha256":sha(prior),"new_report_sha256":sha(ROOT/"docs/technical-reviews/12.10.json"),"own_source_sha256":d["source_sha256"],"verdict":d["verdict"]},ensure_ascii=False))
