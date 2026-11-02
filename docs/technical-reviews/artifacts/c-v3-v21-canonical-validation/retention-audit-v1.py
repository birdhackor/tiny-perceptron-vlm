"""Post-intake immutable retention and product-review identity audit. No neural runs."""
from pathlib import Path
from datetime import datetime, UTC
from collections import Counter
import difflib, hashlib, json, platform, sys, zipfile, importlib.util

ROOT=Path("/tmp/tpb").resolve()
ART=ROOT/"docs/technical-reviews/artifacts/c-v3-v21-canonical-validation"
ACTOR="/root/c_v3_technical_coord/canonical_validation"
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:(ROOT/p).read_bytes()
ref=lambda p:{"path":p,"sha256":sha(read(p))}
def obj(p):return json.loads(read(p))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
started=datetime.now(UTC).isoformat()
original_path="docs/course-revision-20261101-c-v3/technical-v21/canonical_validation/original-v1.json"
assert sha(read(original_path))=="921003bde79f37ec87dd678e83ea418300bd39ba8a85c3416ba90b0673560fe3"
intake_path="docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-validation-original-v1.json"
assert sha(read(intake_path))=="a4195008cbc4c97385c64ad80da87f3e7863c0717850594927f215f5e32a97b4"
intake=obj(intake_path)
assert datetime.fromisoformat(started)>datetime.fromisoformat(intake["received_at"])
map_path="docs/course-revision-20261101-c-v3/technical-coordination-v21/prior-evidence-reference-map-resume-v1.json"
mapping=obj(map_path)
prior_row=next(row for row in mapping["pages"] if row["page_id"]=="validation")
proof=prior_row["prior_technical_proof"]
for key in ("report","original_report","original_source_freeze"):
 assert sha(read(proof[key]["path"]))==proof[key]["sha256"]
prior=obj(proof["report"]["path"])
previous_freeze=obj(proof["original_source_freeze"]["path"])
old_page=next(p for p in previous_freeze["pages"] if p["page_id"]=="validation")
new_freeze=obj("docs/course-revision-20261101-c-v3/source-freeze-v21.json")
page=next(p for p in new_freeze["pages"] if p["page_id"]=="validation")
old=read(old_page["snapshot"]);new=read(page["snapshot"])
assert sha(old)==prior["source_sha256"]==old_page["source_sha256"]
assert sha(new)==page["source_sha256"] and new==read(page["source"])
a=old.decode().splitlines(keepends=True);b=new.decode().splitlines(keepends=True)
ops=[{"op":tag,"source18_lines":[i+1,j],"source21_lines":[k+1,l]} for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if tag!="equal"]
assert ops==[{"op":"replace","source18_lines":[42,42],"source21_lines":[42,44]}]
dependencies=[]
for s in prior["sources"]:
 if s["kind"]=="repository_code":
  actual=sha(read(s["path"]));assert actual==s["sha256"]
  dependencies.append({"id":s["id"],"path":s["path"],"sha256":actual})
assert len(dependencies)==24
prior_artifacts=[]
for entry in prior_row["prior_artifact_refs"]:
 assert sha(read(entry["path"]))==entry["sha256"]
 prior_artifacts.append(entry)
assert len(prior_artifacts)==1
old_raw=obj(prior_artifacts[0]["path"])
assert old_raw["canonical"]["source_sha256"]==old_page["source_sha256"]
assert old_raw["bc_raw_recalculation"]["argmax_disagreements_with_saved_answers"]==[]
assert old_raw["independent_auto_output_join"]["disagreements"]==[]
assert old_raw["choice_raw_target_and_argmax_check"]["disagreements"]==[]
assert old_raw["av_target_generator_check"]["gold_disagreements"]==[]
assert old_raw["independent_auto_output_join"]["totals"]=={"families":100,"route_correct":89,"gold_route_end_to_end_correct":66,"real_route_end_to_end_correct":55}
assert old_raw["omni_raw_recalculation"]["different_numeric_cells"]==43
assert old_raw["omni_raw_recalculation"]["integer_or_nonnumeric_difference"] is False
validator=module("validation_retained_native","scripts/check_technical_reviews.py")
native_errors=validator._validate(ROOT,ROOT/old_page["source"],"validation",old.decode(),prior,{prior["reviewer_task"]:["validation"]},set())
assert not native_errors,native_errors
retained_claims=[]
for c in prior["claims"]:
 scope=c["scope"]
 if c["id"]=="c36":
  scope="Retain only historical C-v2 failure and the default at that original batch. Current source21 explicitly says 當時; do not retain old C as the current default."
 if c["id"]=="c33":
  scope="Retain original omni batch inventory of 18 additions only, as followed by later C-v3; not a claim the present source21 increment totals 18."
 retained_claims.append({"prior_claim":c["id"],"statement":c["statement"],"location":c["location"],"original_evidence":[{"source_id":e["source_id"],"locator":e["locator"]} for e in c["evidence"]],"retained_scope":scope})
assert len(retained_claims)==68
kernel_path="docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json"
kernel=obj(kernel_path)
assert kernel["source_freeze"]["sha256"]==sha(read(kernel["source_freeze"]["path"]))
base=ROOT/"outputs/omni-c-v3-kernel-bundle-v21"
old_base=Path(kernel["retained_bundle"])
four={row["lesson"]:row for row in kernel["actual_executions"]}
kernel_receipts=[]
source_mismatches=[];missing_exec=[];errors=[]
retained_count=0
for p in sorted(base.glob("*/*.ipynb")):
 rel=p.relative_to(base);src=ROOT/"notebooks"/rel
 source_nb=json.loads(src.read_bytes());nb=json.loads(p.read_bytes())
 source_cells=[(c["cell_type"],c.get("source")) for c in source_nb["cells"]]
 output_cells=[(c["cell_type"],c.get("source")) for c in nb["cells"]]
 if source_cells!=output_cells:source_mismatches.append(str(rel))
 for index,c in enumerate(nb["cells"]):
  if c["cell_type"]=="code":
   if c.get("execution_count") is None:missing_exec.append([str(rel),index])
   for output in c.get("outputs",[]):
    if output.get("output_type")=="error":errors.append([str(rel),index])
 lesson=p.stem
 if lesson in four:
  row=four[lesson]
  assert sha(src.read_bytes())==row["source_sha256"]
  assert sha(p.read_bytes())==row["executed_sha256"]
  kernel_receipts.append({"lesson":lesson,"source_sha256":row["source_sha256"],"executed_sha256":row["executed_sha256"],"started_at":row["started_at"],"completed_at":row["completed_at"]})
 else:
  assert p.read_bytes()==(old_base/rel).read_bytes()
  retained_count+=1
assert retained_count==kernel["retained_current_count"]==328
assert len(kernel_receipts)==4 and len(list(base.glob("*/*.ipynb")))==kernel["bundle_count"]==332
assert not source_mismatches and not missing_exec and not errors
archive=Path("/tmp/tpc/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
if not archive.is_file():archive=Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
descriptor=obj("docs/decision-product/v3/public-release.json")
assert archive.stat().st_size==descriptor["bytes"] and sha(archive.read_bytes())==descriptor["sha256"]
members={}
with zipfile.ZipFile(archive) as z:
 def member(n):
  full="tiny-perceptron-c-v3/"+n
  assert z.namelist().count(full)==1
  raw=z.read(full);members[n]={"bytes":len(raw),"sha256":sha(raw)};return raw
 def load(n):return json.loads(member(n))
 idx=load("evidence/reviews/index.json")
 attachments=load("evidence/review-attachments-index.json")
 candidate=load("evidence/reviewed-candidate-bundle.json")
 summary=load("evidence/reviewed-qualification-summary.json")
 delivered_bundle=load("bundle/bundle.json")
 delivered_summary=load("qualification-summary.json")
 assert idx["candidate_bundle_sha256"]==sha(member("evidence/reviewed-candidate-bundle.json"))
 assert delivered_bundle["source_candidate_bundle_sha256"]==idx["candidate_bundle_sha256"]
 assert delivered_bundle["review_index_sha256"]==sha(member("evidence/reviews/index.json"))
 for k in ("config","files","final_opening_sha256","scope"):assert delivered_bundle[k]==candidate[k]
 for k in summary:
  if k!="status":assert delivered_summary[k]==summary[k],k
 assert delivered_bundle["config"]==obj("docs/decision-product/v3/metadata/qualified-bundle-manifest.json")["config"]
 assert sha(member("qualification-summary.json"))==descriptor["qualification_summary_sha256"]
 original_probe=obj("docs/technical-reviews/artifacts/c-v3-v21-canonical-validation/execution-v1.json")
 opening_hashes={Path(p["path"]).name:p["sha256"] for p in original_probe["implementation_identity"]}
 assert idx["implementation"]==opening_hashes
 for n,digest in idx["implementation"].items():
  assert sha(read("tiny_perceptron/decision_product/"+n))==digest
 for n,digest in candidate["files"].items():assert sha(member("bundle/"+n))==digest
 reports=[]
 for entry in idx["reviews"]:
  name="evidence/reviews/"+entry["path"];r=load(name)
  assert sha(member(name))==entry["sha256"]
  assert r["reviewer_task"]==r["reviewer"]==entry["actor"]
  assert r["implementation"]==idx["implementation"]
  assert r["readme_sha256"]==idx["readme_sha256"]
  assert r["candidate_bundle_sha256"]==idx["candidate_bundle_sha256"]
  assert datetime.fromisoformat(r["created_at"])<datetime.fromisoformat(idx["created_at"])
  scope={k:r[k] for k in ("scope","limits","limitations","empirical_claim_limit","binding_explanation","not_completed_or_claimed") if k in r}
  reports.append({"kind":entry["kind"],"reviewer_task":entry["actor"],"path":name,"sha256":entry["sha256"],"created_at":r["created_at"],"scope":scope,"status_not_used_for_new_technical_judgment":True})
 assert len(reports)==3 and len({r["reviewer_task"] for r in reports})==3
 assert {r["kind"] for r in reports}=={"readability","correctness","coherence"}
 assert len(attachments["files"])==57
 attachment_receipts=[]
 for r in attachments["files"]:
  raw=member(r["delivered_path"])
  assert len(raw)==r["bytes"] and sha(raw)==r["sha256"]
  assert r["actor"] in {report["reviewer_task"] for report in reports}
  attachment_receipts.append({"actor":r["actor"],"delivered_path":r["delivered_path"],"bytes":len(raw),"sha256":sha(raw),"reused_identical_payload":r["reused_identical_payload"]})
out={"schema":"canonical-validation-post-original-retention-v1","reviewer_task":ACTOR,"started_at":started,"completed_at":datetime.now(UTC).isoformat(),"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"device":"cpu; byte/JSON inspection only"},"physical_root":str(ROOT),"original":ref(original_path),"original_intake":ref(intake_path),"original_received_at":intake["received_at"],"prior_map":ref(map_path),
"source18_retention":{"proof":proof,"source_sha256":old_page["source_sha256"],"delta":ops,"repository_dependencies":dependencies,"prior_artifacts":prior_artifacts,"prior_native_integrity_errors":native_errors,"original_raw_record_scope":old_raw["scope"],"original_raw_record_environment":old_raw["environment"],"original_raw_record_created_at":old_raw["created_at"],"original_raw_record_completed_at":old_raw["completed_at"],"retained_claims":retained_claims,"retention_limit":"Immutable unchanged claims retain original source, raw artifact and limitations; no old pass is a new technical experiment. No old training/research/logit recomputation or NN inference executed. Historical C-v2 default is explicitly historical, not current."},
"kernel_source21_retention":{"source_receipt":ref(kernel_path),"bundle":str(base),"retained_original_bundle":str(old_base),"retained_identical_executed_bytes":retained_count,"actual_four":kernel_receipts,"source_and_output_cell_mismatches":source_mismatches,"missing_execution_counts":missing_exec,"error_outputs":errors,"total":332,"scope":"Inspect four actual refreshed command receipts/source/outputs and retain328 byte-identical outputs; no notebooks or model inference rerun. Kernel isolation follows unchanged check_notebooks.py."},
"product_reviews":{"archive":{"path":str(archive),"bytes":archive.stat().st_size,"sha256":descriptor["sha256"]},"index_created_at":idx["created_at"],"reports":reports,"attachment_rows":57,"unique_delivered_paths":len({r["delivered_path"] for r in attachment_receipts}),"attachment_rows_per_actor":dict(Counter(r["actor"] for r in attachment_receipts)),"attachment_receipts":attachment_receipts,"members_read":members,"identity":{"implementation_count":len(idx["implementation"]),"candidate_config_matches_delivery":True,"candidate_weights_tokenizer_policy_exact":True,"reviewed_final_metrics_and_scope_unchanged":True,"candidate_to_qualified_manifest_changes":"Metadata status and post-review index/source-candidate references only; config/files/final-opening/scope identical."},"limits":"Three actual distinct named report records and copied attachments are verified against bytes, dates and scope. Their pass/status/disposition are not evidence for new C-v3 counts; the independent blind raw-record audit remains the evidence. No new product acceptance, standalone download, release transport, or whole-book review inferred."},
"result":"Both original evidence gaps resolved within explicit immutable retention and artifact-identity scope; no textbook/source/product/old-evidence edits."}
out["command"]="PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-validation/retention-audit-v1.py"
with (ART/"retention-execution-v1.json").open("x",encoding="utf-8") as stream:stream.write(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"path":(ART/"retention-execution-v1.json").relative_to(ROOT).as_posix(),"sha256":sha((ART/"retention-execution-v1.json").read_bytes()),"prior_claims":len(retained_claims),"stable_repo_dependencies":len(dependencies),"product_reviews":[{"kind":r["kind"],"actor":r["reviewer_task"],"created_at":r["created_at"]} for r in reports],"attachments":57,"unique_delivered_paths":len({r["delivered_path"] for r in attachment_receipts}),"kernel_retained":retained_count,"actual_refreshed":len(kernel_receipts)},ensure_ascii=False))
