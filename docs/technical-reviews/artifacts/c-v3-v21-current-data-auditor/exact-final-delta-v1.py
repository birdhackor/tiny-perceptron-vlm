"""Short exact final closure delta; never verifies this auditor's receipts or C closure."""
import hashlib
import importlib.util
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE="docs/course-revision-20261101-c-v3/"
ASSEMBLY=BASE+"final-assembly-source22-v1/"
TASK="/root/c_v3_technical_coord/current_data_auditor"
started=datetime.now(UTC)
checked={}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def read(path,digest=None):
    raw=(ROOT/path).read_bytes();actual=sha(raw)
    if digest:need(actual==digest,"SHA mismatch: "+path)
    checked[path]=actual
    return json.loads(raw)
def ref(path):return {"path":path,"sha256":checked[path]}
def at(value):
    result=datetime.fromisoformat(value);need(result.tzinfo is not None,"timestamp without timezone");return result

final_path=ASSEMBLY+"ordinary-final-v1.json"
dp_path=ASSEMBLY+"dispositions-final-v1.json"
prior_path=ASSEMBLY+"ordinary-preliminary-v3.json"
prior_dp_path=ASSEMBLY+"dispositions-preliminary-v2.json"
current1_path=BASE+"technical-v21/current_data_auditor/exact-preliminary-current-v1.json"
original_path=BASE+"technical-v21/current_data_auditor/original-v1.json"
m=read(final_path,"88bff45fee320c16382142e216a85c573e905313552bf05a9c334b39eaeecc6e")
dp=read(dp_path,"47e97d4280ecb4795bc8dd04b07190738acb67492ecab8e279ee44dd21a1a8dd")
prior=read(prior_path,"f9842496257af985393d9db2898fe8d19c1cdf3b7f315c64cd56af65f545aa3c")
prior_dp=read(prior_dp_path,"a4fb62415ec1bc6b10352e51437bb2109fbd9b58aef5afdf8cc5b505faa96b10")
current1=read(current1_path,"99124754af274243b59a14f82046497a49489501de9cc4002555f278c17df26b")
original=read(original_path,"a5369ff16065b7de48f3298a509312e2d4bb430ec457fb0c41be2c5f51c450be")
need(m["dispositions"]==ref(dp_path) and m["previous_ordinary_candidate"]==ref(prior_path),"final candidate chain differs")
changed={k for k in set(m)|set(prior) if m.get(k)!=prior.get(k)}
need(changed=={"created_at","issue_reports","dispositions_collection","dispositions","author_assembly_status","previous_ordinary_candidate","current_data_preliminary_audit_collection"},"unexpected final ordinary changes: "+str(changed))
for key in ("pages","source_freeze","implementation","criteria","assignments","author_tasks","review_waves","numbered_section_reviews","historical_fix_rechecks","reader_phase_envelope_correction","qualified_product_publication"):
    need(m[key]==prior[key],"previously audited domain input changed: "+key)
need({k for k in set(dp)|set(prior_dp) if dp.get(k)!=prior_dp.get(k)}=={"pending_count","at","kind","items"},"unexpected final author DP fields")
need(len(dp["items"])==len(prior_dp["items"])==1094,"final occurrence count changed")
indices=[i for i,(before,after) in enumerate(zip(prior_dp["items"],dp["items"])) if before!=after]
need(indices==list(range(1086,1094)),"final DP changed outside eight actual audit closures")
need(dp["items"][:257]==prior_dp["items"][:257] and dp["items"][527]==prior_dp["items"][527],"old grade/first257/actual527 changed")
need(current1["reviewer_task"]==original["reviewer_task"]==TASK and current1["verdict"]=="pass","same original owner/current1 pass chain differs")
need(current1["exact_audited_candidate"]==ref(prior_path) and current1["exact_audited_dispositions"]==ref(prior_dp_path),"current1 not bound to actual preserved preliminary inputs")
ids={"CDA-T-D"+str(i).zfill(2) for i in range(1,9)}
need(set(current1["resolved_issue_ids"])==ids,"current1 resolved ID scope differs")
current_checked={x["path"]:x["sha256"] for x in current1["checked_files"]}
closures=[]
for i in indices:
    before,after=prior_dp["items"][i],dp["items"][i]
    need(before["action"]=="pending" and after["action"]=="fixed" and before["final_necessity"]==after["final_necessity"]=="necessary","eight original action/necessity chain differs")
    allowed={"action","reason","affected_pages","affected_files","verification"}
    need({k for k in set(before)|set(after) if before.get(k)!=after.get(k)}<=allowed,"original finding/support/identity changed at /items/"+str(i))
    need(after["report"]==ref(original_path) and after["issue_id"] in ids,"closure original identity differs")
    n=int(after["json_pointer"].split("/")[-1]);need(original["issues"][n]["issue_id"]==after["issue_id"],"closure issue pointer mismatch")
    need(after["verification"]==[{"report":ref(current1_path),"kind":"tool_audit"}],"final DP depends on future/current2 proof")
    same=[x for x in current1["original_issue_resolutions"] if x["issue_id"]==after["issue_id"] and x["json_pointer"]==after["json_pointer"]]
    need(len(same)==1 and same[0]["status"]=="resolved" and same[0].get("resolution"),"same-owner explicit item closure absent")
    need(not after.get("affected_pages") and after.get("affected_files"),"audit closure affected scope differs")
    for name in after["affected_files"]:
        need(name in current_checked,"affected file absent from real current1 checked_files: "+name)
        actual=sha((ROOT/name).read_bytes());need(actual==current_checked[name],"affected file bytes changed after current1")
    closures.append({"issue_id":after["issue_id"],"final_DP_pointer":"/items/"+str(i),"original_pointer":after["json_pointer"],"status":"verified","same_owner":TASK,"verification_report":ref(current1_path),"affected_files":after["affected_files"]})

def occurrence_keys(items):return [(x["report"]["sha256"],x["json_pointer"]) for x in items]
need(Counter(occurrence_keys(dp["items"]))==Counter(occurrence_keys(prior_dp["items"])),"final occurrence identity set changed")
need(len(set(occurrence_keys(dp["items"])))==1094,"duplicate final occurrence")
spec=importlib.util.spec_from_file_location("cda_final_issue_parser",ROOT/"docs/review-tools/incremental_revision.py")
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
need(api.reported_issues(current1)==[],"current1 introduces unadjudicated issue objects")
old_reports={(x["path"],x["sha256"]) for x in prior["issue_reports"]};new_reports={(x["path"],x["sha256"]) for x in m["issue_reports"]}
need(new_reports==old_reports|{(current1_path,checked[current1_path])} and len(m["issue_reports"])==len(prior["issue_reports"])+1,"issue report delta differs")
need(m["additional_issue_inventories"]==prior["additional_issue_inventories"],"additional issue inventories changed")
actions=Counter(x["action"] for x in dp["items"])
need(dp["pending_count"]==actions.get("pending",0)==0,"pending=0 not supported by actual actions")
need(all(x["action"] in {"fixed","keep","defer","reject"} and (x["final_necessity"]!="necessary" or x["action"]=="fixed") for x in dp["items"]),"necessary final occurrence unresolved")
need(at(current1["created_at"])<=at(dp["at"])<=at(m["dispositions_collection"]["received_at"])<=at(m["created_at"])<=started,"final author data chronology reversed/future")
need("continuity" not in m["review_waves"][-1]["seals"],"unexpected C completion in ordinary final")
for finding in current1["audit_discoveries_and_resolution"]:
    x=finding["original_failure_evidence"];raw=(ROOT/x["path"]).read_bytes();need(sha(raw)==x["sha256"],"original auditor failure evidence changed");checked[x["path"]]=x["sha256"]

result={"kind":"independent short exact final closure delta","reviewer_task":TASK,"started_at":started.isoformat(),"completed_at":datetime.now(UTC).isoformat(),"ordinary_final":ref(final_path),"dispositions_final":ref(dp_path),"changed_ordinary_fields":sorted(changed),"changed_DP_rows":indices,"closure_delta_checks":closures,"unique_complete_occurrence_count":1094,"occurrence_identity_set_unchanged":True,"new_current1_issue_occurrences":0,"action_counts":dict(actions),"recomputed_pending_count":0,"first257_and_527_unchanged":True,"previously_audited_substantive_inputs_unchanged":True,"C_status":"OPEN; no completed C seal in ordinary final","own_current1_or_current2_receipt_audited":False,"full_gate_or_model_rerun":False,"failures":[],"script_sha256":sha(Path(__file__).read_bytes()),"scope":"Only exact final data closure delta. Complete prior occurrence inventory is reused by identity-preserving delta, not by pending=0 flag. The ordinary final adds the actual current1 report with zero issue objects and does not pre-reference this current2. Root owns auditor receipts, later C seal/full native gate and publication."}
out=HERE/"exact-final-delta-results-v1.json"
with out.open("x",encoding="utf-8") as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
artifact_ref={"path":out.relative_to(ROOT).as_posix(),"sha256":sha(out.read_bytes())}
report={"schema_version":1,"kind":"independent_current_data_exact_final_delta_supplement_not_core_tool_primary","review_stage":"technical","lesson_id":"__current_data__","reviewer_task":TASK,"reviewer_context":"fresh","review_kind":"same original auditor bounded current2 closure followup","created_at":datetime.now(UTC).isoformat(),"verdict":"pass","source":original["source"],"source_sha256":original["source_sha256"],"figure_sha256":{},"original_report":ref(original_path),"basis_report":ref(current1_path),"exact_audited_candidate":ref(final_path),"exact_audited_dispositions":ref(dp_path),"checked_files":[{"path":p,"sha256":s} for p,s in sorted(checked.items())],"closure_delta_checks":closures,"resolved_issue_ids":sorted(ids),"occurrence_inventory_check":{"actual_unique_occurrences":1094,"set_unchanged_from_independently_audited_preliminary":True,"new_current1_issue_occurrences":0,"actual_action_counts":dict(actions),"recomputed_pending_count":0,"first257_unchanged":True,"row527_and_original_failures_preserved":True},"unchanged_inputs_scope":"Canonical/source22/source21 retention, implementation, assignments, report proofs, numbered bundle, historical chains, R envelope correction and qualified publication evidence are exact-equal to audited preliminary-v3. Their actual checks remain in current1 by reference; no model/research or full audit rerun.","artifacts":[{"id":"final_delta","kind":"execution",**artifact_ref,"description":"本人實際hash/parse final ordinary/DP並逐項核8 same-owner closure、coverage與完整occurrence identity delta。","command":"UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/tpb uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-current-data-auditor/exact-final-delta-v1.py","result":"所有具體delta assertions完成，failures=[]；未讀或驗本人1/2收件文件，未造C end。","environment":{"python":platform.python_version(),"device":"CPU; metadata/hash only"}},{"id":"final_delta_code","kind":"code","path":Path(__file__).relative_to(ROOT).as_posix(),"sha256":sha(Path(__file__).read_bytes()),"description":"本次短delta核對原碼；不重跑3phase大索引。"}],"issues":[],"new_issues":[],"scope_limits":["C仍OPEN；本人current1/current2收件事件、後續C close、完整native gate與main/Pages部署由root另驗，本報告沒有驗它們。","R exact_start仍未知；保守envelope不是精確spawn/OPEN事件。原R/527失敗證據、原必要性與歷史scope保留。","current2只供後續final manifest supplement，不回寫所審ordinary final或DP，避免SHA循環。","保留old core tool primary與generic I01/I02界線；metadata正確不等於科學、真人理解、圖像觀看、模型品質或部署成功。"]}
path=ROOT/BASE/"technical-v21/current_data_auditor/exact-final-current-v1.json"
with path.open("x",encoding="utf-8") as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"report":path.relative_to(ROOT).as_posix(),"sha256":sha(path.read_bytes()),"created_at":report["created_at"],"verdict":report["verdict"],"ordinary_final":ref(final_path),"dispositions_final":ref(dp_path),"occurrences":1094,"pending_recomputed":0,"delta_artifact":artifact_ref,"own_receipts_or_future_Cclose_audited":False},ensure_ascii=False))
