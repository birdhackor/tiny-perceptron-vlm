"""Final incremental actual-data check: corrected R envelope and DP527 current closure."""
import hashlib
import json
import platform
from datetime import UTC,datetime
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
def read(ref):
    raw=(ROOT/ref["path"]).read_bytes();actual=sha(raw)
    need(actual==ref["sha256"],"changed reference: "+ref["path"])
    checked[ref["path"]]=actual
    return json.loads(raw)
def ref(path,digest):return {"path":path,"sha256":digest}
def at(value):
    x=datetime.fromisoformat(value);need(x.tzinfo is not None,"naive timestamp");return x

candidate_ref=ref(ASSEMBLY+"ordinary-preliminary-v3.json","f9842496257af985393d9db2898fe8d19c1cdf3b7f315c64cd56af65f545aa3c")
dp_ref=ref(ASSEMBLY+"dispositions-preliminary-v2.json","a4fb62415ec1bc6b10352e51437bb2109fbd9b58aef5afdf8cc5b505faa96b10")
m=read(candidate_ref);dp=read(dp_ref)
old=read(ref(ASSEMBLY+"ordinary-preliminary-v2.json","773bb0a800b316276d3b46506a3b1069fa9f47abc2c9d85b8720ea8b709152bf"))
old_dp=read(old["dispositions"])
need(m["dispositions"]==dp_ref,"candidate DP binding differs")
changed={k for k in set(old)|set(m) if old.get(k)!=m.get(k)}
need(changed=={"dispositions","previous_ordinary_candidate","actual_windows_source18_extra_collection","created_at","author_assembly_status","issue_reports","dispositions_collection"},"unexpected prelim2->3 changes: "+str(changed))
need(m["previous_ordinary_candidate"]==ref(ASSEMBLY+"ordinary-preliminary-v2.json","773bb0a800b316276d3b46506a3b1069fa9f47abc2c9d85b8720ea8b709152bf"),"previous candidate identity differs")
need({k for k in set(dp)|set(old_dp) if dp.get(k)!=old_dp.get(k)}=={"at","items"},"unexpected author DP changes")
positions=[i for i,(x,y) in enumerate(zip(old_dp["items"],dp["items"])) if x!=y]
need(len(dp["items"])==len(old_dp["items"])==1094 and positions==[527],"DP changed outside actual527")
need(dp["items"][:257]==old_dp["items"][:257],"first257 changed")
before,after=old_dp["items"][527],dp["items"][527]
need({k for k in set(before)|set(after) if before.get(k)!=after.get(k)}=={"verification","previous_source18_verification","source22_current_extra_collection"},"527 original need/issue/author judgment changed")
need(after["previous_source18_verification"]==before["verification"],"old closure link was erased")
extra_ref=ref(BASE+"original-owner-callbacks-v21/windows-source18-extra-v1/current-v1.json","916f0f5dadd6bc5dbfbc32a9460c535c09e8c657324c718b6b99bee1443971fe")
receipt_ref=ref(BASE+"actual-windows-source18-extra-collected-v1.json","5f5db8824cac7a3052d4160ffc372d71e3b6de0b69d4067be90590f1f1d79b6c")
need(after["verification"]==[{"kind":"tool_audit","report":extra_ref}] and after["source22_current_extra_collection"]==receipt_ref,"527 not bound to actual new independent addendum")
extra=read(extra_ref);receipt=read(receipt_ref)
need(extra["verdict"]=="pass" and extra["resolved_issue_ids"]==["AUTHOR-SOURCE18-WINDOWS-SYNC"] and extra["issues"]==[],"extra scope/resolved IDs differ")
need(extra["reviewer_task"]=="/root/omni_p5_full_tools_review","new addendum owner differs")
assignments=read(m["assignments"]);assignment=next(a for a in assignments["tasks"] if a["task"]==extra["reviewer_task"])
need(assignment["role"]=="technical" and assignment["wave"]!="c-v3-integration-07","existing expert relabeled as fresh current primary")
need(set(x["path"] for x in extra["checked_files"])==set(after["affected_files"]),"extra affected-file scope differs")
for item in extra["checked_files"]:
    actual=sha((ROOT/item["path"]).read_bytes());checked[item["path"]]=actual
    need(actual==item["sha256"]==m["implementation"][item["path"]],"extra current file not actual candidate bytes")
original=read(extra["original_issue"])
need(extra["original_issue"]["path"]==after["report"]["path"] and extra["original_issue"]["sha256"]==after["report"]["sha256"],"extra reviewed a different original finding")
for key in ("same_owner_source18","old_native_verification"):read(extra["prior_closure_read"][key])
basis=read(extra["current_same_owner_basis"]["report"])
need(basis["reviewer_task"]==extra["reviewer_task"],"expert current basis owner differs")
for item in extra["checked_files"]:
    need(any(x["path"]==item["path"] and x["sha256"]==item["sha256"] for x in basis["checked_files"]),"expert reused basis with different source bytes")
for key in ("actual_existing_collection","evidence","targeted_log"):
    r=extra["current_same_owner_basis"][key];raw=(ROOT/r["path"]).read_bytes();need(sha(raw)==r["sha256"],"expert prior controls/receipt changed");checked[r["path"]]=r["sha256"]
need(extra["verification_actions"]["contract_tests_run_this_addendum"]==0,"new tests scope unexpectedly differs")
copen=read(m["actual_C_open"]);need(copen["status"]=="OPEN" and "completed_at" not in copen,"C future completion fabricated")
entries=[x for x in receipt["collection_index"] if x["path"]==extra_ref["path"] and x["sha256"]==extra_ref["sha256"]]
need(len(entries)==1 and entries[0]["report_owner"]==extra["reviewer_task"] and receipt["collection_role"]=="continuity","extra not genuinely/uniquely collected in current C")
need(at(copen["started_at"])<=at(extra["created_at"])<=at(entries[0]["received_at"])<=at(dp["at"])<=at(m["created_at"])<=started,"extra/candidate true chronology differs")
need(m["actual_windows_source18_extra_collection"]==receipt_ref,"candidate lacks actual extra receipt")
oldreports={(r["path"],r["sha256"]) for r in old["issue_reports"]};newreports={(r["path"],r["sha256"]) for r in m["issue_reports"]}
need(newreports==oldreports|{(extra_ref["path"],extra_ref["sha256"])} and len(m["issue_reports"])==len(old["issue_reports"])+1,"actual inventory change differs")
need(dp["pending_count"]==8 and [x["issue_id"] for x in dp["items"] if x["action"]=="pending"]==["CDA-T-D"+str(i).zfill(2) for i in range(1,9)],"self-obligation pending set changed/faked zero")
rnew=read(m["review_waves"][-1]["seals"]["reader"]);rold=read(ref(BASE+"reader-phase-proof-v21-v2.json","1c8a6002b9816a9986c201c613ddf91c214127a945a45df01468f14696d90259"))
correction=read(m["reader_phase_envelope_correction"])
need(rnew["exact_start"] is None and "not observed exactspawn" in rnew["start_time_semantics"],"false R exact-start assertion")
for field in ("completed_at","collection_index","report_sha256s"):need(rnew[field]==rold[field],"R true end/collections changed")
source19=read(rnew["envelope_correction"]["source19_freeze_lower_bound"])
dispatch=read(rnew["envelope_correction"]["original_dispatch_bounds"])
need(at(source19["created_at"])==at(rnew["started_at"]),"R conservative bound is not source19 actual observation")
fresh_r=[a for a in assignments["tasks"] if a["wave"]=="c-v3-integration-07" and a["role"]=="reader"]
need(min(at(a["dispatch_window"]["not_after"]) for a in fresh_r)==at(rnew["actual_start_interval"]["not_after"]),"first child plan upper bound changed")
for a in fresh_r:need(at(rnew["started_at"])<=at(a["dispatch_window"]["not_before"])<=at(a["dispatch_window"]["not_after"])<=at(rnew["completed_at"]),"fresh reader outside actual interval envelope")
prior_results=[]
for name in ["exact-preliminary-audit-results-v3.json","exact-preliminary-audit-phase2-results-v5.json","exact-preliminary-audit-phase3-results-v2.json"]:
    path=(HERE/name).relative_to(ROOT).as_posix();raw=(ROOT/path).read_bytes();value=json.loads(raw)
    if "phase2" in name:
        need(len(value["failures"])==1 and value["failures"][0]["check"]=="d06_fixed_actual_links","unexpected old phase2 failures")
        rows=json.loads(value["failures"][0]["details"].split("fixed verification row failures: ",1)[1]);need([x["pointer"] for x in rows]==["/items/527"],"old failures not exhausted by actual527 correction")
    else:need(value["failures"]==[],"prior unchanged domain has unresolved failure")
    prior_results.append(ref(path,sha(raw)))
for key in ("source_freeze","baseline_supplements","numbered_section_reviews","dependency_audit","historical_fix_rechecks","reader_phase_envelope_correction"):read(m[key])
output={"kind":"exact preliminary-v3/DP2 bounded correction delta","reviewer_task":TASK,"started_at":started.isoformat(),"completed_at":datetime.now(UTC).isoformat(),"candidate":candidate_ref,"dispositions":dp_ref,"candidate_delta":sorted(changed),"DP_changed_rows":positions,"first257_unchanged":True,"DP_occurrences":1094,"explicit_pending_auditor_obligations":8,"new_report_adds_issue_occurrences":0,"actual527_current_closure_verified":True,"old527_finding_necessity_and_verification_preserved":True,"new_expert_callback_tests_executed":0,"expert_reuses_own_unchanged15tests_6controls":True,"reader_exact_start":None,"reader_actual_start_interval":rnew["actual_start_interval"],"reader_original_end_and_collection_unchanged":True,"prior_bounded_domain_evidence_reused_by_ref":prior_results,"checked_files":[ref(n,d) for n,d in sorted(checked.items())],"failures":[],"script_sha256":sha(Path(__file__).read_bytes()),"environment":{"python":platform.python_version(),"device":"CPU; metadata/byte identity only","UV_PROJECT_ENVIRONMENT":"/workspace/tiny-perceptron-vlm/.venv","UV_CACHE_DIR":"/tmp/uv-c3","UV_OFFLINE":"1","PYTHONPATH":"/tmp/tpb"},"scope":"Original failures preserved. Genuine current2-file closure remedies the sole remaining actual link mismatch. C OPEN, eight auditor self-obligations still honestly pending in author input; this delta does not fabricate the auditor receipt, a C end, a final ordinary candidate or a native full-gate result. No product/model/research rerun, network call, broad-index copy or input mutation."}
path=HERE/"exact-preliminary-v3-delta-results-v1.json"
with path.open("x",encoding="utf-8") as stream:stream.write(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k!="checked_files"}|{"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path.read_bytes()),"new_checked_files":len(checked)},ensure_ascii=False))
