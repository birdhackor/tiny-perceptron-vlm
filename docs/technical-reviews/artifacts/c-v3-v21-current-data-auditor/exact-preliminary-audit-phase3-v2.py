"""Fixed product identity, local archive, publication provenance and scoped optionality."""
from pathlib import Path
import sys
import zipfile
engine=Path(__file__).with_name("exact-preliminary-audit-v3.py")
exec(compile(engine.read_text().split("for label,call in [(\"d01_canonical\"")[0],str(engine),"exec"))


def preserved_actors():
    old=doc(m["source18_retained_manifest"])
    prior=doc(old["assignments"])["tasks"]
    need(all(tasks.get(a["task"])==a for a in prior),"source18 actual assignment identity changed")
    for ref in assignments.get("actual_technical_dispatch_refs",[])+assignments.get("actual_technical_callback_refs",[]):doc(ref)
    return {"old_actual_assignment_dictionaries_preserved_exact":len(prior),"actual_technical_dispatch_documents_hashed":len(assignments.get("actual_technical_dispatch_refs",[])),"actual_technical_callback_documents_hashed":len(assignments.get("actual_technical_callback_refs",[])),"scope":"Preserved original actor assignments; bounded callbacks remain original owners, not new fresh primary assignments."}


def product_and_transport():
    release=doc("docs/decision-product/v3/public-release.json")
    for name,digest in release["implementation"].items():
        artifact({"path":name,"sha256":digest});need(m["implementation"][name]==digest,"fixed product source differs from candidate implementation")
    qualification=doc("docs/decision-product/v3/qualification-summary.json")
    need(hashes["docs/decision-product/v3/qualification-summary.json"]==release["qualification_summary_sha256"],"qualification summary digest differs")
    bundle=doc("docs/decision-product/v3/metadata/qualified-bundle-manifest.json")
    need(bundle["status"]=="qualified" and "bounded" in bundle["scope"].lower(),"qualified scope lost")
    path=ROOT/"assets/releases/c-v3-20261101/tiny-perceptron-c-v3-verified.zip"
    if not path.is_file() or path.stat().st_size!=release["bytes"]:
        path=Path("/workspace/work/c-product-v3-20261101/assets/releases/c-v3-20261101/tiny-perceptron-c-v3-verified.zip")
    need(path.is_file(),"actual local qualified ZIP unavailable")
    archive=path.read_bytes();need(len(archive)==release["bytes"] and sha(archive)==release["sha256"],"actual ZIP bytes differ")
    members={}
    with zipfile.ZipFile(path) as z:
        names=z.namelist();need(len(names)==len(set(names)),"ZIP duplicate member")
        for name in names:
            need(not name.startswith("/") and ".." not in Path(name).parts,"ZIP traversal member")
        for name,expected in release["bundle"].items():
            matches=[n for n in names if n==name or n.endswith("/"+name)]
            need(len(matches)==1,"ZIP fixed bundle member not unique: "+name)
            actual=sha(z.read(matches[0]));need(actual==expected,"ZIP member digest differs: "+name)
            members[matches[0]]=actual
    public=doc(m["qualified_product_publication"]["actual_anonymous_transport"])
    actions=doc(m["qualified_product_publication"]["actual_Actions_result"])
    need(public["actor"]=="/root" and public["http_status"]==200 and public["authorization_header_sent"] is False,"public transport receipt scope/actor differs")
    need(public["sha256"]==release["sha256"] and public["bytes"]==release["bytes"] and public["exact_qualified_archive"] is True,"anonymous receipt not exact qualified archive")
    need(str(actions["workflow_run"])==str(public["workflow_run_id"]) and actions["asset_id"]==public["asset_id"] and actions.get("source_commit",actions.get("workflow_revision"))==public["source_commit"],"Actions/anonymous identity differs")
    need(actions["success"] is True and api.at(public["published_at"])<=api.at(public["verification_started_at"])<=api.at(public["verified_at"])<=START,"public receipt chronology differs")
    return {"fixed_product_sources_exact":len(release["implementation"]),"local_archive_path":str(path),"archive_bytes":len(archive),"archive_sha256":sha(archive),"archive_members":len(names),"four_fixed_bundle_member_hashes":members,"anonymous_event_actor":public["actor"],"public_root_verified_at":public["verified_at"],"workflow_revision":public["source_commit"],"scope":"Auditor independently hashed/parsed local archive and 17 source identities; did not load weights, rerun model/final/qualification, download the release, or claim book main/Pages deployed. Anonymous transport is the preserved root event, distinct from local ZIP and bounded model qualification."}


def pair_optional_scope():
    p=catalog["validation"]["continuity"];original=doc(p["original_report"]);current=doc(p["report"])
    old=api.node(original,p["original_json_pointer"])["issues"][0]
    new=api.node(current,p["json_pointer"])["issues"][0]
    need(old["issue_id"]==new["issue_id"]=="C-PUBLISH22-01" and old["necessity"]=="required" and new["necessity"]=="optional","paired original/current classification not preserved")
    need(new["original_impact"]==old["original_impact"] and new["source_changed"] is False and new["post_original_only"] is True,"paired original impact or unchanged source lost")
    actual=doc(m["source_freeze"]);vpage=pages["validation"];text=(ROOT/vpage["snapshot"]).read_text()
    need(old["source_quote"] in text,"paired original observation quote absent")
    reread=new["newly_read_source"];need(reread["source_sha256"]==pages["19.25"]["source_sha256"],"pair current newly read source stale")
    matched=[]
    for i,decision in enumerate(dp["items"]):
        if decision["report"] in (p["original_report"],p["report"]) and api.node(doc(decision["report"]),decision["json_pointer"]).get("issue_id")==old["issue_id"]:
            need(decision["final_necessity"]=="optional" and decision["action"]=="defer","optional unchanged observation falsely fixed")
            matched.append(i)
    need(len(matched)==2,"original/current paired occurrences merged or omitted")
    return {"original_classification":"required / validation revise, preserved", "current_classification":"optional / nonblocking_source_unchanged", "DP_occurrence_positions":matched,"DP_action":"defer; not source repaired", "current_reason":"Same owner subsequently read linked 19.25 and explicitly narrowed the immediate validation evidence-category commitment; original local omission/impact and suggestion remain. Peer pass is expressly not offered as proof.","scope":"Auditor checks provenance and actual stated scope of the changed classification; not a new reader or model-score re-verification."}


def new_executed_scope():
    receipt=doc(BASE+"evidence/kernel-refresh-source22-v1.json")
    need(receipt["source_freeze"]==m["source_freeze"] and len(receipt["actual_executions"])==1 and receipt["retained_current_count"]==331,"kernel fresh/retained scope differs")
    item=receipt["actual_executions"][0]
    need(item["lesson"]=="19.25" and item["source_sha256"]==pages["19.25"]["notebook_sha256"],"kernel current Notebook identity differs")
    technical=doc(catalog["19.25"]["technical"]["report"])
    boundary=technical["post_intake_release_boundary"]
    return {"new_CPU_notebook_execution_count":1,"retained_executed_notebooks":331,"new_execution_owner":receipt["actor"],"source22_T_early_release_limit_retained":boundary,"scope":"Preserved recorded execution identities only. Auditor did not execute Notebook, train, reopen final or turn 332-kernel metadata into scientific correctness. Later root transport evidence has its own actor/time."}


for label,call in [("d03_preserved_661_and_dispatch_refs",preserved_actors),("d08_product_archive_transport",product_and_transport),("d06_pair_optional_scope",pair_optional_scope),("d08_fresh_execution_scope",new_executed_scope)]:run(label,call)
output={"kind":"independent bounded preliminary actual-data audit execution, phase 3","reviewer_task":TASK,"started_at":START.isoformat(),"completed_at":datetime.now(UTC).isoformat(),"exact_candidate":{"path":CANDIDATE,"sha256":PINS[CANDIDATE]},"exact_DP":{"path":DP,"sha256":PINS[DP]},"results":results,"failures":failures,"hash_index":[{"path":n,"sha256":v} for n,v in sorted(hashes.items())],"script_sha256":sha(Path(__file__).read_bytes()),"scope":"Metadata/local archive identity only. No network, model/research/final rerun, future seal, input mutation or own receipt audit."}
target=HERE/sys.argv[1]
with target.open("x",encoding="utf-8") as stream:stream.write(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target.read_bytes()),"results":results,"failures":failures},ensure_ascii=False))
