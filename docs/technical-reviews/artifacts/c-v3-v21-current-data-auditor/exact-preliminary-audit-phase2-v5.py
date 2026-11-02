"""Actual provenance/retention/history checks using real closed R/T and open C receipts."""
from pathlib import Path
import sys

# Reuse this auditor's pinned phase-1 definitions without its runner or writes.
engine = Path(__file__).with_name("exact-preliminary-audit-v3.py")
prefix = engine.read_text().split("for label,call in [(\"d01_canonical\"")[0]
exec(compile(prefix, str(engine), "exec"))
native_artifact = api.artifact


def logged_artifact(root, reference):
    value = native_artifact(root, reference)
    hashes[reference["path"]] = sha(value)
    return value


api.artifact = logged_artifact
api.document = lambda root, reference: json.loads(logged_artifact(root, reference))


def creceipt(reference):
    found=[]
    for ref in {r["path"]:r for r in m["actual_C_intake_receipts"]}.values():
        for entry in doc(ref).get("collection_index",[]):
            if (entry["path"],entry["sha256"])==(reference["path"],reference["sha256"]):found.append(entry)
    need(len(found)==1,"C reference lacks exactly one actual receipt: "+reference["path"])
    data=doc(reference);created,received=api.at(data["created_at"]),api.at(found[0]["received_at"])
    need(api.at(copen["started_at"])<=created<=received<=START,"C creation/receipt reversed: "+reference["path"])
    return data,created,received


def retained_technical():
    patches=api.author_patches(ROOT,m,pages);files=api.file_versions(ROOT,patches)
    checked=[]
    for pid,page in catalog.items():
        proof=page["technical"]
        if proof.get("mode")!="review" or proof.get("wave")==wave:continue
        report=doc(proof["report"]);versions=api.page_versions(ROOT,pages[pid],pages[pid],patches)
        errors=validator._validate(ROOT,ROOT/pages[pid]["source"],pid,(ROOT/pages[pid]["snapshot"]).read_text(),report,owners,readers,earlier_versions={"source_sha256":versions,"files":files})
        need(not errors,pid+" retained actual dependency/schema: "+repr(errors))
        checked.append(pid)
    baseline=module("cda_baseline_input_validation","docs/review-tools/baseline_replay.py")
    restores=baseline.checked_restores(ROOT,api.baseline_supplements(ROOT,m))
    old=doc(m["source18_retained_manifest"]);oldpages={p["page_id"]:p for p in old["pages"]}
    oldfreeze={p["page_id"]:p for p in doc(old["source_freeze"])["pages"]}
    same=[pid for pid,p in pages.items() if pid in oldfreeze and all(p.get(k)==oldfreeze[pid].get(k) for k in ("source","selector","source_sha256","figures_sha256"))]
    retained_c=[pid for pid in same if catalog[pid]["continuity"]==oldpages[pid]["continuity"]]
    need(len(same)==355 and len(retained_c)==354,"source18 retention counts differ")
    return {"retained_technical_current_native_schema_and_dependency_checks":len(checked),"baseline_pre_copy_actual_restore_inputs":len(restores),"source_identical_pages":len(same),"old_C_proofs_retained_exact":len(retained_c),"extra_fresh_C_on_unchanged_page":[pid for pid in same if pid not in retained_c],"scope":"Retained source/dependency identity and original scope only; no old research re-execution or new semantic primary."}


def reader_data():
    checked=[];sessions=0;fresh=[]
    for pid,page in catalog.items():
        proof=page["reader"]
        if proof.get("mode")!="review":continue
        report=doc(proof["report"]);task=api.owner(report)
        current=api.node(report,proof.get("json_pointer",""))
        need(api.patched_version(current["source_sha256"],pages[pid]["source_sha256"],api.page_versions(ROOT,pages[pid],pages[pid],api.author_patches(ROOT,m,pages))) is not None,"reader current source differs beyond declared author patches: "+pid)
        api.reader_closure(ROOT,proof,current,task,pid,sealed[proof["wave"]]["reader"],sealed)
        for session in proof.get("sessions",[]):
            api.reader_session(ROOT,session,task);sessions+=1
            opening=doc(session["opening"])
            if proof.get("wave")==wave:
                plan=doc(opening["reading_plan"]);sf=doc(opening["source_freeze"])
                frozen=next(x for x in sf["pages"] if x["page_id"]==pid)
                need(plan["reviewer_task"]==task and plan["source_freeze"]==opening["source_freeze"],"reader plan/freeze owner mismatch")
                need(session["source"]=={"path":frozen["snapshot"],"sha256":frozen["source_sha256"]},"reader source snapshot provenance mismatch")
                need(api.at(plan["created_at"])<=api.at(opening["started_at"]),"reader plan created after opening")
        checked.append(pid)
        if proof.get("wave")==wave:fresh.append(pid)
    q=catalog["19.25"]["reader"];basis=doc(q["basis"]["report"]);current=doc(q["report"]);closure=doc(q["checkpoint_closure"]["report"])
    need(basis["format_classification"].startswith("AST-equivalent"),"reader22 format inaccurately classified")
    need(api.at(basis["confirmed_at"])<=api.at(current["created_at"])<=api.at(closure["created_at"]),"reader22 basis/current/closure chronology")
    need(len(api.node(closure,q["checkpoint_closure"]["json_pointer"])["checkpoint_resolutions"])==21,"reader22 original observations omitted")
    return {"canonical_reader_closures_checked":len(checked),"sequential_sessions_checked":sessions,"fresh_R_page_proofs":len(fresh),"fresh_R_tasks":sorted({api.owner(doc(catalog[p]["reader"]["report"])) for p in fresh}),"reader22":"Same original owner bounded whole revised page/basis/current/21 closures; not a fabricated new first read or strict whitespace-only claim."}


def current_continuity():
    seen=set();count=0
    for pid,page in catalog.items():
        proof=page["continuity"]
        if proof.get("wave")!=wave:continue
        full,created,received=creceipt(proof["report"])
        original,ocreated,oreceived=creceipt(proof["original_report"])
        owner=api.owner(full);assignment=tasks[owner]
        need(api.owner(original)==owner and assignment["role"]=="continuity" and pid in assignment["pages"],"C owner/assignment mismatch")
        need(api.at(assignment["dispatch_window"]["not_before"])<=ocreated<=oreceived<=created<=received,"C current existed before original receipt: "+pid)
        need(original.get("schema_version")==1 and original.get("review_stage")=="continuity" and original.get("reviewer_context")=="fresh","C original native header incomplete")
        oldfreeze=doc(proof["original_source_freeze"]);oldpage=next(x for x in oldfreeze["pages"] if x["page_id"]==pid)
        for data,pointer,frozen in ((original,proof.get("original_json_pointer",""),oldpage),(full,proof.get("json_pointer",""),pages[pid])):
            member=api.node(data,pointer)
            need(member["page_id"]==pid and member["source_sha256"]==frozen["source_sha256"] and member["figures_sha256"]==frozen["figures_sha256"],"C original/current source identity mismatch: "+pid)
            text=logged_artifact(ROOT,{"path":frozen["snapshot"],"sha256":frozen["source_sha256"]}).decode()
            need(member.get("source_quote") and member["source_quote"] in text,"C original/current quote absent: "+pid)
            need(member.get("checks") and isinstance(member.get("issues"),list),"C original/current page checks absent")
        need(full["verdict"]=="pass","C current still revise")
        seen.add((proof["report"]["path"],proof["original_report"]["path"]));count+=1
    dependency,created,received=creceipt(m["dependency_audit"])
    need(set(dependency["inventory_page_ids"])==set(pages) and dependency["verdict"]=="pass","global C inventory differs")
    for pid,page in catalog.items():
        proof=page["technical"]
        if proof.get("wave")==wave:
            original=doc(proof["original_report"]);current=doc(proof["report"])
            need(api.at(proof["original_collection"]["received_at"])<=api.at(current["created_at"]),"new T current predates true original receipt: "+pid)
    return {"new_C_canonical_pages":count,"fresh_full_original_pairs":len(seen),"new_T_current_after_original_receipt":18,"global_C_inventory":len(dependency["inventory_page_ids"]),"C_status":"OPEN; auditor own future receipt and C close excluded"}


historical_replacements={}
def history():
    helpermodule=module("cda_historical_bounded_v2","docs/review-tools/historical_fix_rechecks_v2.py")
    sidecar=doc(m["historical_fix_rechecks"]);allkeys=set();newT=0;newC=0;rows_total=0;groups=0
    dmap={(x["report"]["path"],x["report"]["sha256"],x["json_pointer"]):x for x in dp["items"]}
    for chainref in sidecar["chains"]:
        chain=doc(chainref)
        helper=helpermodule.Rechecks.__new__(helpermodule.Rechecks)
        helper.root=ROOT;helper.manifest=m;helper.live=pages;helper.tasks={t:a["role"] for t,a in tasks.items()};helper.sealed=sealed;helper.api=api
        helper.entries={};helper.used=set();helper.reports=[];helper.checked_readers=set();helper.assignments=tasks;helper.decisions=dmap;helper.frozen=pages
        helper.prior(chain["prior_active_pointer"])
        for group in chain["rechecks"]:
            groups+=1;appendix=doc(group["appendix"]);rows=appendix["historical_issue_rechecks"];rows_total+=len(rows)
            if group["kind"]!="continuity":
                try:
                    helper.group(group)
                except Exception as error:
                    raise ValueError(chainref["path"]+" /rechecks/"+str(chain["rechecks"].index(group))+" "+str(group.get("page_id"))+": "+str(error)) from error
                if group["collection"]["wave"]==wave:newT+=len(rows)
            else:
                full,fc,fr=creceipt(group["full_review"]);original,oc,orr=creceipt(group["original_review"]);app,ac,ar=creceipt(group["appendix"])
                task=api.owner(full);assigned=tasks[task]
                need(task==api.owner(original)==api.owner(app) and task not in helper.old_tasks and assigned["role"]=="continuity","historical C fresh owner mismatch")
                need(oc<=orr<=fc<=fr<=ac<=ar and api.at(group["collection"]["received_at"])==ar,"historical C original/current/appendix chronology")
                for row in rows:
                    ref=row["disposition_ref"];need((ref["path"],ref["sha256"])==(helper.old_disposition_ref["path"],helper.old_disposition_ref["sha256"]),"wrong historical prior DP")
                    old=api.node(helper.old_dispositions,ref["json_pointer"]);key=(old["report"]["path"],old["report"]["sha256"],old["json_pointer"])
                    need(dmap.get(key)==old,"historical closed disposition changed")
                    need(old["action"]=="fixed" and row["necessity"]==old["final_necessity"],"historical necessity downgraded")
                    verification=row["historical_fix_verification"];need(verification in old["verification"],"old verification replaced")
                    oldissue=doc(old["report"]);api.node(oldissue,old["json_pointer"])
                    need(helper.old_tasks[api.owner(oldissue)]["role"]=="continuity","old owner role differs")
                    helper.unavailable(group["unavailability"],api.owner(oldissue),ac)
                    need(row["original_report"]==old["report"] and row["json_pointer"]==old["json_pointer"] and row["issue_id"]==old["issue_id"],"original historical C issue identity differs")
                    need(row["status"]=="still_fixed" and row["verdict"]=="pass" and row.get("reason"),"historical C issue not explained still_fixed")
                    need(set(row["pages"])==set(old["affected_pages"]),"historical affected page evidence incomplete")
                    oldfix=api.node(doc(verification["report"]),verification["json_pointer"]);oldpid=oldfix.get("page_id",oldfix.get("lesson_id"));prior=helper.old_pages[oldpid]
                    need(oldfix["source_sha256"]==prior["source_sha256"] and oldfix.get("figures_sha256",oldfix.get("figure_sha256"))==prior["figures_sha256"],"historical original fix source differs")
                    for pid,e in row["pages"].items():
                        proof=catalog[pid]["continuity"]
                        need(proof["report"]==group["full_review"] and proof["original_report"]==group["original_review"],"historical current full/original affected-page mismatch: "+pid)
                        need(pid in assigned["pages"] and e["source_sha256"]==pages[pid]["source_sha256"] and e["figures_sha256"]==pages[pid]["figures_sha256"],"historical C page identity differs")
                        need(e["source_quote"] in logged_artifact(ROOT,{"path":pages[pid]["snapshot"],"sha256":pages[pid]["source_sha256"]}).decode() and e.get("reason"),"historical C page quote/reason absent")
                        if pid in helper.old_pages and (pages[pid]["source_sha256"]!=helper.old_pages[pid]["source_sha256"] or pages[pid]["figures_sha256"]!=helper.old_pages[pid]["figures_sha256"]):
                            try:
                                helper.validate_reader(pid,api.owner(doc(catalog[pid]["technical"]["report"])),wave)
                            except Exception as error:
                                raise ValueError(chainref["path"]+" /rechecks/"+str(chain["rechecks"].index(group))+" reader "+pid+": "+str(error)) from error
                    for evidence in row["current_evidence"]:logged_artifact(ROOT,evidence)
                newC+=len(rows)
            for row in rows:
                old=api.node(doc(row["disposition_ref"]),row["disposition_ref"]["json_pointer"])
                k=((old["report"]["path"],old["report"]["sha256"],old["json_pointer"]),(row["historical_fix_verification"]["report"]["path"],row["historical_fix_verification"]["report"]["sha256"],row["historical_fix_verification"].get("json_pointer",""),row["historical_fix_verification"].get("kind","page")))
                need(k not in allkeys,"duplicate historical replacement")
                allkeys.add(k);historical_replacements[k]=(group,row)
    portability=doc(m["historical_input_portability"])
    for row in portability["copies"]:
        logged_artifact(ROOT,row["portable_byteidentical_copy"])
        need(row["sha256"]==row["portable_byteidentical_copy"]["sha256"],"portable input expected SHA differs")
        original=Path(row["actual_original_locator"])
        if original.is_file():need(sha(original.read_bytes())==row["sha256"],"portable original/copy actual bytes differ")
    return {"chains":len(sidecar["chains"]),"groups":groups,"unique_historical_rows":len(allkeys),"rows_total":rows_total,"new_T_rechecks":newT,"new_C_rechecks":newC,"portable_immutable_inputs":len(portability["copies"]),"scope":"Actual native prior/unavailability and closed-T group checks; open-C groups independently checked against real intake, without constructing or claiming a completed-C native historical gate."}


def fixed_links():
    checked=0;historical_used=set()
    fixed_errors=[]
    for i,decision in enumerate(dp["items"]):
        try:
            if decision["action"]!="fixed":continue
            oldreport=doc(decision["report"]);oldowner=api.owner(oldreport);checked_pages=set();checked_files=set()
            for verification in decision.get("verification",[]):
                report=doc(verification["report"]);task=api.owner(report)
                need(task in tasks and (task==oldowner or oldowner in m["author_tasks"] or oldowner is None),"fixed not rechecked by owner at /items/"+str(i))
                hkey=((decision["report"]["path"],decision["report"]["sha256"],decision["json_pointer"]),(verification["report"]["path"],verification["report"]["sha256"],verification.get("json_pointer",""),verification.get("kind","page")))
                if verification.get("kind")=="tool_audit":
                    need(report["verdict"]=="pass" and decision["issue_id"] in report.get("resolved_issue_ids",[]),"tool fix missing independent closure")
                    checked_files.update(x["path"] for x in report.get("checked_files",[]) if (ROOT/x["path"]).is_file() and sha((ROOT/x["path"]).read_bytes())==x["sha256"])
                else:
                    member=api.node(report,verification.get("json_pointer",""));pid=member.get("page_id",member.get("lesson_id"))
                    need(pid in pages and member.get("verdict")=="pass","fixed page lacks reviewer pass /items/"+str(i))
                    if api.patched_version(member.get("source_sha256"),pages[pid]["source_sha256"],api.page_versions(ROOT,pages[pid],pages[pid],api.author_patches(ROOT,m,pages))) is not None and member.get("figures_sha256",member.get("figure_sha256"))==pages[pid]["figures_sha256"]:checked_pages.add(pid)
                if hkey in historical_replacements:
                    group,row=historical_replacements[hkey];historical_used.add(hkey)
                    checked_pages.update(decision.get("affected_pages",[]));checked_files.update(decision.get("affected_files",[]))
            need((decision.get("affected_pages") or decision.get("affected_files")) and set(decision.get("affected_pages",[]))<=checked_pages and set(decision.get("affected_files",[]))<=checked_files,"fixed affected scope not currently covered /items/"+str(i)+" issue="+str(decision["issue_id"]))
            checked+=1
        except Exception as error:
            fixed_errors.append({"pointer":"/items/"+str(i),"issue_id":decision.get("issue_id"),"details":str(error)})
    need(not fixed_errors,"fixed verification row failures: "+json.dumps(fixed_errors,ensure_ascii=False))
    need(historical_used==set(historical_replacements),"unused historical replacement")
    return {"fixed_disposition_verifications_checked":checked,"historical_replacements_used":len(historical_used),"scope":"Actual linkage, identity, same-owner and preserved necessity; does not assert semantic truth merely from pass."}


for label,call in [("d05_retention_dependencies",retained_technical),("d05_reader_actual_traces",reader_data),("d04_d05_actual_C",current_continuity),("d06_historical_actual",history),("d06_fixed_actual_links",fixed_links)]:run(label,call)
output={"kind":"independent bounded preliminary actual-data audit execution, phase 2","reviewer_task":TASK,"started_at":START.isoformat(),"completed_at":datetime.now(UTC).isoformat(),"exact_candidate":{"path":CANDIDATE,"sha256":PINS[CANDIDATE]},"exact_DP":{"path":DP,"sha256":PINS[DP]},"results":results,"failures":failures,"hash_index":[{"path":name,"sha256":digest} for name,digest in sorted(hashes.items())],"script_sha256":sha(Path(__file__).read_bytes()),"engine_sha256":sha(engine.read_bytes()),"scope":"No full gate or fabricated C completion. Own receipt and actual later C close excluded. Product/publication scope still needs separate final inspection."}
target=HERE/sys.argv[1]
with target.open("x",encoding="utf-8") as stream:stream.write(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target.read_bytes()),"checks":results,"failures":failures,"hashed_files":len(hashes)},ensure_ascii=False))
