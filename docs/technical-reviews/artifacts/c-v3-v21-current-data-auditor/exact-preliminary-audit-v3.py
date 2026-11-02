"""Independent bounded actual-data checks. Never synthesizes a C end or runs a gate."""
import ast
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK = "/root/c_v3_technical_coord/current_data_auditor"
BASE = "docs/course-revision-20261101-c-v3/"
CANDIDATE = BASE + "final-assembly-source22-v1/ordinary-preliminary-v2.json"
DP = BASE + "final-assembly-source22-v1/dispositions-preliminary-v1.json"
PINS = {CANDIDATE: "773bb0a800b316276d3b46506a3b1069fa9f47abc2c9d85b8720ea8b709152bf", DP: "b09b250db429afc8fd352eececb67c716705b51858ba08aa3778146b361b4df6"}
START = datetime.now(UTC)
hashes, docs, failures, results = {}, {}, [], {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(name):
    p = ROOT / name
    value = p.read_bytes()
    hashes[name] = sha(value)
    return value


def doc(ref):
    name = ref["path"] if isinstance(ref, dict) else ref
    if name not in docs:
        docs[name] = json.loads(raw(name))
    if isinstance(ref, dict) and hashes[name] != ref["sha256"]:
        raise ValueError("reference SHA mismatch: " + name + " expected=" + ref["sha256"] + " observed=" + hashes[name])
    return docs[name]


def artifact(ref):
    name = ref["path"]
    if name not in hashes:
        raw(name)
    if hashes[name] != ref["sha256"]:
        raise ValueError("artifact SHA mismatch: " + name + " expected=" + ref["sha256"] + " observed=" + hashes[name])


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def run(label, call):
    try:
        value = call()
        results[label] = value if value is not None else "bounded check completed"
    except Exception as error:
        failures.append({"check": label, "error_type": type(error).__name__, "details": str(error)})


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def refs(value, pointer=""):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and re.fullmatch("[0-9a-f]{64}", str(value.get("sha256", ""))):
            yield pointer, value
        for key, item in value.items():
            yield from refs(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from refs(item, pointer + "/" + str(i))


api = module("cda_actual_incremental", "docs/review-tools/incremental_revision.py")
setup = module("cda_actual_setup", "scripts/omni_revision/review_setup.py")
validator = module("cda_actual_technical", "scripts/check_technical_reviews.py")
m = doc({"path": CANDIDATE, "sha256": PINS[CANDIDATE]})
dp = doc({"path": DP, "sha256": PINS[DP]})
f = doc(m["source_freeze"])
pages = {p["page_id"]: p for p in f["pages"]}
catalog = {p["page_id"]: p for p in m["pages"]}
assignments = doc(m["assignments"])
tasks = {a["task"]: a for a in assignments["tasks"]}
sealed = api.review_seals(ROOT, m["review_waves"], ("reader", "technical"))
technical_reports, owners = api.technical_reports(m, ROOT)
readers = {t for t, a in tasks.items() if a["role"] == "reader"}
wave = "c-v3-integration-07"
new_seals = sealed[wave]
copen = doc(m["actual_C_open"])


def canonical():
    inventory = module("cda_actual_inventory", "docs/review-tools/composite_review.py").current_inventory(ROOT)
    live = {p["page_id"]: p for p in inventory["pages"]}
    need(len(live) == len(pages) == len(catalog) == 370, "canonical count/duplicates differ")
    need(set(live) == set(pages) == set(catalog), "canonical ID sets differ")
    fields = ("source", "selector", "source_sha256", "figures_sha256", "notebook", "notebook_sha256")
    notebooks = set()
    for pid, p in pages.items():
        for field in fields:
            need(live[pid].get(field) == p.get(field) == catalog[pid].get(field), pid + " /" + field + " differs")
        artifact({"path": p["snapshot"], "sha256": p["source_sha256"]})
        if p.get("notebook"):
            artifact({"path": p["notebook"], "sha256": p["notebook_sha256"]})
            notebooks.add(p["notebook"])
    for name, ref in f["figure_snapshots"].items():
        artifact(ref)
        need(hashes[ref["path"]] == sha((ROOT / name).read_bytes()), "figure live/snapshot differs: " + name)
    previous = doc(f["previous"])
    prior = {p["page_id"]: p for p in previous["pages"]}
    delta = [pid for pid in pages if any(pages[pid].get(x) != prior[pid].get(x) for x in fields)]
    need(delta == ["19.25"], "source21->22 delta differs: " + repr(delta))
    oldtext = (ROOT / prior["19.25"]["snapshot"]).read_text()
    newtext = (ROOT / pages["19.25"]["snapshot"]).read_text()
    pattern = re.compile(r"^```python[^\n]*\n(.*?)^```", re.S | re.M)
    oldblocks, newblocks = pattern.findall(oldtext), pattern.findall(newtext)
    need(len(oldblocks) == len(newblocks) > 0, "python block count differs")
    oldast = [ast.dump(ast.parse(b), include_attributes=False) for b in oldblocks]
    newast = [ast.dump(ast.parse(b), include_attributes=False) for b in newblocks]
    need(oldast == newast, "source22 Python AST differs")
    need(pattern.sub("<PYTHON>", oldtext) == pattern.sub("<PYTHON>", newtext), "text outside Python changed")
    retained = doc(m["source18_retained_manifest"])
    oldfreeze = {p["page_id"]: p for p in doc(retained["source_freeze"])["pages"]}
    delta18 = [pid for pid,p in pages.items() if pid not in oldfreeze or any(p.get(x) != oldfreeze[pid].get(x) for x in ("source","selector","source_sha256","figures_sha256"))]
    return {"canonical": len(pages), "notebooks": len(notebooks), "figures": len(f["figure_snapshots"]), "source21_delta": delta,
            "AST_identical_python_blocks": len(oldast), "strict_whitespace_only": re.sub(r"\s", "", oldtext) == re.sub(r"\s", "", newtext),
            "source18_delta": delta18, "source18_identical": len(pages)-len(delta18)}


def implementation():
    for key in ("criteria", "implementation"):
        for name, digest in m[key].items():
            artifact({"path": name, "sha256": digest})
    files = api.baseline_supplements(ROOT, m)
    for item in files:
        artifact({"path": item["path"], "sha256": item["sha256"]})
    api.check_tool_audit(ROOT, m, {t:a["role"] for t,a in tasks.items()}, sealed)
    return {"implementation": len(m["implementation"]), "criteria": len(m["criteria"]), "baseline_supplement_copies": len(files), "tool_primary":m["tool_audit"]["report"], "scope":"Existing primary byte binding checked; no new core approval or baseline/full gate."}


def actors():
    need(len(tasks)==len(assignments["tasks"]), "duplicate assigned task")
    need(not(set(tasks)&set(m["author_tasks"])), "author/reviewer task overlap")
    for task, a in tasks.items():
        need(task.startswith("/root/") and a["role"] in api.ROLES and a.get("context")=="fresh", "bad assignment: "+task)
    preserved = doc(assignments["preserved_assignments"])["tasks"]
    need(all(tasks.get(a["task"])==a for a in preserved), "original assignment dictionary changed")
    counts = Counter()
    for pid, page in catalog.items():
        for role in api.ROLES:
            proof=page[role]
            if proof.get("mode")!="review":continue
            report=doc(proof["report"]);task=api.owner(report);a=tasks[task]
            need(a["role"]==role, "wrong role "+pid+" "+task)
            need(a.get("lesson_id")==pid if role=="technical" else pid in a.get("pages",[]), "assignment membership "+pid+" "+task)
            if proof.get("wave")==wave:counts[role]+=1
    return {"unique_tasks":len(tasks),"preserved_task_dictionaries":len(preserved),"new_canonical_page_proofs":dict(counts),"new_actor_tasks":[a for a in tasks.values() if a["wave"]==wave]}


def stages():
    r,t=new_seals["reader"],new_seals["technical"]
    need(api.at(r["completed_at"])<=api.at(t["started_at"])<=api.at(t["completed_at"])<=api.at(copen["started_at"])<=START,"R/T/C stage chronology")
    need(copen["status"]=="OPEN" and "completed_at" not in copen and "continuity" not in m["review_waves"][-1]["seals"],"C open/complete contradiction")
    need(copen["preceding_technical_seal"]==m["review_waves"][-1]["seals"]["technical"],"C preceding technical ref differs")
    return {"R_end":r["completed_at"],"T_start":t["started_at"],"T_end":t["completed_at"],"C_start":copen["started_at"],"C_status":"OPEN; no manufactured end"}


c_receipts={}
def receipts():
    checked=0;self_excluded=0
    for role,phase in new_seals.items():
        byid=Counter((x["path"],x["sha256"]) for x in phase.get("collection_index",[]))
        for entry in phase.get("collection_index",[]):
            artifact(entry)
            try:
                report=doc(entry)
            except json.JSONDecodeError:
                report={}
            if api.owner(report)==TASK:
                self_excluded+=1;continue
            need(byid[(entry["path"],entry["sha256"]) ]==1,"duplicate phase receipt: "+entry["path"])
            need(entry["sha256"] in phase["report_sha256s"],"receipt SHA absent from phase")
            received=api.at(entry["received_at"])
            need(api.at(phase["started_at"])<=received<=api.at(phase["completed_at"])<=START,"receipt outside actual stage: "+entry["path"])
            for field in ("created_at","confirmed_at"):
                if field in report and "T" in str(report[field]):need(api.at(report[field])<=received,"report created after receipt: "+entry["path"])
            checked+=1
    for ref in {r["path"]:r for r in m["actual_C_intake_receipts"]}.values():
        intake=doc(ref)
        need(intake["wave"]==wave and intake["collection_role"]=="continuity","C intake role/wave")
        for entry in intake.get("collection_index",[]):
            key=(entry["path"],entry["sha256"])
            need(key not in c_receipts,"same report truly recollected: "+entry["path"])
            report=doc(entry);need(api.owner(report)!=TASK,"self receipt must be root audited")
            received=api.at(entry["received_at"]);created=api.at(report["created_at"])
            need(api.at(copen["started_at"])<=created<=received<=START,"C report created/received outside OPEN: "+entry["path"])
            need(entry.get("report_owner",api.owner(report))==api.owner(report),"C receipt owner differs")
            c_receipts[key]=entry
    return {"R_T_receipts_checked":checked,"own_original_receipt_excluded":self_excluded,"C_intake_ref_occurrences":len(m["actual_C_intake_receipts"]),"unique_C_intake_documents":len({r["path"] for r in m["actual_C_intake_receipts"]}),"unique_C_report_receipts":len(c_receipts)}


def technical():
    checked=0;new=0
    for pid,page in catalog.items():
        proof=page["technical"]
        if proof.get("mode")!="review":continue
        report=doc(proof["report"])
        if proof.get("wave")!=wave:continue
        body=(ROOT/pages[pid]["snapshot"]).read_text()
        errors=validator._validate(ROOT,ROOT/pages[pid]["source"],pid,body,report,owners,readers)
        need(not errors,pid+" current technical: "+repr(errors))
        original=doc(proof["original_report"]);task=api.owner(report)
        setup.check_original(original,proof["original_source_freeze"],pid,pid,task,validator,api,original_source_snapshots=proof.get("original_source_snapshots",[]))
        setup.check_report_pair(api,sealed,proof,original,report,tasks[task]);checked+=1
        if api.reported_issues(original):
            closure=report.get("original_issue_resolutions",[])
            positions={x.get("json_pointer") for x in closure}
            need(set(api.reported_issues(original))<=positions,pid+" original issue lacks explicit closure pointer")
        new+=1
    return {"new_canonical_original_current_pairs":checked,"new_current_native_schema_checks":new,"scope":"Bounded native schemas/receipts; no scientific re-verification."}


def numbered():
    intake=setup.check_numbered(m,intake=True)
    bundle=doc(m["numbered_section_reviews"])
    for entry in bundle["sections"]:
        report=doc(entry["report"]);original=doc(entry["original_report"])
        setup.check_report_pair(api,sealed,entry,original,report,tasks[api.owner(report)])
    collector_refs={}
    for pid in m["numbered_section_requirements"]["page_ids"]:
        report=doc(catalog[pid]["technical"]["report"])
        need(any(a.get("path")==m["numbered_section_reviews"]["path"] and a.get("sha256")==m["numbered_section_reviews"]["sha256"] for a in report["artifacts"]),"collector final bundle SHA differs: "+pid)
        collector_refs[pid]=catalog[pid]["technical"]["report"]
    return {"intake_native":intake,"all_26_actual_pair_receipts":len(bundle["sections"]),"collectors":collector_refs,"bundle":m["numbered_section_reviews"],"index":bundle["issue_inventory"],"scope":"Intake plus actual pair/collector checks. Full C-completed gate not run."}


def issue_inventory():
    listed={r["sha256"]:r for r in m["issue_reports"]}
    note_refs={r["sha256"]:r for p in m["pages"] for s in p.get("reader",{}).get("sessions",[]) for r in s["notes"]}
    expected=set()
    for digest,ref in (listed|note_refs).items():
        for pointer in api.reported_issues(doc(ref)):expected.add((digest,pointer))
    for extra in m.get("additional_issue_inventories",[]):
        items=api.node(doc(extra["report"]),extra["json_pointer"])
        need(isinstance(items,list),"additional issue inventory not list")
        expected.update((extra["report"]["sha256"],extra["json_pointer"]+"/"+str(i)) for i in range(len(items)))
    actual=[];pending=[]
    for i,row in enumerate(dp["items"]):
        actual.append((row["report"]["sha256"],row["json_pointer"]))
        api.node(doc(row["report"]),row["json_pointer"])
        for field in ("source_support","learning_benefit","change_burden","reason"):
            need(isinstance(row.get(field),str) and row[field].strip(),"DP /items/"+str(i)+" missing "+field)
        if row["action"]=="pending":pending.append(row["issue_id"])
        else:
            need(row.get("final_necessity") in {"necessary","optional"},"DP necessity")
            need(row["action"] in {"fixed","keep","defer","reject"},"DP action")
            need(row["final_necessity"]!="necessary" or row["action"]=="fixed","necessary issue not fixed at /items/"+str(i))
            if row["action"]=="fixed":need(row.get("verification"),"fixed lacks verification /items/"+str(i))
    need(len(actual)==len(set(actual)),"duplicate disposition occurrence")
    need(expected==set(actual),"issue occurrence exact set differs missing="+repr(sorted(expected-set(actual))[:8])+" extra="+repr(sorted(set(actual)-expected)[:8]))
    need(set(pending)=={"CDA-T-D"+str(i).zfill(2) for i in range(1,9)} and dp["pending_count"]==8,"pending obligations differ")
    accepted=doc(setup.ACCEPTED_BASELINE);old=doc(accepted["dispositions"])["items"]
    need(dp["items"][:257]==old[:257],"accepted first257 changed")
    return {"listed_reports_unique_sha":len(listed),"reader_note_reports":len(note_refs),"expected_occurrences":len(expected),"actual_occurrences":len(actual),"exact_set_equal":True,"accepted_first257_exact":True,"explicit_pending":pending}


for label,call in [("d01_canonical",canonical),("d02_implementation",implementation),("d03_actors",actors),("d04_stages",stages),("d04_receipts",receipts),("d05_new_technical",technical),("d07_numbered",numbered),("d06_issue_inventory",issue_inventory)]:
    run(label,call)
output={"kind":"independent bounded preliminary actual-data audit execution, phase 1", "reviewer_task":TASK,"started_at":START.isoformat(),"completed_at":datetime.now(UTC).isoformat(),"exact_candidate":{"path":CANDIDATE,"sha256":PINS[CANDIDATE]},"exact_DP":{"path":DP,"sha256":PINS[DP]},"results":results,"failures":failures,"hash_index":[{"path":name,"sha256":digest} for name,digest in sorted(hashes.items())],"script_sha256":sha(Path(__file__).read_bytes()),"scope":"No full gate, baseline replay, model/research rerun, fabricated C completion, candidate mutation, or own receipt audit. Phase 2 must inspect history, semantics/scope and remaining reader/retention data before any overall verdict."}
target=HERE/sys.argv[1]
with target.open("x",encoding="utf-8") as stream:stream.write(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target.read_bytes()),"checks":{key:({k:v for k,v in value.items() if k!='new_actor_tasks'} if isinstance(value,dict) else value) for key,value in results.items()},"failures":failures,"hashed_files":len(hashes)},ensure_ascii=False))
