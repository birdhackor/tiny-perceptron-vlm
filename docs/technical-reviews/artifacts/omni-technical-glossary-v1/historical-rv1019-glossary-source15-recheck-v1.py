"""Reinspect the four group6 identities and their current source15 semantics.

Reuse the unchanged personally read primary sources and previous numerical
execution; do not relabel those executions as new or merge four identities.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RD = "docs/course-revision-20261029-omni/reviews/technical/technical_glossary"
OWNER = "/root/omni_technical_coordinator/technical_glossary"


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_bytes())


def reference(path):
    return {"path":str(path),"sha256":digest(path)}


def ref(item):
    assert digest(item["path"])==item["sha256"]
    return read(item["path"])


def node(value,pointer):
    for token in pointer.split("/")[1:] if pointer else []:
        token=token.replace("~1","/").replace("~0","~")
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value


prior_path=RD+"/technical-historical-appendix-rv1019-technical-glossary-v3.json"
full_path=RD+"/technical-current-v4.json"
assert digest(prior_path)=="f424845e8a9bf27acc76aca3133ea2f0e061113a7a4bc1ddfee24c43caa3cf3c"
assert digest(full_path)=="af1597aa4659b97f66bb66926d29568d972ff864d772c849cee7ef914ee1eacd"
prior,full=read(prior_path),read(full_path)
assert prior["reviewer_task"]==full["reviewer_task"]==OWNER
group=ref(prior["request"]["report"])["groups"][6]
freeze_ref=full["source_freeze"]
assert freeze_ref["sha256"]=="1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
freeze=ref(freeze_ref);pages={p["page_id"]:p for p in freeze["pages"]}
page=pages["glossary"]
assert digest(page["snapshot"])==digest("course/glossary.md")==prior["source_sha256"]==full["source_sha256"]==page["source_sha256"]
assert page["figures_sha256"]==prior["figure_sha256"]==full["figure_sha256"]=={}
body=(ROOT/page["snapshot"]).read_text()
primary_path=OUT.relative_to(ROOT)/"historical-rv1019-glossary-primary-v1.json"
primary=read(primary_path)
assert primary["group"]==group
source_checks=[]
for s in primary["original_source_receipts"]:
    for field in ["raw","inspected_text"]:
        if field in s:
            item=s[field];assert digest(item["path"])==item["sha256"]
    source_checks.append({"source_id":s["source_id"],"raw":s["raw"],"version":s["version"],
                          "scope":"Same personally read pinned original; fingerprint rechecked, not a new HTTP or full original reread."})
repository_checks=[]
for source in prior["sources"]:
    if source["kind"]=="repository_code":
        assert digest(source["path"])==source["sha256"]
        repository_checks.append({"source_id":source["id"],"path":source["path"],"sha256":source["sha256"]})
for pid,item in primary["necessary_current_frozen_sections"].items():
    assert digest(item["path"])==item["sha256"]==pages[pid]["source_sha256"]
    assert (ROOT/item["path"]).read_bytes()==(ROOT/pages[pid]["snapshot"]).read_bytes()
for a in prior["artifacts"]:assert digest(a["path"])==a["sha256"]
audit=read(OUT.relative_to(ROOT)/"collector-current-audit-v4.json")
assert set(audit["source_changes_since14"])=={"5.7","chapter-10","chapter-14"}
assert audit["canonical_source_freeze"]==freeze_ref
assert audit["glossary"]["source_sha256"]==page["source_sha256"]
rows=[]
for index,(request,row) in enumerate(zip(group["items"],prior["historical_issue_rechecks"])):
    assert all(row[k]==v for k,v in request.items())
    disposition=node(ref(request["disposition_ref"]),request["disposition_ref"]["json_pointer"])
    preserved=node(ref(row["exact_original_disposition_snapshot"]["report"]),row["exact_original_disposition_snapshot"]["json_pointer"])
    assert disposition==preserved and disposition["action"]=="fixed"
    assert disposition["final_necessity"]==row["necessity"]=="necessary"
    assert row["prior_generation"]==group["prior_generation"]=="accepted_20261027"
    assert row["old_owner"]==group["old_owner"]=="/root/rv1019_technical_glossary" and row["old_role"]=="technical"
    assert request["historical_fix_verification"] in disposition["verification"]
    quote=row["pages"]["glossary"]["source_quote"]
    assert quote in body and quote==row["current_source_quote"]
    location=16 if "visual_character" in row["issue_id"] else 180
    assert body.splitlines()[location-1]==quote
    assessment=(
        "本次重新核source15 GL第16行及其G1/6.1上下文：已明示多碼點👍🏽及每scalar的UTF8域。UAX29 GB9/Unicode17 Extend/RFC條件與必要源皆未變，仍不把使用者圖案與碼點硬定一對一。2碼點8bytes、邊界及surrogate數值明確reuse前次本人實跑，非新跑。"
        if location==16 else
        "本次重新核source15 GL第180行及偏好/策略/替代目標上下文：仍說設法避免但不保每次小步，ratio/min定義正確。當前posttraining函式及13.12/13.13未變，clamp仍只是objective一路，不是policy硬投影。原式7/官方無constraint文句與六點/1.98反例明確reuse，不稱本次新執行。"
    )
    rows.append({"issue_id":row["issue_id"],"disposition_ref":row["disposition_ref"],"original_report":row["original_report"],
                 "json_pointer":row["json_pointer"],"historical_fix_verification":row["historical_fix_verification"],
                 "necessity":"necessary","status":"still_fixed","verdict":"pass","current_location":f"course/glossary.md:{location}",
                 "current_source_quote":quote,"reason":assessment,"original_dict_and_identity_unchanged":True})
assert len(rows)==4
assert len({(x["original_report"]["path"],x["json_pointer"]) for x in rows})==4
output={
    "created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":OWNER,
    "command":"uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/historical-rv1019-glossary-source15-recheck-v1.py",
    "environment":{"python":platform.python_version(),"device":"CPU source/identity/text only; no model or numeric probe execution"},
    "request":prior["request"],"prior_generation":group["prior_generation"],"old_owner":group["old_owner"],"old_role":"technical",
    "source_freeze":freeze_ref,"glossary_source_sha256":page["source_sha256"],"figure_sha256":{},
    "previous_appendix":reference(prior_path),"new_full_review":reference(full_path),"source15_delta_audit":reference(OUT.relative_to(ROOT)/"collector-current-audit-v4.json"),
    "unavailability":group["unavailability"],"primary_sources_reused_after_fingerprint_check":source_checks,
    "necessary_repository_sources_unchanged":repository_checks,"necessary_frozen_6_1_13_12_13_13_unchanged":True,
    "prior_numeric_probe_reused":reference(OUT.relative_to(ROOT)/"historical-rv1019-glossary-probe-v1.json"),
    "four_historical_items":rows,"verdict":"pass","meaning":"Four distinct necessary historical identities still fixed after personally checking current semantics and actual source15 delta; not merely a full_ref change.",
    "limits":"Only group6 technical. Unchanged primary/numerics reused with original dates/results. No new reader/coherence/210-tool/DP/stage/seal signature; actual collection/seal still pending at this artifact creation. All earlier reports and failures preserved.",
}
path=OUT/"historical-rv1019-glossary-source15-recheck-v1.json"
with path.open("x") as stream:json.dump(output,stream,ensure_ascii=False,indent=2);stream.write("\n")
print(json.dumps({"artifact":reference(path.relative_to(ROOT)),"verdict":"pass","historical_items":4,
                  "current_lines":[x["current_location"] for x in rows],"scope":"source15 actual semantics; unchanged evidence reused"},ensure_ascii=False))
