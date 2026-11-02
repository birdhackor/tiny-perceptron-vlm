"""Source14/bundle4 bounded callback; reuse unchanged personally read science.

New work checks the actual delta and provenance, not other sections' science,
reader acceptance, the 210-tool audit, dependency audit, or a stage seal.
"""

import hashlib
import importlib.util
import json
import platform
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OWNER = "/root/omni_technical_coordinator/technical_glossary"
RD = "docs/course-revision-20261029-omni/reviews/technical/technical_glossary"
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v4.json"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v5.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def reference(path):
    return {"path": str(path), "sha256": digest(path)}


def ref(item):
    assert digest(item["path"]) == item["sha256"], item
    return read(item["path"])


def utc(value):
    result = datetime.fromisoformat(value)
    assert result.tzinfo is not None
    return result


bundle = read(BUNDLE)
assert digest(BUNDLE) == "1ddabe842cbb17502ae1fb08783825c16f5669c22b8c294c0e9f83046c8b19da"
old_bundle = ref(bundle["previous_bundle"])
canonical = ref(bundle["canonical_source_freeze"])
assert bundle["canonical_source_freeze"]["path"] == FREEZE
numbered = ref(bundle["section_freeze"])
assert numbered["canonical_source_freeze"] == bundle["canonical_source_freeze"]
manifest = read(MANIFEST)
assert manifest["source_freeze"] == bundle["canonical_source_freeze"]
assert manifest["numbered_section_reviews"] == reference(BUNDLE)
assignments = ref(manifest["assignments"])
tasks = {x["task"]: x for x in assignments["tasks"]}
excluded = set(manifest["author_tasks"]) | {OWNER} | {x["task"] for x in assignments["tasks"] if x["role"] == "reader"}
collection = ref(bundle["actual_collection_snapshot"])
raw_collection = ref(collection["raw_collection_snapshot"])
assert collection["role"] == "technical" and collection["status"] == "in_progress"
expected = {f"{p}.{i}" for p,n in [("R",4),("W",7),("T",11),("G",4)] for i in range(1,n+1)}
assert collection["collection_index"] == [x for x in raw_collection["collection_index"] if x["lesson_id"] in expected]
assert len(collection["collection_index"]) == len(collection["report_sha256s"]) == 45
assert set(collection["report_sha256s"]) == {x["sha256"] for x in collection["collection_index"]}
assert collection["notified_findings"] == raw_collection["notified_findings"]

spec = importlib.util.spec_from_file_location("glossary_current3_reading", ROOT / "scripts/reading_time.py")
reading = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reading)
pages = {p["page_id"]:p for p in canonical["pages"]}
old_canonical = ref(old_bundle["canonical_source_freeze"])
old_pages = {p["page_id"]:p for p in old_canonical["pages"]}
assert set(pages) == set(old_pages)
source_changes = [pid for pid in pages if any(pages[pid][f] != old_pages[pid][f] for f in ["source_sha256", "figures_sha256"])]
assert source_changes == ["12.2"]
new12, old12 = pages["12.2"], old_pages["12.2"]
new12text, old12text = (ROOT/new12["snapshot"]).read_text(), (ROOT/old12["snapshot"]).read_text()
assert digest(new12["snapshot"]) == new12["source_sha256"]
assert digest(old12["snapshot"]) == old12["source_sha256"]
assert re.findall(r"```python\n(.*?)\n```", new12text, re.S) == re.findall(r"```python\n(.*?)\n```", old12text, re.S)
changed12 = [{"line":i,"before":a,"after":b} for i,(a,b) in enumerate(zip(old12text.splitlines(),new12text.splitlines()),1) if a!=b]
assert len(changed12) == 1 and "也把兩行print中的頻率名稱改成9500Hz、6500Hz" in changed12[0]["after"]
current_slices = {}
for pid in numbered["canonical_page_ids"]:
    p=pages[pid]
    assert digest(p["snapshot"]) == p["source_sha256"]
    current_slices.update({sid:(pid,body) for sid,body in reading.lesson_slices((ROOT/p["snapshot"]).read_text()).items()})
entries = {e["lesson_id"]:e for e in bundle["sections"]}
previous = {e["lesson_id"]:e for e in old_bundle["sections"]}
section_pages = {p["page_id"]:p for p in numbered["pages"]}
assert set(entries) == set(section_pages) == set(current_slices) == expected
changed_entries = [sid for sid in entries if entries[sid] != previous[sid]]
assert set(changed_entries) == {"R.1", "W.1"}
report_refs, issues, owners, records = {}, [], set(), []
artifact_checks = 0
for sid,e in entries.items():
    current, original = ref(e["report"]), ref(e["original_report"])
    assert e["original_report"] == previous[sid]["original_report"]
    assert e["original_section_freeze"] == previous[sid]["original_section_freeze"]
    ref(e["original_section_freeze"])
    pid, exact = current_slices[sid]
    p = section_pages[sid]
    assert (ROOT/p["snapshot"]).read_text() == exact
    assert digest(p["snapshot"]) == p["source_sha256"] == current["source_sha256"]
    assert p["figures_sha256"] == current["figure_sha256"]
    assert current["reviewer_task"] == original["reviewer_task"]
    task = current["reviewer_task"]
    assert task not in excluded
    assigned = tasks[task]
    assert assigned["role"] == "technical" and assigned["context"] == "fresh" and assigned["lesson_id"] == sid
    assert current["lesson_id"] == original["lesson_id"] == sid
    assert current["reviewer_context"] == original["reviewer_context"] == "fresh"
    assert utc(assigned["dispatch_window"]["not_before"]) <= utc(original["created_at"]) <= utc(current["created_at"])
    assert current["verdict"] == "pass" and all(c["status"] == "verified" for c in current["claims"])
    if "current_dispatch_window" in e:
        assert utc(e["current_dispatch_window"]["not_before"]) <= utc(current["created_at"])
        assert any(x.get("task") == task and x.get("dispatch_window") == e["current_dispatch_window"] for x in assignments.get("callback_dispatches",[]))
    for report_key, collection_key, report in [("original_report","original_collection",original),("report","current_collection",current)]:
        rr=e[report_key];receipt=e[collection_key]
        matches=[x for x in collection["collection_index"] if x["path"]==rr["path"] and x["sha256"]==rr["sha256"] and x["received_at"]==receipt["received_at"]]
        assert len(matches)==1 and matches[0]["reviewer_task"]==task and matches[0]["created_at"]==report["created_at"]
        assert matches[0]["collection_role"]==receipt["collection_role"]=="technical"
        assert utc(report["created_at"])<=utc(receipt["received_at"])
        if "reviewer_self_check_receipt" in matches[0]:
            check_ref=matches[0]["reviewer_self_check_receipt"]
            assert digest(check_ref["path"])==check_ref["sha256"]
        report_refs[rr["path"]]=rr
        for a in report["artifacts"]:
            assert digest(a["path"])==a["sha256"]
            artifact_checks+=1
        for i in range(len(report.get("issues",[]))):
            row={"report":rr,"json_pointer":f"/issues/{i}","lesson_id":sid,"canonical_page_id":pid}
            if row not in issues:issues.append(row)
    for historic in e.get("original_source_snapshots",[]):
        snap=historic["snapshot"]
        assert digest(snap["path"])==snap["sha256"]
        oldsource=next(s for s in original["sources"] if s.get("path")==historic["source"])
        assert oldsource["sha256"]==snap["sha256"]
        newsource=next(s for s in current["sources"] if s.get("path")==historic["source"])
        assert digest(newsource["path"])==newsource["sha256"]!=snap["sha256"]
    owners.add(task)
    records.append({"lesson_id":sid,"report":e["report"],"original_report":e["original_report"],"reviewer_task":task,
                    "source_sha256":current["source_sha256"],"current_received_at":e["current_collection"]["received_at"],
                    "current_and_original_identity_source_receipt_checks":True})
assert len(owners)==26 and len(report_refs)==35 and len(issues)==5
index=ref(bundle["issue_inventory"])
assert {x["path"]:x for x in index["reports"]}==report_refs
sortkey=lambda x:(x["report"]["path"],x["json_pointer"])
assert sorted(index["issues"],key=sortkey)==sorted(issues,key=sortkey)
preserved_indices=[]
for rr in bundle["preserved_intermediate_issue_inventories"]:
    old=ref(rr)
    preserved_indices.append({**rr,"reports":len(old["reports"]),"issues":len(old["issues"])})
discovery=[]
for rr in bundle["preserved_current_discovery_reports"]:
    old=ref(rr);new=ref(entries[old["lesson_id"]]["report"])
    assert old["reviewer_task"]==new["reviewer_task"]
    for field in ["id","details","source_quote"]:assert old["issues"][0][field]==new["issues"][0][field]
    assert new["issues"][0]["status"]=="resolved"
    discovery.append({**rr,"old_verdict":old["verdict"],"old_issue_status":old["issues"][0]["status"],"current_status":"resolved"})
new_report_source_checks=[]
for sid in changed_entries:
    new=ref(entries[sid]["report"])
    assert new["previous_current_report"]==previous[sid]["report"]
    assert new["original_issue_resolutions"]==[] and ref(entries[sid]["original_report"])["issues"]==[]
    for s in new["sources"]:
        if s["kind"]=="repository_code":
            assert digest(s["path"])==s["sha256"]
            new_report_source_checks.append({"lesson_id":sid,"source_id":s["id"],"path":s["path"],"sha256":s["sha256"]})
w1=ref(entries["W.1"]["report"])
invalid=w1["superseded_format_invalid_attempt"]
ref(invalid["report"]);ref(invalid["actual_failed_schema_receipt"])

prior=read(RD+"/technical-current-v2.json")
assert digest(RD+"/technical-current-v2.json")=="f84d63c0934b2db16b823e1f0f50b4c662f119d526f8437b7d9e7e9aeb8b8680"
gl=pages["glossary"]
assert digest(gl["snapshot"])==digest("course/glossary.md")==prior["source_sha256"]==gl["source_sha256"]
assert gl["figures_sha256"]==prior["figure_sha256"]=={}
text=(ROOT/gl["snapshot"]).read_text()
for i,line in enumerate([42,108,198,28,107,223]):
    assert prior["original_issue_resolutions"][i]["json_pointer"]==f"/issues/{i}"
    assert prior["original_issue_resolutions"][i]["current_source_quote"]==text.splitlines()[line-1]
    assert prior["issues"][i]["status"]=="resolved"
source_checks=[]
for s in prior["sources"]:
    if s["kind"]=="repository_code":
        assert digest(s["path"])==s["sha256"]
        source_checks.append({"source_id":s["id"],"path":s["path"],"sha256":s["sha256"]})
for a in prior["artifacts"]:assert digest(a["path"])==a["sha256"]
links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)",text)
for label,destination in links:
    path,_,fragment=destination.partition("#");target=ROOT/"course"/path
    assert target.is_file() and (not fragment or re.search(r"^##\s+"+re.escape(fragment)+r"\s",target.read_text(),re.M))
assert len(links)==455
assert all(entries[sid]==previous[sid] for sid in ["G.1","G.2","G.3","G.4"])

author_path="docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json"
author=read(author_path)
assert author["source_freeze"]==bundle["canonical_source_freeze"]
assert author["reexecuted_this_fix"]==["12.2"] and author["reused_matching_cells"]==330 and author["total"]==331
method_counts=Counter()
for row in author["records"]:
    method_counts[row["method"]]+=1
    assert digest(row["source"])==digest(row["bundle_path"])==row["executed_sha256"]
    suffix=Path(row["bundle_path"]).relative_to(ROOT/"outputs/omni-notebooks-technical-fix14-current")
    authored=ROOT/"notebooks"/suffix
    assert digest(authored)==row["notebook_sha256"]
    aa,bb=read(authored),read(row["bundle_path"])
    assert [(c["cell_type"],c["source"]) for c in aa["cells"]]==[(c["cell_type"],c["source"]) for c in bb["cells"]]
assert sum(method_counts.values())==331
kernel_path="outputs/omni-notebooks-technical-fix14-run/validation-kernel.json"
kernel=read(kernel_path)
assert [kernel[k] for k in ["total","passed","failed"]]==[1,1,0]
new_record=next(x for x in author["records"] if x["lesson"]=="12.2")

preview_path="outputs/omni-reading-times-technical-fix14-preview.json"
time_data=read("course/reading-times.json");preview=read(preview_path)
valid=[r for r in time_data["pages"] if r["source_sha256"]==pages[r["page_id"]]["source_sha256"] and r["figures_sha256"]==pages[r["page_id"]]["figures_sha256"]]
assert len(time_data["pages"])==369 and len(valid)==367 and preview["pages"]==valid
estimates=reading.load_estimates(ROOT/preview_path,canonical,require_complete=False)
missing=set(pages)-set(estimates)
assert len(estimates)==367 and missing=={"first-steps","12.2"}
build_path="outputs/omni-review-site-technical-fix14/build-info.json"
build=read(build_path)
assert build["reading_time_pages"]==367 and build["reading_time_complete"] is False
assert build["publication_ready"] is False and build["colab_git_objects_verified"] is False and build["colab_mode"]=="upload"
HTTP=[]
for url,expected_path in [("http://127.0.0.1:8775/build-info.json",build_path)]:
    with urllib.request.urlopen(url,timeout=20) as response:raw=response.read();status=response.status
    assert status==200 and hashlib.sha256(raw).hexdigest()==digest(expected_path)
    HTTP.append({"url":url,"status":status,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
with urllib.request.urlopen("http://127.0.0.1:8775/12.2.html",timeout=20) as response:html=response.read().decode()
href=next(h for h in re.findall(r'href=[\"\']([^\"\']+)[\"\']',html) if h.split('?')[0].endswith('.ipynb'))
url=urllib.parse.urljoin("http://127.0.0.1:8775/12.2.html",href)
with urllib.request.urlopen(url,timeout=20) as response:raw=response.read();status=response.status
assert status==200 and hashlib.sha256(raw).hexdigest()==digest("notebooks/12/12.2.ipynb")
HTTP.append({"url":url,"status":status,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"scope":"One real current source14 download; not a new331HTTP or browser/UI audit."})

result={
    "created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":OWNER,
    "command":"uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/audit-current-v3.py",
    "environment":{"python":platform.python_version(),"device":"CPU metadata/text/source checks and bounded localHTTP; no model or lesson execution"},
    "bundle":reference(BUNDLE),"issue_inventory":bundle["issue_inventory"],"canonical_source_freeze":bundle["canonical_source_freeze"],"section_freeze":bundle["section_freeze"],
    "previous_bundle":bundle["previous_bundle"],"previous_whole_report":reference(RD+"/technical-current-v2.json"),
    "source_changes_since13":source_changes,"new12_2":{"old_snapshot":{"path":old12["snapshot"],"sha256":old12["source_sha256"]},"new_snapshot":{"path":new12["snapshot"],"sha256":new12["source_sha256"]},"changed_lines":changed12,"Python_fences_unchanged":True,
                  "personal_semantic_assessment":"本次親讀唯一變動與12.2全文：補練習print標籤，不改每秒點數的定義、程式值或GL的sample_rate條目。GL科學來源沿用，不代簽12.2完整取樣理論/reader。"},
    "changed_sections_in_bundle":changed_entries,"section_count":26,"unique_owners":26,"report_refs":35,"issue_pointers":5,"sections":records,
    "selected_collection":bundle["actual_collection_snapshot"],"raw_collection":collection["raw_collection_snapshot"],
    "selected_rows":45,"raw_rows":len(raw_collection["collection_index"]),"selection_verbatim_filter_of_raw":True,"all_report_artifact_checks":artifact_checks,
    "preserved_issue_inventories":preserved_indices,"preserved_current_discovery_chain":discovery,"W1_schema_failed_attempt_preserved":invalid,
    "current_R1_W1_repository_source_fingerprints":new_report_source_checks,
    "G1_G4_reports_slices_figures_unchanged":True,"glossary":{"source_sha256":gl["source_sha256"],"figures_sha256":{},"all_context_and201_claims_reused_after_byte_identity":True,"six_original_issue_quotes_checked":6,"local_links_checked":455,"index_terms_reused_unchanged":194},
    "unchanged_scientific_evidence_reuse":{"prior_audit":reference(OUT.relative_to(ROOT)/"collector-current-audit-v2.json"),"source_ids_reused":[s["id"] for s in prior["sources"]],"actual_current_repository_checks":source_checks,
         "note":"既有本人primary實讀、數值probe、HTTPS及其artifact SHA重新核，未變科學來源明確reuse；不冒稱本次再取原文或再跑數值/模型。"},
    "author_kernel_and_reuse_identity_check":{"receipt":reference(author_path),"total":331,"methods":dict(method_counts),"newly_executed_by_author":["12.2"],"matching_reuse":330,"all_source_cells_type_and_source_match_current_authored":True,"new12_2_record":new_record,"actual_validation_receipt":reference(kernel_path),"author_result":kernel,"scope":"本人核既存成功kernel與原件/沿用身分，未再次執行kernel，亦不以此簽其他課程科學。"},
    "preview_estimate_identity_check":{"formal_metadata":reference("course/reading-times.json"),"preview_metadata":reference(preview_path),"original_rows":369,"exact_current_rows":367,"omitted_stale":sorted(missing),"retained_rows_exact_equal":True,"actual_current_load_estimates_count":len(estimates),"build_info":reference(build_path),"HTTP_samples":HTTP},
    "reader_boundary":{"owner_reported_source_identity_rows":363,"source":"R1 current4 reader_mapping metadata, not independently accepted fullreader by this collector","proper_wave02_fullreader_pending_at_bundle_intake":["first-steps","12.2"],"reader_rows_are_not367_AI_estimates":True},
    "intake":{"manifest":reference(MANIFEST),"own_result":reference(OUT.relative_to(ROOT)/"numbered-intake-current-v4.json"),"exit_code":0,"status":"schema_checked_collection_pending","scope":26,"coordinator_receipt":reference("docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-execution-v5.json")},
    "manual_context_review":"重新核全頁前言/四節上下文、六修正原句、source14唯一新前置和兩歷史主題；其餘未變正文及201claims沿本人先前完整實讀與原始證據。",
    "not_claimed":["R/W/T科學事實代簽","363全reader驗收或367估時重新估算","properwave02已完成","210工具審查或dependency audit","新訓練/能力、GPU、付費、部署、Colab執行","整體stage/seal或global gatepass"],
}
assert read(OUT.relative_to(ROOT)/"numbered-intake-current-v4.json")["status"]=="schema_checked_collection_pending"
path=OUT/"collector-current-audit-v3.json"
with path.open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
print(json.dumps({"output":reference(path.relative_to(ROOT)),"canonical_changes":source_changes,"bundle_changes":changed_entries,
                  "section_owners_reports_issues":[26,26,35,5],"selected_raw_collection_rows":[45,len(raw_collection["collection_index"])],
                  "exact_current_estimates":367,"reader_metadata_rows":363,"intake":"schema_checked_collection_pending"},ensure_ascii=False))
