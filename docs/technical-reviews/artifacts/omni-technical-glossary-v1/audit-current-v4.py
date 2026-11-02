"""Source15/bundle5 bounded callback; reuse unchanged personally read science.

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
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v5.json"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v6.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"


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
assert digest(BUNDLE) == "0077db6122efdd4371d02334fa8aab5ab7002cd3312431b48fc57660bcb66d4f"
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
assert collection["collection_index"] == [x for x in raw_collection["collection_index"] if x.get("lesson_id") in expected]
assert len(collection["collection_index"]) == len(collection["report_sha256s"]) == 48
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
assert set(source_changes)=={"5.7","chapter-10","chapter-14"}
changes=[]
judgments={
    "5.7":"親讀完整新5.7與save/load/RNG helper及實裝官方PyTorch source。修句收緊weights_only載入宣稱，未改GL checkpoint定義所需參數/設定/optimizer/RNG/資料進度條件；GL沒有原不執行任意程式的絕對承諾。",
    "chapter-10":"親讀新導讀與GL圖片/影格上下文：明列10.9–10.11配對、10.12時間戳，支持GL把影格索引與時間分開；未改編碼器/projector/patch本體或其理論。",
    "chapter-14":"親讀新導讀與GL context/RoPE/Looped/TMRoPE上下文：前置指向完整TinyLM4.6並補14.19；未改位置/共享參數定義或以座標能力保證長文品質。",
}
for pid in source_changes:
    before,after=old_pages[pid],pages[pid]
    assert digest(before["snapshot"])==before["source_sha256"]
    assert digest(after["snapshot"])==after["source_sha256"]
    oldtext,newtext=(ROOT/before["snapshot"]).read_text(),(ROOT/after["snapshot"]).read_text()
    if pid=="5.7":
        assert re.findall(r"```python\n(.*?)\n```",oldtext,re.S)==re.findall(r"```python\n(.*?)\n```",newtext,re.S)
        assert "限制可還原的型別與可呼叫的函式" in newtext
    changes.append({"page_id":pid,"old_snapshot":{"path":before["snapshot"],"sha256":before["source_sha256"]},"new_snapshot":{"path":after["snapshot"],"sha256":after["source_sha256"]},"changed_lines":[{"line":i,"before":a,"after":b} for i,(a,b) in enumerate(zip(oldtext.splitlines(),newtext.splitlines()),1) if a!=b],"personal_semantic_assessment":judgments[pid]})
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
assert set(changed_entries) == {"R.1", "W.1", "T.2"}
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
        current_paths={s.get("path"):s for s in current["sources"] if s["kind"]=="repository_code"}
        if historic["source"] in current_paths:
            newsource=current_paths[historic["source"]]
            assert digest(newsource["path"])==newsource["sha256"]!=snap["sha256"]
        else:
            assert sid=="T.2" and historic["source"] in {"course/chapters/05.md","course/chapters/10.md"}
            assert all(digest(s["path"])==s["sha256"] for s in current_paths.values())
    owners.add(task)
    records.append({"lesson_id":sid,"report":e["report"],"original_report":e["original_report"],"reviewer_task":task,
                    "source_sha256":current["source_sha256"],"current_received_at":e["current_collection"]["received_at"],
                    "current_and_original_identity_source_receipt_checks":True})
assert len(owners)==26 and len(report_refs)==36 and len(issues)==5
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
    if sid in {"R.1","W.1"}:
        assert new["previous_current_report"]==previous[sid]["report"]
    else:
        assert sid=="T.2" and previous[sid]["report"]==entries[sid]["original_report"]
    assert new["original_issue_resolutions"]==[] and ref(entries[sid]["original_report"])["issues"]==[]
    for s in new["sources"]:
        if s["kind"]=="repository_code":
            assert digest(s["path"])==s["sha256"]
            new_report_source_checks.append({"lesson_id":sid,"source_id":s["id"],"path":s["path"],"sha256":s["sha256"]})
t2proof=read("docs/technical-reviews/artifacts/omni-technical-t-2-v1/callback-source-drift-v1.json")
assert t2proof["current_canonical_freeze"]==bundle["canonical_source_freeze"]
assert t2proof["original_report"]==entries["T.2"]["original_report"]
for item in t2proof["current_exact_prerequisites"]:
    assert digest(item["snapshot"]["path"])==item["snapshot"]["sha256"]==pages[item["page_id"]]["source_sha256"]
assert len(t2proof["all_original_claim_dispositions"])==len(ref(entries["T.2"]["original_report"])["claims"])
assert all(item["status"]=="still_supported" for item in t2proof["all_original_claim_dispositions"])
w1=ref(entries["W.1"]["report"])
invalid=w1["superseded_format_invalid_attempt"]
ref(invalid["report"]);ref(invalid["actual_failed_schema_receipt"])

prior=read(RD+"/technical-current-v3.json")
assert digest(RD+"/technical-current-v3.json")=="fc9c7ef1a0e7d0a69946a842a21c3ce669b062604d28afaf54b8a7a1b2d48efc"
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

author_path="docs/course-revision-20261029-omni/evidence/notebooks-technical-fix15-cumulative.json"
author=read(author_path)
assert author["source_freeze"]==bundle["canonical_source_freeze"]
assert author["reexecuted_this_fix"]==["5.7"] and author["reused_matching_cells"]==330 and author["total"]==331
method_counts=Counter()
for row in author["records"]:
    method_counts[row["method"]]+=1
    assert digest(row["source"])==digest(row["bundle_path"])==row["executed_sha256"]
    suffix=Path(row["bundle_path"]).relative_to(ROOT/"outputs/omni-notebooks-technical-fix15-current")
    authored=ROOT/"notebooks"/suffix
    assert digest(authored)==row["notebook_sha256"]
    aa,bb=read(authored),read(row["bundle_path"])
    assert [(c["cell_type"],c["source"]) for c in aa["cells"]]==[(c["cell_type"],c["source"]) for c in bb["cells"]]
assert sum(method_counts.values())==331
kernel_path="outputs/omni-notebooks-technical-fix15-run/validation-kernel.json"
kernel=read(kernel_path)
assert [kernel[k] for k in ["total","passed","failed"]]==[1,1,0]
new_record=next(x for x in author["records"] if x["lesson"]=="5.7")

preview_path="outputs/omni-reading-times-technical-fix15-preview.json"
time_data=read("course/reading-times.json");preview=read(preview_path)
valid=[r for r in time_data["pages"] if r["source_sha256"]==pages[r["page_id"]]["source_sha256"] and r["figures_sha256"]==pages[r["page_id"]]["figures_sha256"]]
assert len(time_data["pages"])==369 and len(valid)==364 and preview["pages"]==valid
estimates=reading.load_estimates(ROOT/preview_path,canonical,require_complete=False)
missing=set(pages)-set(estimates)
assert len(estimates)==364 and missing=={"first-steps","12.2","5.7","chapter-10","chapter-14"}
build_path="outputs/omni-review-site-technical-fix15/build-info.json"
build=read(build_path)
assert build["reading_time_pages"]==364 and build["reading_time_complete"] is False
assert build["publication_ready"] is False and build["colab_git_objects_verified"] is False and build["colab_mode"]=="upload"
HTTP=[]
for url,expected_path in [("http://127.0.0.1:8776/build-info.json",build_path)]:
    with urllib.request.urlopen(url,timeout=20) as response:raw=response.read();status=response.status
    assert status==200 and hashlib.sha256(raw).hexdigest()==digest(expected_path)
    HTTP.append({"url":url,"status":status,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
with urllib.request.urlopen("http://127.0.0.1:8776/5.7.html",timeout=20) as response:html=response.read().decode()
href=next(h for h in re.findall(r'href=[\"\']([^\"\']+)[\"\']',html) if h.split('?')[0].endswith('.ipynb'))
url=urllib.parse.urljoin("http://127.0.0.1:8776/5.7.html",href)
with urllib.request.urlopen(url,timeout=20) as response:raw=response.read();status=response.status
assert status==200 and hashlib.sha256(raw).hexdigest()==digest("notebooks/05/5.7.ipynb")
HTTP.append({"url":url,"status":status,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"scope":"One real current source15 download; not a new331HTTP or browser/UI audit."})

result={
    "created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":OWNER,
    "command":"uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/audit-current-v4.py",
    "environment":{"python":platform.python_version(),"device":"CPU metadata/text/source checks and bounded localHTTP; no model or lesson execution"},
    "bundle":reference(BUNDLE),"issue_inventory":bundle["issue_inventory"],"canonical_source_freeze":bundle["canonical_source_freeze"],"section_freeze":bundle["section_freeze"],
    "previous_bundle":bundle["previous_bundle"],"previous_whole_report":reference(RD+"/technical-current-v3.json"),
    "source_changes_since14":source_changes,"changed_prerequisite_semantics":changes,
    "new_checkpoint_primary_context":reference(OUT.relative_to(ROOT)/"source15-checkpoint-context-primary-v1.json"),
    "changed_sections_in_bundle":changed_entries,"section_count":26,"unique_owners":26,"report_refs":36,"issue_pointers":5,"sections":records,
    "selected_collection":bundle["actual_collection_snapshot"],"raw_collection":collection["raw_collection_snapshot"],
    "selected_rows":48,"raw_rows":len(raw_collection["collection_index"]),"selection_verbatim_filter_of_raw":True,"all_report_artifact_checks":artifact_checks,
    "preserved_issue_inventories":preserved_indices,"preserved_current_discovery_chain":discovery,"W1_schema_failed_attempt_preserved":invalid,
    "current_R1_W1_T2_repository_source_fingerprints":new_report_source_checks,"T2_current_scope_binding":reference("docs/technical-reviews/artifacts/omni-technical-t-2-v1/callback-source-drift-v1.json"),
    "G1_G4_reports_slices_figures_unchanged":True,"glossary":{"source_sha256":gl["source_sha256"],"figures_sha256":{},"all_context_and201_claims_reused_after_byte_identity":True,"six_original_issue_quotes_checked":6,"local_links_checked":455,"index_terms_reused_unchanged":194},
    "unchanged_scientific_evidence_reuse":{"prior_audit":reference(OUT.relative_to(ROOT)/"collector-current-audit-v3.json"),"source_ids_reused":[s["id"] for s in prior["sources"]],"actual_current_repository_checks":source_checks,
         "note":"既有本人primary實讀、數值probe、HTTPS及其artifact SHA重新核，未變科學來源明確reuse；不冒稱本次再取原文或再跑數值/模型。"},
    "author_kernel_and_reuse_identity_check":{"receipt":reference(author_path),"total":331,"methods":dict(method_counts),"newly_executed_by_author":["5.7"],"matching_reuse":330,"all_source_cells_type_and_source_match_current_authored":True,"new5_7_record":new_record,"actual_validation_receipt":reference(kernel_path),"author_result":kernel,"scope":"本人核既存成功kernel與原件/沿用身分，未再次執行kernel，亦不以此簽其他課程科學。"},
    "preview_estimate_identity_check":{"formal_metadata":reference("course/reading-times.json"),"preview_metadata":reference(preview_path),"original_rows":369,"exact_current_rows":364,"omitted_stale":sorted(missing),"retained_rows_exact_equal":True,"actual_current_load_estimates_count":len(estimates),"build_info":reference(build_path),"HTTP_samples":HTTP},
    "reader_boundary":{"owner_reported_source_identity_rows":360,"source":"R1 current5 reader_mapping metadata, not independently accepted fullreader by this collector","proper_wave02_fullreader_pending_at_bundle_intake":["first-steps","12.2","5.7","chapter-10","chapter-14"],"reader_source_identity_count_is_not_AI_estimate_count":True},
    "intake":{"manifest":reference(MANIFEST),"own_result":reference(OUT.relative_to(ROOT)/"numbered-intake-current-v5.json"),"exit_code":0,"status":"schema_checked_collection_pending","scope":26,"coordinator_receipt":reference("docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-execution-v6.json")},
    "manual_context_review":"本人同一collector重核全頁前言/四節上下文、六修正原句、完整5.7與三個source15變動，及UTF8/PPO兩歷史主題。未變正文/201claims/原primary和實跑真reuse，不冒稱fresh全重讀重跑。",
    "not_claimed":["R/W/T科學事實代簽","全reader驗收或估時重新估算","properwave02已完成","210工具審查或dependency audit","新訓練/能力、GPU、付費、部署、Colab執行","整體stage/seal或global gatepass"],
}
assert read(OUT.relative_to(ROOT)/"numbered-intake-current-v5.json")["status"]=="schema_checked_collection_pending"
path=OUT/"collector-current-audit-v4.json"
with path.open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
print(json.dumps({"output":reference(path.relative_to(ROOT)),"canonical_changes":source_changes,"bundle_changes":changed_entries,
                  "section_owners_reports_issues":[26,26,36,5],"selected_raw_collection_rows":[48,len(raw_collection["collection_index"])],
                  "exact_current_estimates":364,"reader_metadata_rows":360,"intake":"schema_checked_collection_pending"},ensure_ascii=False))
