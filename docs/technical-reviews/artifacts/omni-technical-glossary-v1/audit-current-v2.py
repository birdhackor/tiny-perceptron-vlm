"""Bounded same-owner fix13 callback: new collection data, unchanged glossary facts.

This audits exact bytes and provenance. Manual semantic findings are recorded
separately below; no R/W/T factual approval or completed stage seal is implied.
All prior reports, failed intake outputs and original evidence remain untouched.
"""

import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OWNER = "/root/omni_technical_coordinator/technical_glossary"
REPORT_DIR = "docs/course-revision-20261029-omni/reviews/technical/technical_glossary"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v3.json"
OLD_BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v1.json"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v4.json"
INTAKE = OUT.relative_to(ROOT).as_posix() + "/numbered-intake-current-v3.json"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def reference(path):
    return {"path": str(path), "sha256": digest(path)}


def ref(item):
    assert digest(item["path"]) == item["sha256"], item
    return read(item["path"])


def exact_slices(canonical, page_ids):
    result = {}
    for page in canonical["pages"]:
        if page["page_id"] not in page_ids:
            continue
        raw = (ROOT / page["snapshot"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
        text = raw.decode("utf-8")
        headers = list(re.finditer(r"^## ([^\n]+)\n", text, re.M))
        for n, header in enumerate(headers):
            sid = header[1].split()[0]
            if re.fullmatch(r"[RWTG]\.\d+", sid):
                end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
                result[sid] = (page["page_id"], text[header.start():end].encode())
    return result


def utc(value):
    result = datetime.fromisoformat(value)
    assert result.utcoffset() is not None
    return result


bundle = read(BUNDLE)
assert digest(BUNDLE) == "d8c8015797dff2e030b211499822873f236d54f5c6025f0ac9e08fce4218adba"
assert bundle["canonical_source_freeze"]["path"] == FREEZE
canonical = ref(bundle["canonical_source_freeze"])
numbered = ref(bundle["section_freeze"])
assert numbered["canonical_source_freeze"] == bundle["canonical_source_freeze"]
manifest = read(MANIFEST)
assert manifest["source_freeze"] == bundle["canonical_source_freeze"]
assert manifest["numbered_section_reviews"] == reference(BUNDLE)
assignments = ref(manifest["assignments"])
collection = ref(bundle["actual_collection_snapshot"])
assert collection["role"] == "technical" and collection["status"] == "in_progress"
assert read(INTAKE) == {
    "status": "schema_checked_collection_pending", "sections": 26,
    "canonical_pages": ["course", "first-steps", "training", "glossary"],
}
old_bundle = read(OLD_BUNDLE)
ref(bundle["previous_bundle"])
expected = {f"{p}.{i}" for p, n in [("R", 4), ("W", 7), ("T", 11), ("G", 4)] for i in range(1, n + 1)}
slices = exact_slices(canonical, numbered["canonical_page_ids"])
pages = {p["page_id"]: p for p in numbered["pages"]}
entries = {e["lesson_id"]: e for e in bundle["sections"]}
assert set(entries) == set(pages) == set(slices) == expected
assert len(entries) == len(bundle["sections"]) == 26
tasks = {t["task"]: t for t in assignments["tasks"]}
assert len(tasks) == len(assignments["tasks"])
readers = {t["task"] for t in assignments["tasks"] if t["role"] == "reader"}
excluded = readers | set(manifest["author_tasks"]) | {OWNER}
refs = {}
issues = []
owners = set()
records = []
snapshot_records = []
collection_receipts = []
report_artifact_checks = 0
for sid, entry in entries.items():
    current = ref(entry["report"])
    original = ref(entry["original_report"])
    original_freeze = ref(entry["original_section_freeze"])
    original_page = next(p for p in original_freeze["pages"] if p["page_id"] == sid)
    original_canonical = ref(original_freeze["canonical_source_freeze"])
    original_exact = exact_slices(original_canonical, original_freeze["canonical_page_ids"])[sid][1]
    pid, current_exact = slices[sid]
    page = pages[sid]
    assert (ROOT / original_page["snapshot"]).read_bytes() == original_exact
    assert hashlib.sha256(original_exact).hexdigest() == original["source_sha256"] == original_page["source_sha256"]
    assert (ROOT / page["snapshot"]).read_bytes() == current_exact
    assert hashlib.sha256(current_exact).hexdigest() == current["source_sha256"] == page["source_sha256"]
    assert current["figure_sha256"] == page["figures_sha256"]
    assert original["figure_sha256"] == original_page["figures_sha256"]
    owner = current["reviewer_task"]
    assert owner == original["reviewer_task"] and owner not in excluded
    assigned = tasks[owner]
    assert assigned["role"] == "technical" and assigned["lesson_id"] == sid and assigned["context"] == "fresh"
    assert original["lesson_id"] == current["lesson_id"] == sid
    assert original["reviewer_context"] == current["reviewer_context"] == "fresh"
    assert utc(assigned["dispatch_window"]["not_before"]) <= utc(original["created_at"])
    assert utc(original["created_at"]) <= utc(current["created_at"])
    assert utc(entry["original_collection"]["received_at"]) <= utc(entry["current_collection"]["received_at"])
    if "current_dispatch_window" in entry:
        assert utc(entry["current_dispatch_window"]["not_before"]) <= utc(current["created_at"])
        assert any(t.get("task") == owner and t.get("dispatch_window") == entry["current_dispatch_window"]
                   for t in assignments.get("callback_dispatches", []))
    assert current["verdict"] == "pass" and all(c["status"] == "verified" for c in current["claims"])
    assert all(i.get("status") == "resolved" and i.get("resolution") for i in current.get("issues", []))
    assert {x["json_pointer"] for x in current.get("original_issue_resolutions", [])} == {
        f"/issues/{i}" for i in range(len(original.get("issues", [])))
    }
    for role, report_key in [("original_collection", "original_report"), ("current_collection", "report")]:
        report_ref = entry[report_key]
        report = original if report_key == "original_report" else current
        assert entry[role]["collection_role"] == "technical"
        assert utc(report["created_at"]) <= utc(entry[role]["received_at"])
        matches = [x for x in collection["collection_index"] if x["path"] == report_ref["path"]
                   and x["sha256"] == report_ref["sha256"] and x["received_at"] == entry[role]["received_at"]]
        assert len(matches) == 1, (sid, report_key)
        received = matches[0]
        assert received["reviewer_task"] == owner and received["lesson_id"] == sid
        assert received["created_at"] == report["created_at"] and received["collection_role"] == "technical"
        assert received["verdict"] == report["verdict"]
        assert report_ref["sha256"] in collection["report_sha256s"]
        if "reviewer_self_check_receipt" in received:
            receipt = received["reviewer_self_check_receipt"]
            assert digest(receipt["path"]) == receipt["sha256"]
            collection_receipts.append(receipt)
        refs[report_ref["path"]] = report_ref
        for a in report["artifacts"]:
            assert digest(a["path"]) == a["sha256"]
            report_artifact_checks += 1
        for i in range(len(report.get("issues", []))):
            issue = {"report": report_ref, "json_pointer": f"/issues/{i}", "lesson_id": sid, "canonical_page_id": pid}
            if issue not in issues:
                issues.append(issue)
    for previous in entry.get("previous_current_reports", []):
        previous_report = ref(previous)
        assert previous_report["reviewer_task"] == owner
        assert utc(original["created_at"]) <= utc(previous_report["created_at"]) <= utc(current["created_at"])
    for historical in entry.get("original_source_snapshots", []):
        snapshot = historical["snapshot"]
        assert digest(snapshot["path"]) == snapshot["sha256"]
        source = next(s for s in original["sources"] if s.get("path") == historical["source"] and s["kind"] == "repository_code")
        assert source["sha256"] == snapshot["sha256"]
        current_source = next(s for s in current["sources"] if s.get("path") == historical["source"] and s["kind"] == "repository_code")
        assert digest(current_source["path"]) == current_source["sha256"]
        assert current_source["sha256"] != snapshot["sha256"]
        snapshot_records.append({"lesson_id": sid, "original_source_id": source["id"], **historical,
                                 "current_source_sha256": current_source["sha256"], "current_exemption_used": False})
    owners.add(owner)
    records.append({"lesson_id": sid, "reviewer_task": owner, "canonical_page_id": pid,
                    "report": entry["report"], "original_report": entry["original_report"],
                    "source_sha256": current["source_sha256"], "original_source_sha256": original["source_sha256"],
                    "original_received_at": entry["original_collection"]["received_at"],
                    "current_received_at": entry["current_collection"]["received_at"],
                    "exact_source_figures_owner_assignment_collection_and_time": True})
assert len(owners) == 26 and len(refs) == 35 and len(issues) == 5
assert {r["lesson_id"] for r in snapshot_records} == {"R.1", "W.1", "W.7"}
inventory = ref(bundle["issue_inventory"])
assert inventory["canonical_source_freeze"] == bundle["canonical_source_freeze"]
assert {r["path"]: r for r in inventory["reports"]} == refs
key = lambda x: (x["report"]["path"], x["json_pointer"])
assert sorted(inventory["issues"], key=key) == sorted(issues, key=key)
preserved_inventory_records = []
for item in bundle["preserved_intermediate_issue_inventories"]:
    old = ref(item)
    for r in old["reports"]:
        ref(r)
    for issue in old["issues"]:
        report = ref(issue["report"])
        assert report["issues"][int(issue["json_pointer"].split("/")[-1])]
    preserved_inventory_records.append({**item, "reports": len(old["reports"]), "issues": len(old["issues"])})
discovery_records = []
for item in bundle["preserved_current_discovery_reports"]:
    old = ref(item)
    sid = old["lesson_id"]
    original = ref(entries[sid]["original_report"])
    current = ref(entries[sid]["report"])
    assert old["reviewer_task"] == current["reviewer_task"] == original["reviewer_task"]
    assert original["issues"] == [] and current["original_issue_resolutions"] == []
    assert old["issues"][0]["id"] == current["issues"][0]["id"]
    assert old["issues"][0]["source_quote"] == current["issues"][0]["source_quote"]
    assert old["issues"][0]["details"] == current["issues"][0]["details"]
    assert utc(old["created_at"]) < utc(current["created_at"])
    discovery_records.append({**item, "lesson_id": sid, "reviewer_task": old["reviewer_task"],
                              "verdict_then": old["verdict"], "issue_status_then": old["issues"][0]["status"],
                              "issue_id": old["issues"][0]["id"], "current_status": current["issues"][0]["status"],
                              "original_issue_count": 0, "preserved_as_current_discovery": True})
old_entries = {e["lesson_id"]: e for e in old_bundle["sections"]}
changed_sections = {sid: {k: {"before": old_entries[sid].get(k), "after": entry.get(k)}
                          for k in set(old_entries[sid]) | set(entry)
                          if old_entries[sid].get(k) != entry.get(k)}
                    for sid, entry in entries.items() if old_entries[sid] != entry}
assert set(changed_sections) == {"R.1", "W.1", "W.7", "T.4", "T.5"}

original_path = REPORT_DIR + "/technical-original-v1.json"
prior_path = REPORT_DIR + "/technical-current-v1.json"
assert digest(original_path) == "f41987fdfb492664e559e32d5f1b774e57e7464057d989ccb61b17480e981819"
assert digest(prior_path) == "b3b45f2ccd0f2b3a8624ade5823f20c174116d080d4efb1f9743641be5a54341"
prior = read(prior_path)
original = read(original_path)
for a in prior["artifacts"]:
    assert digest(a["path"]) == a["sha256"]
source_checks = []
for s in prior["sources"]:
    if s["kind"] == "repository_code":
        assert digest(s["path"]) == s["sha256"]
        source_checks.append({"id": s["id"], "path": s["path"], "sha256": s["sha256"]})
external_checks = []
for evidence in ["original-evidence.json", "current-original-additions-v1.json"]:
    for s in read(OUT.relative_to(ROOT) / evidence)["original_receipts"]:
        for k, h in [("raw_cache_path", "raw_sha256"), ("text_cache_path", "text_sha256")]:
            if k in s:
                assert digest(s[k]) == s[h]
                external_checks.append({"source_id": s["source_id"], "path": s[k], "sha256": s[h]})
torch_sources = read(OUT.relative_to(ROOT) / "verification-output.json")["inspected_official_torch_docstrings"]
assert all(digest(s["installed_path"]) == s["file_sha256"] for s in torch_sources)
page = next(p for p in canonical["pages"] if p["page_id"] == "glossary")
raw = (ROOT / page["snapshot"]).read_bytes()
assert raw == (ROOT / "course/glossary.md").read_bytes()
assert hashlib.sha256(raw).hexdigest() == prior["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == prior["figure_sha256"] == {}
text = raw.decode()
assert not re.search(r"!\[|<img\b", text)
lines = text.splitlines()
issue_lines = [42, 108, 198, 28, 107, 223]
assert len(prior["claims"]) == 201 and len(prior["issues"]) == len(original["issues"]) == 6
for i, line in enumerate(issue_lines):
    assert prior["issues"][i]["current_source_quote"] == lines[line - 1]
    assert prior["original_issue_resolutions"][i]["current_source_quote"] == lines[line - 1]
    assert prior["original_issue_resolutions"][i]["json_pointer"] == f"/issues/{i}"
    assert prior["issues"][i]["source_quote"] == original["issues"][i]["source_quote"]
    assert prior["issues"][i]["status"] == "resolved"
links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
for label, destination in links:
    path, _, fragment = destination.partition("#")
    target = ROOT / "course" / path
    assert target.is_file()
    assert not fragment or re.search(r"^##\s+" + re.escape(fragment) + r"\s", target.read_text(), re.M)
index_text = text.split("### 英文與縮寫索引", 1)[1]
index_terms = {}
for line in index_text.splitlines():
    if not line.startswith("**"):
        continue
    terms = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
    if line.startswith("**D**"):
        terms = [("decode", "chapters/01.md#1.1, chapters/16.md#16.2")] + terms[2:]
    keys = [re.sub("[^a-z0-9]", "", label.lower()) for label, _ in terms]
    assert keys == sorted(keys)
    for label, target in terms:
        assert label not in index_terms
        index_terms[label] = target
assert len(index_terms) == 194 and len(links) == 455
g_records = []
for sid in ["G.1", "G.2", "G.3", "G.4"]:
    entry = entries[sid]
    assert entry == old_entries[sid]
    g = ref(entry["report"])
    for s in g["sources"]:
        if s["kind"] == "repository_code":
            assert digest(s["path"]) == s["sha256"]
    g_records.append({"lesson_id": sid, "report": entry["report"], "claims": len(g["claims"]),
                      "report_and_slice_unchanged": True, "current_repository_sources_match": True})
g4 = ref(entries["G.4"]["report"])
reported_index = {}
for c in g4["claims"]:
    match = re.fullmatch(r"英文索引「(.+)」連到本課定義位置：(.+)", c["statement"])
    if match:
        reported_index[match[1]] = match[2]
assert reported_index == index_terms
old_canonical = ref(prior["source_freeze"])
old_pages = {p["page_id"]: p for p in old_canonical["pages"]}
new_pages = {p["page_id"]: p for p in canonical["pages"]}
dependency_changes = []
for sid, note in [
    ("9.12", "新實作按softmax前分數及固定ID處理有限精度平手；glossary未宣稱點積或最大機率足以保重排，pointer/readout限制仍成立。"),
    ("9.15", "只把官方核對日期改為真2026-10-29；n>1集中度公式未變，與本人固定官方adapter相符。"),
    ("9.18", "現在明分同生成器換seed是同來源重抽；模板/組合留出另需分組與排重。glossary只說換seed不叫新來源，仍正確；不把G.2 c45舊scope的組合留出簡稱當成自動成立。"),
]:
    old, new = old_pages[sid], new_pages[sid]
    assert digest(old["snapshot"]) == old["source_sha256"]
    assert digest(new["snapshot"]) == new["source_sha256"]
    dependency_changes.append({"lesson_id": sid, "previous_snapshot": {"path": old["snapshot"], "sha256": old["source_sha256"]},
                               "current_snapshot": {"path": new["snapshot"], "sha256": new["source_sha256"]},
                               "manual_semantic_assessment_after_reading_actual_diff": note})

result = {
    "created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": OWNER,
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/audit-current-v2.py",
    "environment": {"python": platform.python_version(), "device": "CPU metadata/text only; no model loaded"},
    "scope": "本人完整重讀glossary前言及269行、G.1–G.4所有主張/範圍與完整索引；保留201主張的原始依據，六修正仍成立。新R/W/T只核資料/身分/來源/收件，不作內容事實代簽。",
    "bundle": reference(BUNDLE), "previous_whole_report": reference(prior_path), "original_report": reference(original_path),
    "canonical_source_freeze": bundle["canonical_source_freeze"], "section_freeze": bundle["section_freeze"],
    "assignments": manifest["assignments"], "actual_collection": bundle["actual_collection_snapshot"],
    "section_count": 26, "unique_owners": len(owners), "unique_report_refs": len(refs), "indexed_issues": len(issues),
    "section_family_counts": dict(Counter(e["lesson_id"].split(".")[0] for e in records)), "sections": records,
    "changed_sections_since_personally_audited_bundle_v1": changed_sections,
    "authenticated_original_exporter_snapshots": snapshot_records,
    "all_current_report_artifacts_hash_checked": report_artifact_checks,
    "owner_self_check_receipts_hash_checked": len(collection_receipts),
    "issue_inventory": bundle["issue_inventory"], "preserved_issue_inventories": preserved_inventory_records,
    "preserved_current_discovery_chain": discovery_records,
    "issue_binding_note": "35份original/current正好5指標。R1與W1是current才發現的問題，原判issues仍0、original_issue_resolutions仍空；保留R1 current2的revise以及W1 current1/2舊fail指標，未抹成原判已知。",
    "intake": {"manifest": reference(MANIFEST), "output": reference(INTAKE), "exit_code": 0,
               "result": read(INTAKE), "meaning": "資料schema/來源/身分收件通過；collection仍in_progress，whole collector/stage/seal另需真收件，未冒稱完成。"},
    "glossary": {"source": "course/glossary.md", "snapshot": reference(page["snapshot"]), "figure_sha256": {},
                 "same_bytes_as_current_v1": True, "claims_retained": 201, "claim_family_counts": dict(Counter(c["id"].split(".")[0] for c in prior["claims"])),
                 "six_issue_current_quotes": {str(i): lines[n-1] for i, n in enumerate(issue_lines)},
                 "original_issue_quotes_and_pointers_preserved": True, "local_links_checked_now": len(links),
                 "index_terms_checked_now": len(index_terms), "index_definition_links": 195, "index_exactly_matches_G4_claim_bindings": True},
    "G_reports_semantically_crosschecked_after_original": g_records,
    "necessary_dependency_changes_personally_read": dependency_changes,
    "original_repository_sources_unchanged": source_checks,
    "cached_original_external_fingerprints_rechecked": external_checks,
    "installed_official_torch_source_fingerprints_rechecked": len(torch_sources),
    "six_own_semantic_closures": [
        "CE每題-log(p)限無權重單一正確類別；平均是reduction，不是CE命名條件。保留原官方unbatched/none/sum/mean及本人兩題CPU證據。",
        "L2理想公式限非零；F.normalize分母max(norm,eps)，零及極小向量可不足1；保留0與0.01範數反例。",
        "Jev本課只介紹三型，未聲稱官方總數三；本人原讀OpenAPI0.2.0的Noul/Choice/Score/Box證據仍是同bytes。",
        "JSON本課常用物件文字是例子；解析才成資料，不把所有JSON限制成大括號或當工具已執行。",
        "LayerNorm核心中心化/尺度，實作sqrt(var+eps)與可學affine條件已補，不保證所有後續數字不增大。",
        "索引彙整全頁涉及概念，含activation checkpointing中文對應；本次新核194詞/195定義連結與455本地連結。",
    ],
    "numeric_evidence_policy": "本次無新數字變動，未重跑未改的CPU探針或HTTPS；原判與current1的實跑輸入/結果及其程式/依賴指紋全保留並重核，不把舊結果冒稱本次新執行。",
    "historical_four_verifications": "四項歷史fix verification由各真owner的獨立appendix另行收件；未納入本次六原issues，也未由本collector代簽。",
    "limitations": ["未簽R/W/T事實結論。", "intake的schema_checked_collection_pending不是整體stage/seal。",
                    "未做GPU、長訓練、付費、雲端Colab執行、部署或新能力評測。", "未修改任何舊report、fail、checker、正文或模型。"],
}
path = OUT / "collector-current-audit-v2.json"
with path.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": reference(path.relative_to(ROOT)), "sections": 26, "owners": len(owners),
                  "reports": len(refs), "issues": len(issues), "changed_sections": sorted(changed_sections),
                  "original_exporter_snapshots": len(snapshot_records), "glossary_claims": 201,
                  "own_original_issues_still_resolved": 6, "intake_status": read(INTAKE)["status"]}, ensure_ascii=False))
