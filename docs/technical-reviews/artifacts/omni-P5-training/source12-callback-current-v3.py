"""Bounded training-only source12 callback; technical seal immutable, no NN/kernel."""
import copy
import datetime
import difflib
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
D = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-training"
TASK = "/root/omni_p5_technical_training_whole"
SIDECAR = {"path": "docs/technical-reviews/artifacts/omni-P5-numbered-v3/numbered-section-reviews-v1.json", "sha256": "5225b24a74b01fe824481d792c61d299f6400c3141c1bfd185ed196b432ec158"}
REPAIRS = {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/coherence-three-author-repairs-source12-v1.json", "sha256": "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = path if isinstance(path, Path) else ROOT / path
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def raw(reference):
    result = (ROOT / reference["path"]).read_bytes()
    assert sha(result) == reference["sha256"], reference["path"]
    return result


def read(reference):
    return json.loads(raw(reference))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


validator = module("training_source12_validator", "scripts/check_technical_reviews.py")
reading = module("training_source12_reading", "scripts/reading_time.py")
setup = module("training_source12_setup", "scripts/omni_revision/review_setup.py")
api = module("training_source12_api", "docs/review-tools/incremental_revision.py")
old_ref = ref(D / "technical-current-v2.json")
assert old_ref["sha256"] == "7284ce2234782759ddbea67d9bfd19522bba91cfbfa51e8951c06375592cda9c"
old = read(old_ref)
original = read(old["original_report"])
appendix2_ref = ref(D / "historical-group-9-appendix-v2.json")
assert appendix2_ref["sha256"] == "4140974e35f886a788be0793830636ebdade64a1619f5f0c7992e75b74698e7c"
side, old_side, repairs = read(SIDECAR), read(old["numbered_section_reviews"]), read(REPAIRS)
assert side["canonical_source_freeze"]["sha256"] == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
assert side["section_freeze"]["sha256"] == "f7ba7cc4df7edfb693f2024f819e8886a0fabba56bdfb0b6b94f18c364551423"
assert repairs["source_freeze"] == side["canonical_source_freeze"] and repairs["before_freeze"] == old["source_freeze"]
freeze, prior_freeze = read(side["canonical_source_freeze"]), read(old["source_freeze"])
pages, prior_pages = ({x["page_id"]: x for x in f["pages"]} for f in [freeze, prior_freeze])
assert set(pages) == set(prior_pages)
changed = {key for key in pages if pages[key] != prior_pages[key]}
assert changed == {"9.17", "12.17", "19.23"}
page = pages["training"]
assert page == prior_pages["training"]
body = raw({"path": page["snapshot"], "sha256": page["source_sha256"]})
assert body == (ROOT / "course/training.md").read_bytes()
assert sha(body) == old["source_sha256"] == original["source_sha256"]
assert page["figures_sha256"] == old["figure_sha256"] == original["figure_sha256"] == {}
prior_dependency_checks = []
for kind, rows in [("source", old["sources"]), ("artifact", old["artifacts"])]:
    for item in rows:
        if "path" in item:
            assert ref(item["path"])["sha256"] == item["sha256"], (kind, item["path"])
            prior_dependency_checks.append({"kind": kind, "id": item["id"], "path": item["path"], "sha256": item["sha256"]})
assert sum(x["kind"] == "source" for x in prior_dependency_checks) == 42
assert sum(x["kind"] == "artifact" for x in prior_dependency_checks) == 16

technical_ref = ref("docs/course-revision-20261029-omni/reviews/P5-actual/technical-phase-proof-v1.json")
assert technical_ref["sha256"] == "b756abd42038c1baf010b0ee64de67cd016e903f0fc03371322ec5c64b35be64"
technical = read(technical_ref)
assert technical["status"] == "completed" and technical["role"] == "technical"
continuity_ref = ref("docs/course-revision-20261029-omni/reviews/P5-actual/coherence-stage-open-v1.json")
continuity = read(continuity_ref)
assert continuity["role"] == "continuity" and continuity["preceding_technical_seal"] == technical_ref
assert api.at(technical["completed_at"]) <= api.at(continuity["started_at"])
collection_ref = ref("docs/course-revision-20261029-omni/reviews/P5-actual/coherence-current-collection-intake-v3.json")
assert collection_ref["sha256"] == "64e2284b04cbb618fde53e21cd9365b4398eb4b2b5af8944cdbfedf4ab55cbca"
collection = read(collection_ref)
assert collection["role"] == "continuity" and "OPEN" in collection["status"]
prior_own_receipts = []
for reference in [old_ref, appendix2_ref]:
    matches = [x for x in technical["collection_index"] if x["path"] == reference["path"] and x["sha256"] == reference["sha256"]]
    assert len(matches) == 1 and reference["sha256"] in technical["report_sha256s"]
    prior_own_receipts.append(matches[0])

entries, prior_entries = ({x["lesson_id"]: x for x in s["sections"]} for s in [side, old_side])
assert len(entries) == len(prior_entries) == 26 and set(entries) == set(prior_entries)
assert [key for key in entries if entries[key] != prior_entries[key]] == ["W.7"]
new_w7, old_w7 = entries["W.7"], prior_entries["W.7"]
new_w7_report, old_w7_report = read(new_w7["report"]), read(old_w7["report"])
assert new_w7["original_report"] == old_w7["original_report"]
assert new_w7_report["reviewer_task"] == old_w7_report["reviewer_task"] == "/root/omni_p5_technical_w_7"
assert new_w7_report["source_sha256"] == old_w7_report["source_sha256"]
w7_receipts = [(i, row) for i, row in enumerate(collection["collection_index"]) if row["path"] == new_w7["report"]["path"] and row["sha256"] == new_w7["report"]["sha256"]]
assert len(w7_receipts) == 1
w7_index, w7_receipt = w7_receipts[0]
assert w7_receipt["received_at"] == new_w7["current_collection"]["received_at"] and w7_receipt["collection_role"] == "continuity"
window = new_w7["current_dispatch_window"]
assert api.at(continuity["started_at"]) <= api.at(window["not_before"]) <= api.at(window["not_after"]) <= api.at(new_w7_report["created_at"]) <= api.at(w7_receipt["received_at"])
numbered, prior_numbered = read(side["section_freeze"]), read(old_side["section_freeze"])
assert numbered["canonical_source_freeze"] == side["canonical_source_freeze"]
nf, pf = ({x["page_id"]: x for x in f["pages"]} for f in [numbered, prior_numbered])
assert set(nf) == set(pf)
assert all({k: v for k, v in nf[sid].items() if k != "snapshot"} == {k: v for k, v in pf[sid].items() if k != "snapshot"} for sid in nf)
inventory, prior_inventory = read(side["issue_inventory"]), read(old_side["issue_inventory"])
assert inventory["issues"] == prior_inventory["issues"]
assert inventory["canonical_source_freeze"] == side["canonical_source_freeze"] and inventory["section_freeze"] == side["section_freeze"]
assert len(inventory["reports"]) == len(prior_inventory["reports"])

packet = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json").read_bytes())
reader_tasks = set(packet["reader_tasks"])
slices = reading.lesson_slices(body.decode())
assert set(slices) == {f"T.{i}" for i in range(1, 12)}
sections = []
for sid in sorted(slices, key=lambda value: int(value.split(".")[1])):
    entry = entries[sid]
    assert entry == prior_entries[sid]
    section_raw = slices[sid].encode()
    assert section_raw == raw({"path": nf[sid]["snapshot"], "sha256": nf[sid]["source_sha256"]}) == raw({"path": pf[sid]["snapshot"], "sha256": pf[sid]["source_sha256"]})
    report, initial = read(entry["report"]), read(entry["original_report"])
    setup.check_original(initial, entry["original_section_freeze"], sid, "training", initial["reviewer_task"], validator, api, original_source_snapshots=entry.get("original_source_snapshots", []))
    errors = validator._validate(ROOT, ROOT / "course/training.md", sid, slices[sid], report, {report["reviewer_task"]: [sid]}, reader_tasks)
    assert not errors, (sid, errors)
    sections.append({"lesson_id": sid, "source_sha256": sha(section_raw), "source_bytes": len(section_raw), "figure_sha256": {}, "snapshot": nf[sid]["snapshot"], "report": entry["report"], "original_report": entry["original_report"], "entry_unchanged": True, "original_contract_passed_on_true_original_freeze": True, "current_validate_errors": errors, "substantive_judgment_reused": True})

links = re.findall(r"\[([^\]]+)\]\((chapters/(?:09|12|19)\.md(?:#[^)]+)?)\)", body.decode())
link_checks = []
for label, link in links:
    target_id = link.partition("#")[2]
    if target_id:
        assert target_id in pages and pages[target_id] == prior_pages[target_id]
        raw({"path": pages[target_id]["snapshot"], "sha256": pages[target_id]["source_sha256"]})
        link_checks.append({"label": label, "url": link, "direct_anchor_target_id": target_id, "direct_target_source_sha256": pages[target_id]["source_sha256"], "direct_target_unchanged": True, "scope": "Label19.21–19.24 also refers to19.23; that bounded paragraph change is separately assessed below." if label == "19.21–19.24" else "Exact existing direct anchor only."})
    else:
        assert link == "chapters/19.md"
        link_checks.append({"label": label, "url": link, "direct_target_unchanged": False, "scope": "Broad chapter19 link retains its destination; whole chapter file changed only at known19.23 paragraph,whose relation to training is assessed below. No whole-chapter technical re-review."})
relations = {
    "9.17": "Training's chapter09 anchors9.1/9.2/9.4/9.6/9.18 retain exact bytes. The one-line raw-score selection repair inside9.17 is not a training command or recorded model/measurement source. No training reference or claim changes.",
    "12.17": "Training's chapter12 targets12.1 sample-rate explanation and12.12 fixed joint-training route retain exact bytes. The inserted mean-square scale explanation belongs to12.17's artificial energy rule example; it does not change trainingT.6/T.10 commands,dependencies or original measured records.",
    "19.23": "Training's unchanged19.21–19.24 label and broad chapter19 link include this page. The paragraph now permits problem text plus level descriptions jointly specifying rubric rules and identifies a candidate from description plus this problem's rules. The exact diff changes no program,data,checkpoint,T/tau or measurement. TrainingT.11 still reports ten completed jobs and C-v2 final failure/research-candidate limits; its registered executable/data/release/measurement evidence is unchanged. This is only the necessary reference relation,not acceptance of all19.23 claims."
}
repair_checks = []
assert {x["page_id"] for x in repairs["changes"]} == changed
for change in repairs["changes"]:
    sid = change["page_id"]
    before, after = raw(change["before"]).decode(), raw(change["after"]).decode()
    assert change["before"]["sha256"] == prior_pages[sid]["source_sha256"] and change["after"]["sha256"] == pages[sid]["source_sha256"]
    actual_diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="source11/" + sid, tofile="source12/" + sid, n=2))
    assert actual_diff == change["diff"]
    repair_checks.append({"page_id": sid, "before": change["before"], "after": change["after"], "exact_diff_verified": actual_diff, "necessary_training_relation": relations[sid]})

observed_at = datetime.datetime.now(datetime.UTC).isoformat()
probe = {
    "created_at": observed_at, "reviewer_task": TASK, "kind": "Bounded training collector source12 callback; prior substantive reading reused",
    "previous_current": old_ref, "independent_original": old["original_report"], "source_freeze": side["canonical_source_freeze"], "numbered_freeze": side["section_freeze"], "shared_sidecar": SIDECAR, "shared_issue_inventory": side["issue_inventory"], "author_three_repairs": REPAIRS,
    "whole_page": {"source_bytes": len(body), "source_sha256": sha(body), "source12_snapshot_live_current2_original_all_equal": True, "figures": {}, "all118_substantive_claims_unchanged": True},
    "prior_registered_dependency_checks": prior_dependency_checks, "training_sections": sections,
    "sidecar_scope": {"all26_section_source_identities_unchanged": True, "unchanged_proof_entries": 25, "changed_metadata_entry_only": "W.7", "training11_entries_unchanged": True, "issue_entries_unchanged": True, "W7_metadata_only": {"same_original_owner": new_w7_report["reviewer_task"], "same_section_source_sha256": new_w7_report["source_sha256"], "current_report": new_w7["report"], "actual_continuity_receipt": {"report": collection_ref, "json_pointer": "/collection_index/" + str(w7_index), "entry": w7_receipt}, "dispatch_window": window, "W7_substantive_claims_not_reviewed_by_this_collector": True}, "same_shared_sidecar_referenced_in_place_no_cycle": True},
    "necessary_link_targets": link_checks, "three_exact_diffs_and_necessary_training_relations": repair_checks,
    "phase_context": {"technical_completed_immutable": technical_ref, "technical_completed_at": technical["completed_at"], "prior_own_technical_receipts": prior_own_receipts, "current_continuity_open": continuity_ref, "continuity_started_at": continuity["started_at"], "current_receipt_metadata": collection_ref, "this_current3_collection_role": "continuity", "this_current3_has_not_yet_been_received": True},
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "execution": "source/hash/JSON/diff/link/schema only; no torch import,NN,kernel,training,network,workflow,publication or seal writes"},
    "scope_limits": ["Not a new blind review: reuse own118 independent whole-page claims,already-read T1–T11 reports and prior evidence after exact identity checks.", "Only necessary three-diff training reference relations assessed; no role as independent coherence reviewer or W.7 technical owner.", "Author's outputs/kernel receipts are not used as this collector's evidence and no ignored outputs are required.", "No NN/model load/forward/training/tuning/new arithmetic measurement,kernel,GPU/MPS/Windows/Colab execution,paid service,Modal,workflow or publication. Native Windows T.11 remains pending book PR CI.", "Technical03 completed seal remains byte-identical. Current callback belongs to still-open continuity03 and has no actual receipt yet.", "Appendix-v2 remains bound to actually received current2; no current3-binding historical appendix is written before actual root continuity receipt. No current/final phase gate or seal created."]
}
probe_path = A / "source12-callback-current-v3.json"
with probe_path.open("x", encoding="utf-8") as stream:
    json.dump(probe, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
current = copy.deepcopy(old)
current["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
current["source_freeze"], current["numbered_section_reviews"] = side["canonical_source_freeze"], SIDECAR
current["previous_current_report"] = old_ref
current["review_kind"] = "Same original owner bounded source12 callback collected later in continuity03; technical03 immutable; no fresh blind review or NN/kernel"
for item in current["artifacts"]:
    if item["id"] == "a_numbered_shared":
        item.update(SIDECAR)
        item["description"] = "Exact shared v3 sidecar referenced in place. All26 section source identities unchanged;25 proof entries retained and W.7 replaced by its same-owner callback with actual continuity receipt/dispatch. Training T.1–T.11 entries unchanged and substantive reading reused; W.7 substance not reviewed by this collector. No copied sidecar or collector hash cycle."
    if item["id"] == "a_numbered_issues_shared":
        item.update(side["issue_inventory"])
        item["description"] = "Exact shared v3 issue inventory referenced in place. Existing issue entries unchanged,report identity inventory reflects W.7 current2;training T.9 original issue and prior same-owner closure are reused."
current["artifacts"] += [
    {"id": "a_source12_callback_code", "kind": "code", **ref(Path(__file__)), "description": "Read-only training source12/11T/exact evidence and necessary three-diff relation callback; same shared v3 metadata binding and phase identity checks. No NN/kernel,artifact rerun,other-page full review or seal mutation."},
    {"id": "a_source12_callback", "kind": "execution", **ref(probe_path), "description": "Actual bounded source12 observations:training body,42 prior local sources/16 artifacts and11T exact slices/true original/current contracts;three actual diff/necessary-link relationships;sharedv3 exact metadata;immutable technical seal and open continuity metadata.", "command": "PYTHONDONTWRITEBYTECODE=1 python docs/technical-reviews/artifacts/omni-P5-training/source12-callback-current-v3.py", "result": "training64057bytes/118claims unchanged;11T exact sections and original/current contracts pass;42prior local sources/16artifacts match;three diffs preserve necessary training relations;25shared proofs retained and W.7 same-owner actualC receipt metadata verified;no NN/kernel or technical seal writes.", "environment": probe["environment"]}
]
current["sources"] += [
    {"id": "s_source12_callback", "kind": "execution", "title": "Bounded source12 training callback and exact evidence/ref checks", "verified": True, "artifact_id": "a_source12_callback"},
    {"id": "s_source12_repair_metadata", "kind": "repository_code", "title": "Exact first-party source12 three-repair diff record", **REPAIRS, "version": "Immutable76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7", "verified": True, "inspection_note": "Read only the three explicit before/after diffs and compare their raw source snapshots;assess only necessary training references and retained results/limits. Do not use author's output/kernel references as executed evidence or accept other pages as a coherence reviewer."}
]
current["collector_crosscheck"] = {
    "performed_at": observed_at, "scope": "Training-only bounded source12 body/11T/dependency identity and necessary three-diff reference relations; prior substantive reading explicitly reused.", "shared_sidecar": SIDECAR, "shared_issue_inventory": side["issue_inventory"], "execution_artifact_id": "a_source12_callback", "prior_substantive_crosscheck": {"report": old_ref, "json_pointer": "/collector_crosscheck"},
    "summary": {"training_sections_unchanged": 11, "shared_section_sources_unchanged": 26, "shared_proof_entries_unchanged": 25, "same_owner_W7_callback_metadata": True, "prior_local_sources_unchanged": 42, "prior_artifacts_unchanged": 16, "original_and_current_section_contract_errors": 0, "new_blind_review": False, "new_NN_or_kernel_execution": False, "technical_phase_already_completed_and_immutable": True, "current_continuity_phase_sealed": False},
    "body_reuse": probe["whole_page"], "new_work": "Exact source12/source11 training and11T byte/hash/contract checks,all prior dependencies,three raw diffs and necessary training link relations,sharedv3/issueindex exact binding,and actual W.7 metadata plus technical-closed/continuity-open identities. No new full-page blind or other-page/coherence review,NN or kernel.",
    "reused_work": "Own118 independent full-page substantive claims,original official sources and prior dry-run/measurement evidence;already-read11 section judgments and T.9 closure. Original,current1,current2 and appendix1/2 remain unchanged.", "phase_context": probe["phase_context"], "scope_limits": "Await actual current3 receipt in continuity03 before historical appendix3. Technical03 seal is already completed and must remain unchanged. Native Windows T.11 still pending book PR CI;no final gate/seal claimed."
}
current["checks"]["factual_accuracy"]["details"] = "沿用本人original118項整頁主張與既讀11節判斷；source12 training/11T逐byte不變。本人只新增三筆實際diff與必要引用關係核對：9.17 raw-score toy選擇、12.17能量尺度文字、19.23題目+描述規準修字不改本頁命令、正式結果/失敗限制或既有產物證據；非全頁新盲讀或其他頁/coherence審閱。"
current["checks"]["numeric_verification"]["details"] = "整頁原數值驗證及23筆歷史time/正式rows純重算證據全部沿用本人original/current1；原始來源與登記artifacts exact hashes無drift。本次沒有重新量測、純算新成績或執行NN/kernel，118claims及其數值驗證內容保持原樣。"
current["checks"]["source_verification"]["details"] = "實核source12/sharedv3/issueindex exact references；training64057bytes、42 prior本地來源、16 priorartifacts均不變，11T original依真舊freeze/保存快照核、current依現bytes_validate均通過。新共享26 source identities相同；25proof entries不變，僅W.7 same-owner current2及真continuity receipt/dispatch metadata更新。三修actualdiff逐字核對並只評training必要引用關係。technical03既有seal SHA保持不動。"
current["checks"]["limitations"]["details"] = "同owner限定沿用，沒有新blind/NN/kernel/訓練/GPU/MPS/Windows/Colab/付費/工作流/發布；原模型/測量與他人執行不冒稱本次完成。僅training與必要三diff關係，未兼任W.7 owner或獨立coherence。原technical03已真封存；本callback等待continuity03真收件，不能回寫舊seal。Windows T.11仍待book PR CI；appendix3必須等current3真收件後另建。"
current["unverified_scope"] = probe["scope_limits"]
current["post_original_context"] = {"bounded_callback": True, "fresh_blind_review": False, "historical_group9_was_prompted_after_current1_actual_receipt": True, "existing_historical_appendix_v2_unchanged": appendix2_ref, "current3_binding_historical_appendix": "Not written;must await actual root continuity03 receipt of this current3."}
current["current_phase_context"] = probe["phase_context"]
assert current["claims"] == old["claims"] == original["claims"] and current["issues"] == [] and not current.get("historical_issue_rechecks")
target = D / "technical-current-v3.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(current, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
command = [sys.executable, "docs/course-review-20261019/tools/check_technical_report.py", "training", target.relative_to(ROOT).as_posix(), "--freeze", current["source_freeze"]["path"]]
for task in sorted(reader_tasks):
    command += ["--reader-task", task]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
selfcheck = {"created_at": datetime.datetime.now(datetime.UTC).isoformat(), "reviewer_task": TASK, "report": ref(target), "command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "source_freeze": current["source_freeze"], "shared_sidecar": SIDECAR, "scope": "Existing canonical _validate only plus bounded training callback;not a new blind review,current continuity seal or final gate."}
selfcheck_path = A / "current-selfcheck-v3.json"
with selfcheck_path.open("x", encoding="utf-8") as stream:
    json.dump(selfcheck, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
assert ref(D / "technical-current-v2.json") == old_ref and ref(D / "historical-group-9-appendix-v2.json") == appendix2_ref
assert ref(technical_ref["path"]) == technical_ref and ref(continuity_ref["path"]) == continuity_ref
print(json.dumps({"report": {**ref(target), "bytes": target.stat().st_size, "created_at": current["created_at"], "verdict": current["verdict"]}, "callback": {**ref(probe_path), "bytes": probe_path.stat().st_size, "created_at": observed_at}, "selfcheck": {**ref(selfcheck_path), "bytes": selfcheck_path.stat().st_size, "created_at": selfcheck["created_at"], "exit_code": completed.returncode, "stdout": completed.stdout}, "technical_seal_unchanged": technical_ref, "intended_collection_role": "continuity", "current3_actual_receipt_or_appendix3_not_created": True}, ensure_ascii=False))
if completed.returncode:
    raise SystemExit(completed.returncode)
