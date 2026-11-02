"""Bounded source/freeze/sidecar callback; reuse prior substantive reading, never run NN."""
import copy
import datetime
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
D = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-training"
TASK = "/root/omni_p5_technical_training_whole"
SIDECAR = {"path": "docs/technical-reviews/artifacts/omni-P5-numbered-v2/numbered-section-reviews-v1.json", "sha256": "a6b2189776362c7bc883c0ea26d01c2223b191b5989799dfa411fc48ba0dd87f"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = path if isinstance(path, Path) else ROOT / path
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def read(reference):
    raw = (ROOT / reference["path"]).read_bytes()
    assert sha(raw) == reference["sha256"], reference["path"]
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


validator = module("training_callback_validator", "scripts/check_technical_reviews.py")
reading = module("training_callback_slices", "scripts/reading_time.py")
setup = module("training_callback_setup", "scripts/omni_revision/review_setup.py")
api = module("training_callback_api", "docs/review-tools/incremental_revision.py")
old_path = D / "technical-current-v1.json"
old_ref = ref(old_path)
assert old_ref["sha256"] == "c987365ee519a9ea7dcbe3c7384883683323a8920916ded2adfdfcbd67d92163"
old = read(old_ref)
original_ref = old["original_report"]
assert original_ref["sha256"] == "d31b396b2fd906cfdfd672edd65ac7621cfe218779fac8edb9553dfb03184c4e"
original = read(original_ref)
old_side = read(old["numbered_section_reviews"])
side = read(SIDECAR)
assert side["sections"] == old_side["sections"] and len(side["sections"]) == 26
assert {k for k in side if side[k] != old_side[k]} == {"created_at", "canonical_source_freeze", "section_freeze", "issue_inventory"}
freeze = read(side["canonical_source_freeze"])
old_freeze = read(old["source_freeze"])
numbered = read(side["section_freeze"])
old_numbered = read(old_side["section_freeze"])
assert side["section_freeze"]["sha256"] == "1ce6261143dc8321cf9ec7bc9e298c678aab07eed5daaaf66cb8ca04dd16a68a"
assert numbered["canonical_source_freeze"] == side["canonical_source_freeze"]
inventory = read(side["issue_inventory"])
old_inventory = read(old_side["issue_inventory"])
assert inventory["issues"] == old_inventory["issues"]
assert {k for k in inventory if inventory[k] != old_inventory[k]} == {"canonical_source_freeze", "section_freeze"}
assert inventory["canonical_source_freeze"] == side["canonical_source_freeze"] and inventory["section_freeze"] == side["section_freeze"]
pages = {x["page_id"]: x for x in freeze["pages"]}
old_pages = {x["page_id"]: x for x in old_freeze["pages"]}
assert set(pages) == set(old_pages)
changed_pages = [key for key in pages if pages[key] != old_pages[key]]
assert changed_pages == ["curriculum"]
page = pages["training"]
assert page == old_pages["training"]
body = (ROOT / page["snapshot"]).read_bytes()
assert body == (ROOT / "course/training.md").read_bytes()
assert sha(body) == page["source_sha256"] == old["source_sha256"] == original["source_sha256"]
assert page["figures_sha256"] == old["figure_sha256"] == original["figure_sha256"] == {}

dependency_checks = []
for kind, rows in [("current1_repository_source", old["sources"]), ("current1_artifact", old["artifacts"])]:
    for row in rows:
        if "path" in row:
            actual = sha((ROOT / row["path"]).read_bytes())
            assert actual == row["sha256"], (kind, row["path"])
            dependency_checks.append({"kind": kind, "id": row["id"], "path": row["path"], "sha256": actual, "matches_prior_review": True})
assert not any(x["path"] == "course/curriculum.md" for x in dependency_checks)
packet = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json").read_bytes())
readers = set(packet["reader_tasks"])
slices = reading.lesson_slices(body.decode())
assert set(slices) == {f"T.{i}" for i in range(1, 12)}
nf = {x["page_id"]: x for x in numbered["pages"]}
of = {x["page_id"]: x for x in old_numbered["pages"]}
sections = []
for entry in side["sections"]:
    sid = entry["lesson_id"]
    if sid not in slices:
        continue
    new_row, old_row = nf[sid], of[sid]
    assert {k: v for k, v in new_row.items() if k != "snapshot"} == {k: v for k, v in old_row.items() if k != "snapshot"}
    current_raw = slices[sid].encode()
    assert current_raw == (ROOT / new_row["snapshot"]).read_bytes() == (ROOT / old_row["snapshot"]).read_bytes()
    assert sha(current_raw) == new_row["source_sha256"] == old_row["source_sha256"]
    assert new_row["canonical_page_id"] == "training" and new_row["selector"] == "section:" + sid
    assert new_row["figures_sha256"] == reading.figure_hashes(ROOT, ROOT / "course/training.md", slices[sid]) == {}
    report, raw_original = read(entry["report"]), read(entry["original_report"])
    assert report["source_sha256"] == new_row["source_sha256"] and report["figure_sha256"] == {}
    checked = []
    for evidence_kind, evidence_report in [("current", report), ("original", raw_original)]:
        for source in evidence_report["sources"]:
            if source.get("kind") == "repository_code":
                actual = sha((ROOT / source["path"]).read_bytes())
                if evidence_kind == "current":
                    assert actual == source["sha256"]
                checked.append({"kind": evidence_kind + "_source", "path": source["path"], "recorded_sha256": source["sha256"], "live_sha256": actual, "live_matches": actual == source["sha256"], "version_contract": "current _validate" if evidence_kind == "current" else "original check_original with its true original canonical freeze and explicit original_source_snapshots; old document versions need not equal today's chapter whole-file"})
        for artifact in evidence_report["artifacts"]:
            assert sha((ROOT / artifact["path"]).read_bytes()) == artifact["sha256"]
            checked.append({"kind": evidence_kind + "_artifact", "path": artifact["path"], "sha256": artifact["sha256"]})
    setup.check_original(raw_original, entry["original_section_freeze"], sid, "training", raw_original["reviewer_task"], validator, api, original_source_snapshots=entry.get("original_source_snapshots", []))
    errors = validator._validate(ROOT, ROOT / "course/training.md", sid, slices[sid], report, {report["reviewer_task"]: [sid]}, readers)
    assert not errors, (sid, errors)
    sections.append({"lesson_id": sid, "source_sha256": sha(current_raw), "source_bytes": len(current_raw), "new_snapshot": new_row["snapshot"], "previous_snapshot": old_row["snapshot"], "figures_sha256": {}, "report": entry["report"], "original_report": entry["original_report"], "report_and_original_identity_unchanged": True, "current_dependencies_match_and_true_original_versions_preserved": True, "original_full_contract_passed": True, "original_source_snapshots": entry.get("original_source_snapshots", []), "evidence_digest_checks": checked, "current_validate_errors": errors, "substantive_reading_reused_from_current1": True, "fresh_blind_reading_or_section_execution_claimed": False})
assert len(sections) == 11
probe_path = A / "metadata-callback-current-v2.json"
observed_at = datetime.datetime.now(datetime.UTC).isoformat()
probe = {
    "created_at": observed_at, "reviewer_task": TASK, "kind": "bounded metadata/source/dependency callback; reuse already-read substance",
    "previous_current": old_ref, "independent_original": original_ref, "source_freeze": side["canonical_source_freeze"], "numbered_freeze": side["section_freeze"], "shared_sidecar": SIDECAR, "shared_issue_inventory": side["issue_inventory"],
    "freeze_page_metadata_changes": changed_pages, "unreviewed_changed_page_scope": "Only identified curriculum in freeze metadata; no substantive curriculum review performed or claimed.",
    "whole_page": {"source_bytes": len(body), "source_sha256": sha(body), "source11_snapshot_live_current1_original_all_equal": True, "figure_sha256": {}, "all118_prior_substantive_claims_reused": True},
    "prior_registered_dependencies": dependency_checks, "sections": sections,
    "sidecar_reuse": {"all26_section_records_exactly_equal": True, "inventory_issues_exactly_equal": True, "new_metadata_fields_only": ["created_at", "canonical_source_freeze", "section_freeze", "issue_inventory"], "substantive_scope": "Training T.1–T.11 only; other sidecar units are compared as record identities, not semantically read.", "same_shared_artifact_referenced_in_place": True, "no_copy_or_collector_backreference_cycle": True},
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "execution": "source/hash/JSON/schema only; no NN,torch import,training,network,dispatch,workflow,publication"},
    "status": "technical03 remains open; current2 not yet actually received; no collection/seal created",
    "scope_limits": ["No fresh blind full-page or eleven-section review; all substantive judgments and original/current1 evidence are explicitly reused after exact identity checks.", "No NN/model construction/load/forward,training,tuning,paid service,Modal,workflow or publishing.", "Native Windows T.11 remains pending book PR CI.", "Historical appendix-v1 is preserved with current1 binding. No current2-binding historical appendix is written before actual root receipt of current2.", "No current technical seal,actual current2 receipt,final historical assembly or check-numbered gate is created or claimed."]
}
with probe_path.open("x", encoding="utf-8") as stream:
    json.dump(probe, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

current = copy.deepcopy(old)
current["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
current["source_freeze"] = side["canonical_source_freeze"]
current["previous_current_report"] = old_ref
current["numbered_section_reviews"] = SIDECAR
current["review_kind"] = "Same original reviewer bounded metadata callback current2; no fresh blind reading; byte-identical substantive review reused"
for artifact in current["artifacts"]:
    if artifact["id"] == "a_numbered_shared":
        artifact.update(SIDECAR)
        artifact["description"] = "Exact single new shared26-section sidecar referenced in place. All section proof records are byte-equivalent JSON objects to prior shared sidecar; only outer freeze/inventory references and creation time changed. No copied sidecar or collector backreference. Training T.1–T.11 substantive reading is reused from prior independent work."
    if artifact["id"] == "a_numbered_issues_shared":
        artifact.update(side["issue_inventory"])
        artifact["description"] = "Exact new shared issue inventory referenced in place. All issue entries equal prior inventory; only canonical/numbered freeze references changed. Training T.9 closure judgment is explicitly reused."
current["artifacts"] += [
    {"id": "a_metadata_callback_code", "kind": "code", **ref(Path(__file__)), "description": "Bounded read-only source11/numbered-v6/new-shared-sidecar callback; all prior registered evidence hashes and eleven source slices checked; existing per-section/current canonical validators run. No model calls or receipts/seals created."},
    {"id": "a_metadata_callback", "kind": "execution", **ref(probe_path), "description": "Actual callback source/hash/JSON/schema observations. Entire training body and all registered dependencies match current1; all eleven exact slice bytes and section report/evidence identities preserved. Original section dependencies checked against true original freezes/snapshots, current dependencies against live bytes. Shared26 records identical; no new blind reading or model execution.", "command": "PYTHONDONTWRITEBYTECODE=1 python docs/technical-reviews/artifacts/omni-P5-training/metadata-callback-current-v2.py", "result": "Completed metadata callback: body64057bytes unchanged,42 registered local sources and14 artifacts match; eleven exact T slices,current dependencies and true original frozen dependencies verified; all original contracts and each current section _validate pass; no new substantive or NN work.", "environment": probe["environment"]}
]
current["sources"].append({"id": "s_metadata_callback", "kind": "execution", "title": "Bounded source11/shared-sidecar identity callback", "verified": True, "artifact_id": "a_metadata_callback"})
current["collector_crosscheck"] = {
    "performed_at": observed_at, "scope": "Bounded metadata/source/dependency and exact T.1–T.11 slice/section-evidence identity verification only; substantive reading/judgment reused from previous current1.", "shared_sidecar": SIDECAR, "shared_issue_inventory": side["issue_inventory"], "execution_artifact_id": "a_metadata_callback",
    "prior_substantive_crosscheck": {"report": old_ref, "json_pointer": "/collector_crosscheck"}, "prior_crosscheck_artifact": ref(A / "crosscheck-current-v1.json"),
    "summary": {"checked_training_sections": 11, "unchanged_shared_section_records": 26, "unchanged_local_sources": 42, "unchanged_current1_artifacts": 14, "section_contract_errors": 0, "unresolved_training_section_issues": 0, "current_phase_sealed": False, "new_model_execution": False, "new_blind_review": False},
    "body_reuse": probe["whole_page"], "new_work": "Read updated freeze/sidecar metadata and verify live/snapshot hashes,all prior registered dependencies,eleven T slice bytes and registered evidence digests; run existing section/canonical schema checks. No new substantive full-page reread,measurement recomputation or model run.",
    "reused_work": "Own independent118 whole-page substantive claims and original source/execution evidence; prior current1 actual full11-section report reading,owner/receipt chronology and T.9 issue closure. T.1–T.10 genuine older tasks and T.11 genuine P5 task remain unchanged,not rerun or rediscovered.", "scope_limits": "technical03 remains open. Native Windows T.11 pending book PR CI. No current2 actual receipt,collection/seal or final check-numbered/historical gate claimed. New historical appendix binding waits for root's actual current2 receipt."
}
current["checks"]["factual_accuracy"]["details"] += " Current2限定沿用：本文與11切片逐byte不變，118原主張與既讀逐節判斷明確沿用，不冒稱新盲讀。"
current["checks"]["source_verification"]["details"] += " Current2實際核source11/numbered-v6/new共享sidecar；26records與全部issue entries相同，42原本地來源/14原artifacts無drift。11節current來源/artifact核現行bytes；original來源依其真正舊freeze/明示保存快照核，不假稱T.2/T.5的歷史章全文hash等於今日全文。11原判完整契約與11current_validate通過。"
current["checks"]["limitations"]["details"] += " Current2只做metadata callback；已在先前fullcurrent實收後讀歷史group9，不能把這次稱盲讀。尚未為current2建立歷史appendix或宣稱root收件。"
current["unverified_scope"] = [x for x in old["unverified_scope"] if not x.startswith("Historical group9 appendix has not been delivered")]
current["unverified_scope"] += probe["scope_limits"]
current["post_original_context"] = {"bounded_callback": True, "fresh_blind_review": False, "historical_group9_was_prompted_after_current1_actual_receipt": True, "existing_appendix_v1_unchanged": ref(D / "historical-group-9-appendix-v1.json"), "new_current2_binding_appendix": "Not written; must await root actual receipt of this new full current2."}
assert current["claims"] == old["claims"] == original["claims"] and current["issues"] == old["issues"] == []
assert not current.get("historical_issue_rechecks")
target = D / "technical-current-v2.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(current, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
command = [sys.executable, "docs/course-review-20261019/tools/check_technical_report.py", "training", target.relative_to(ROOT).as_posix(), "--freeze", current["source_freeze"]["path"]]
for reader in sorted(readers):
    command += ["--reader-task", reader]
completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
selfcheck = {"created_at": datetime.datetime.now(datetime.UTC).isoformat(), "reviewer_task": TASK, "report": ref(target), "command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "scope": "Existing canonical _validate only,plus actual bounded source/slice/dependency callback;not phase gate or collection/seal.", "source_freeze": current["source_freeze"], "shared_sidecar": SIDECAR}
selfcheck_path = A / "current-selfcheck-v2.json"
with selfcheck_path.open("x", encoding="utf-8") as stream:
    json.dump(selfcheck, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
assert ref(old_path) == old_ref and ref(ROOT / original_ref["path"]) == original_ref
assert ref(D / "historical-group-9-appendix-v1.json")["sha256"] == "e53d8a214826307a97282e21477ea7dc979445234a227c5d922eb21d8b036855"
print(json.dumps({"report": {**ref(target), "bytes": target.stat().st_size, "created_at": current["created_at"], "verdict": current["verdict"]}, "callback": {**ref(probe_path), "bytes": probe_path.stat().st_size, "created_at": observed_at}, "selfcheck": {**ref(selfcheck_path), "bytes": selfcheck_path.stat().st_size, "created_at": selfcheck["created_at"], "exit_code": completed.returncode, "stdout": completed.stdout}, "original_and_current1_and_appendix1_preserved": True}, ensure_ascii=False))
if completed.returncode:
    raise SystemExit(completed.returncode)
