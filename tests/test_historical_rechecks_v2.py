"""虛構完整三輪測試；不把fixture當真讀者或審閱證據。"""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("history_v2_fixture", ROOT / "tests/test_historical_rechecks.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NEW_CONTINUITY = "/root/new_continuity"


def complete_page_original(root, built, variant=None):
    """新v2 page fixture真的保存完整原判、原freeze與唯一實際收件；不是header空殼。"""
    manifest, _, _, sealed = built
    proof = manifest["pages"][0]["technical"]
    current = base.review.document(root, proof["report"])
    current["created_at"] = base.at(22)
    current_ref = base.save(root, "current/page-current-v2.json", current)
    original = copy.deepcopy(current)
    original["created_at"] = base.at(21)
    if variant == "shell":
        original = {
            key: original[key]
            for key in [
                "schema_version",
                "review_stage",
                "reviewer_context",
                "reviewer_task",
                "lesson_id",
                "created_at",
            ]
        }
        original.update(source="NO_SUCH_SOURCE.md", source_sha256="f" * 64, verdict="revise", issues=[])
    elif variant == "honest_revise":
        original["verdict"] = "revise"
        original["claims"][0].update(status="unresolved", evidence=[], artifact_ids=[])
        for value in original["checks"].values():
            value["status"] = "unresolved"
    elif variant in {"claims", "sources", "artifacts", "checks", "created_at"}:
        del original[variant]
    elif variant == "source":
        original["source"] = "NO_SUCH_SOURCE.md"
    elif variant == "source_sha":
        original["source_sha256"] = "f" * 64
    elif variant == "figures":
        original["figure_sha256"] = {}
    elif variant == "created_after_current":
        original["created_at"] = base.at(24)
    original_ref = base.save(root, "current/page-original-v2.json", original)
    proof.update(report=current_ref, original_report=original_ref, original_source_freeze=manifest["source_freeze"])
    if variant == "missing_freeze":
        del proof["original_source_freeze"]
    if variant == "original_pointer":
        proof["original_json_pointer"] = "/another_report"
    phase = sealed["new"]["technical"]
    for reference, received in [
        (original_ref, base.at(25) if variant in {"late_receipt", "created_after_current"} else base.at(21)),
        (current_ref, base.at(23)),
    ]:
        phase["report_sha256s"].append(reference["sha256"])
        phase["collection_index"].append({**reference, "received_at": received})
        manifest["issue_reports"].append(reference)
    sidecar = base.review.document(root, manifest["historical_fix_rechecks"])
    sidecar["rechecks"][0]["full_review"] = current_ref
    manifest["historical_fix_rechecks"] = base.save(root, "current/page-chain-v2.json", sidecar)
    return built


def add_unchanged_peer(root, built):
    """虛構舊問題跨兩頁，但此次只改第一頁；兩頁的銜接仍須本人新核。"""
    manifest, live, tasks, sealed = built
    old_sidecar = base.review.document(root, manifest["historical_fix_rechecks"])
    pointer_ref = old_sidecar["prior_active_pointer"]
    pointer = base.review.document(root, pointer_ref)
    prior_manifest = base.review.document(root, {"path": pointer["manifest"], "sha256": pointer["manifest_sha256"]})
    prior_freeze = base.review.document(root, prior_manifest["source_freeze"])
    current_freeze = base.review.document(root, manifest["source_freeze"])
    text = b"Before publishing, validate the saved version.\n"
    old_snapshot = base.save(root, "prior/publishing.md", text)
    snapshot = base.save(root, "current/publishing.md", text)
    base.save(root, "PUBLISHING.md", text)
    common = {
        "page_id": "publishing",
        "source": "PUBLISHING.md",
        "selector": "whole-file",
        "source_sha256": snapshot["sha256"],
        "figures_sha256": {},
    }
    prior_freeze["pages"].append({**common, "snapshot": old_snapshot["path"]})
    current_freeze["pages"].append({**common, "snapshot": snapshot["path"]})
    prior_manifest["source_freeze"] = base.save(root, "prior/peer-freeze.json", prior_freeze)
    manifest["source_freeze"] = base.save(root, "current/peer-freeze.json", current_freeze)
    live["publishing"] = current_freeze["pages"][-1]
    peer_fix = base.save(
        root,
        "prior/peer-fix.json",
        {
            "reviewer_task": base.OLD_PAGE,
            "page_id": "publishing",
            "source_sha256": common["source_sha256"],
            "figures_sha256": {},
            "verdict": "pass",
            "issues": [],
        },
    )
    dispositions = base.review.document(root, prior_manifest["dispositions"])
    dispositions["items"][0]["affected_pages"].append("publishing")
    dispositions["items"][0]["verification"].append({"report": peer_fix, "json_pointer": ""})
    prior_dispositions = base.save(root, "prior/peer-dispositions.json", dispositions)
    prior_manifest["dispositions"] = prior_dispositions
    prior_manifest_ref = base.save(root, "prior/peer-manifest.json", prior_manifest)
    prior_gate = base.review.document(root, pointer["actual_completed_gate"])
    prior_gate["records"].append({"page_id": "publishing", "source_sha256": common["source_sha256"]})
    pointer.update(
        manifest=prior_manifest_ref["path"],
        manifest_sha256=prior_manifest_ref["sha256"],
        actual_completed_gate=base.save(root, "prior/peer-gate.json", prior_gate),
    )
    base.save(root, "docs/course-reviews-active.json", pointer)
    old_sidecar["prior_active_pointer"] = base.save(
        root, "current/peer-previous-pointer.json", (root / "docs/course-reviews-active.json").read_bytes()
    )
    for group in old_sidecar["rechecks"]:
        previous = group["appendix"]
        appendix = base.review.document(root, previous)
        for row in appendix["historical_issue_rechecks"]:
            row["disposition_ref"].update(prior_dispositions)
        group["appendix"] = base.save(root, previous["path"], appendix)
        manifest["issue_reports"] = [group["appendix"] if ref == previous else ref for ref in manifest["issue_reports"]]
        phase = sealed["new"]["technical"]
        phase["report_sha256s"] = [
            group["appendix"]["sha256"] if h == previous["sha256"] else h for h in phase["report_sha256s"]
        ]
        for entry in phase["collection_index"]:
            if entry["path"] == previous["path"]:
                entry.update(group["appendix"])
    manifest["historical_fix_rechecks"] = base.save(root, "current/peer-old-sidecar.json", old_sidecar)
    current_dispositions = base.review.document(root, manifest["dispositions"])
    current_dispositions["items"] = copy.deepcopy(dispositions["items"])
    manifest["dispositions"] = base.save(root, "current/peer-current-dispositions.json", current_dispositions)
    manifest["pages"].append({"page_id": "publishing", "reader": {"mode": "inherit"}, "technical": {"mode": "inherit"}})
    return built


def fixture(root, variant=None, multi=False):
    old_variant = {
        "unavailable_missing": "not_unavailable",
        "unavailable_listing_nonempty": "nonempty_listing",
        "unanchored_prior": "unanchored_prior",
        "prior_gate_failed": "prior_gate_failed",
        "prior_source_differs": "prior_gate_source",
        "old_fix_not_pass": "old_fix_revise",
    }.get(variant)
    manifest, live, tasks, sealed = base.fixture(
        root, old_variant, old_role="technical" if variant == "wrong_old_role" else "continuity"
    )
    if multi:
        add_unchanged_peer(root, (manifest, live, tasks, sealed))
    old_sidecar = base.review.document(root, manifest["historical_fix_rechecks"])
    group = old_sidecar["rechecks"][0]
    old_appendix = base.review.document(root, group["appendix"])
    row = copy.deepcopy(old_appendix["historical_issue_rechecks"][0])
    page = live["readme"]
    proof = manifest["pages"][0]["technical"]
    technical = base.review.document(root, proof["report"])
    technical["created_at"] = base.at(22)
    full = base.save(root, "current/whole-technical-v2.json", technical)
    proof.update(report=full, original_report=full, original_source_freeze=manifest["source_freeze"])
    technical_refs = [full]
    if variant in {
        "technical_original_created_after_current",
        "technical_original_header_wrong_source",
        "technical_original_honest_revise",
    }:
        original_technical = copy.deepcopy(technical)
        if variant == "technical_original_created_after_current":
            original_technical["created_at"] = base.at(23)
        elif variant == "technical_original_header_wrong_source":
            original_technical = {
                k: technical[k]
                for k in (
                    "schema_version",
                    "review_stage",
                    "lesson_id",
                    "reviewer_task",
                    "reviewer_context",
                    "created_at",
                    "figure_sha256",
                )
            }
            original_technical.update(source="NO_SUCH_SOURCE.md", source_sha256="f" * 64, verdict="revise", issues=[])
        else:
            original_technical["verdict"] = "revise"
            original_technical["claims"][0].update(status="unresolved", evidence=[], artifact_ids=[])
            for value in original_technical["checks"].values():
                value["status"] = "unresolved"
            prior_pointer = base.review.document(root, old_sidecar["prior_active_pointer"])
            prior_manifest = base.review.document(
                root, {"path": prior_pointer["manifest"], "sha256": prior_pointer["manifest_sha256"]}
            )
            proof["original_source_freeze"] = prior_manifest["source_freeze"]
            original_technical["source_sha256"] = base.review.document(root, proof["original_source_freeze"])["pages"][
                0
            ]["source_sha256"]
        original_technical_ref = base.save(root, "current/technical-original-separate.json", original_technical)
        proof["original_report"] = original_technical_ref
        technical_refs.append(original_technical_ref)
    if variant == "technical_original_shell":
        shell = base.save(root, "current/technical-shell.json", {"reviewer_task": base.NEW_PAGE, "lesson_id": "readme"})
        proof["original_report"] = shell
        technical_refs.append(shell)
    if variant == "technical_not_whole":
        proof["mode"] = "section-composite"
    if variant == "technical_late_phase":
        proof["collection_role"] = "continuity"
    assignments = base.review.document(root, manifest["assignments"])["tasks"]
    reader_variant = "fresh_reader_missing_note" if variant == "reader_missing_note" else None
    reader_proof, reader_task, reader_refs = base.fresh_reader(root, page, manifest["source_freeze"], reader_variant)
    assignments.append(reader_task)
    tasks[base.NEW_READER] = "reader"
    if variant != "reader_absent":
        manifest["pages"][0]["reader"] = reader_proof
    if variant == "reader_stale":
        record = base.review.document(root, reader_proof["report"])
        record["source_sha256"] = "0" * 64
        changed = base.save(root, "current/stale-reader.json", record)
        reader_proof["report"] = changed
        reader_refs.append(changed)
    member = {
        "page_id": "readme",
        "source_sha256": page["source_sha256"],
        "figures_sha256": page["figures_sha256"],
        "verdict": "pass",
        "source_quote": "313 notebooks; local rendering remains fixed.",
        "checks": {"prerequisites": "Fictional complete route checked", "limits": "Fictional fixture only"},
        "issues": [],
    }
    members = [member]
    if multi:
        members.append(
            {
                "page_id": "publishing",
                "source_sha256": live["publishing"]["source_sha256"],
                "figures_sha256": {},
                "verdict": "pass",
                "source_quote": "Before publishing, validate the saved version.",
                "checks": {"prerequisites": "Fictional two-page dependency checked."},
                "issues": [],
            }
        )
    original_doc = {
        "schema_version": 1,
        "review_stage": "continuity",
        "reviewer_context": "fresh",
        "reviewer_task": NEW_CONTINUITY,
        "source_freeze": manifest["source_freeze"],
        "created_at": base.at(42),
        "verdict": "pass",
        "pages": copy.deepcopy(members),
        "issues": [],
    }
    if variant == "continuity_original_unbound_source":
        original_doc["pages"][0]["source_sha256"] = "f" * 64
    if variant == "continuity_original_schema_missing":
        del original_doc["schema_version"]
    if variant == "continuity_original_checks_empty":
        original_doc["pages"][0]["checks"] = {}
    if variant == "continuity_original_quote_absent":
        original_doc["pages"][0]["source_quote"] = "This original quote is absent."
    original = base.save(root, "current/continuity-original.json", original_doc)
    current_doc = copy.deepcopy(original_doc)
    current_doc["pages"] = copy.deepcopy(members)
    current_doc["created_at"] = base.at(43)
    if variant == "continuity_current_created_before_original":
        current_doc["created_at"] = base.at(41)
    if variant == "continuity_stale":
        current_doc["pages"][0]["source_sha256"] = "0" * 64
    if variant == "continuity_revise":
        current_doc["pages"][0]["verdict"] = "revise"
    current = base.save(root, "current/continuity-current.json", current_doc)
    row["pages"] = {
        "readme": {
            "source_sha256": page["source_sha256"],
            "figures_sha256": page["figures_sha256"],
            "source_quote": "313 notebooks; local rendering remains fixed.",
            "reason": "Fictional source retains the previously fixed local-rendering explanation.",
        }
    }
    row["current_evidence"] = [{k: old_appendix["artifacts"][0][k] for k in ("path", "sha256")}]
    if multi:
        row["pages"]["publishing"] = {
            "source_sha256": live["publishing"]["source_sha256"],
            "figures_sha256": {},
            "source_quote": "Before publishing, validate the saved version.",
            "reason": "Fictional reviewer reread the unchanged next page and checked this dependency.",
        }
    if variant == "quote_absent":
        row["pages"]["readme"]["source_quote"] = "This quote is absent."
    if variant == "peer_quote_absent":
        row["pages"]["publishing"]["source_quote"] = "This peer quote is absent."
    if variant == "peer_evidence_absent":
        del row["pages"]["publishing"]
    if variant == "issue_source_stale":
        row["pages"]["readme"]["source_sha256"] = "0" * 64
    if variant == "issue_reason_empty":
        row["reason"] = ""
    if variant == "downgrade_necessity":
        row["necessity"] = "optional"
    if variant == "missing_page_evidence":
        row["pages"] = {}
    if variant == "missing_evidence":
        row["current_evidence"] = []
    if variant == "historical_verification_replaced":
        row["historical_fix_verification"] = {"report": current, "json_pointer": "/pages/0"}
    appendix = base.save(
        root,
        "current/continuity-appendix.json",
        {
            "reviewer_task": NEW_CONTINUITY,
            "created_at": base.at(45),
            "verdict": "pass",
            "issues": [],
            "historical_issue_rechecks": [row, copy.deepcopy(row)] if variant == "duplicate_row" else [row],
        },
    )
    manifest["pages"][0]["continuity"] = {
        "mode": "review",
        "wave": "new",
        "report": current,
        "json_pointer": "/pages/0",
        "original_report": original,
        "original_json_pointer": "/pages/0",
    }
    if multi:
        manifest["pages"][1]["continuity"] = {
            "mode": "review",
            "wave": "new",
            "report": current,
            "json_pointer": "/pages/1",
            "original_report": original,
            "original_json_pointer": "/pages/1",
        }
    continuity_task = {
        "task": NEW_CONTINUITY,
        "role": "continuity",
        "context": "fresh",
        "wave": "new",
        "pages": ["readme"],
        "dispatch_window": {"not_before": base.at(41), "not_after": base.at(42)},
    }
    if multi:
        continuity_task["pages"].append("publishing")
    if variant == "continuity_shared":
        continuity_task["context"] = "shared"
    if variant == "outside_assignment":
        continuity_task["pages"] = []
    if variant == "author_is_continuity":
        manifest["author_tasks"].append(NEW_CONTINUITY)
    assignments.append(continuity_task)
    tasks[NEW_CONTINUITY] = "continuity"
    manifest["assignments"] = base.save(root, "current/assignments-v2.json", {"tasks": assignments})
    group = {
        "kind": "continuity",
        "full_review": current,
        "original_review": original,
        "original_source_freeze": manifest["source_freeze"],
        "appendix": appendix,
        "unavailability": group["unavailability"],
        "collection": {"wave": "new", "collection_role": "continuity", "received_at": base.at(46)},
    }
    if variant == "appendix_as_full":
        group["full_review"] = appendix
    if variant == "wrong_collection":
        group["collection"]["collection_role"] = "technical"
    if variant == "false_receipt":
        group["collection"]["received_at"] = base.at(47)
    chain = base.save(
        root,
        "current/chain-v1.json",
        {
            "schema_version": 1,
            "prior_active_pointer": old_sidecar["prior_active_pointer"],
            "rechecks": [group, old_sidecar["rechecks"][1]],
        },
    )
    if variant == "nested_chain":
        chain = base.save(root, "current/nested-chain.json", {"schema_version": 2, "chains": [chain]})
    sidecar = {
        "schema_version": True if variant == "boolean_schema" else 2,
        "chains": [chain, chain] if variant == "duplicate_chain" else [chain],
    }
    manifest["historical_fix_rechecks"] = base.save(root, "current/history-v2.json", sidecar)
    for ref in technical_refs:
        sealed["new"]["technical"]["report_sha256s"].append(ref["sha256"])
        sealed["new"]["technical"]["collection_index"].append({**ref, "received_at": base.at(23)})
    phase = sealed["new"]["reader"]
    phase["report_sha256s"] = [r["sha256"] for r in reader_refs]
    phase["collection_index"] = [
        {**r, "received_at": base.at(7 if "original" in r["path"] else 9)} for r in reader_refs
    ]
    phase = sealed["new"]["continuity"]
    continuity_refs = [original, current, appendix]
    phase["report_sha256s"] = [r["sha256"] for r in continuity_refs]
    phase["collection_index"] = [
        {**r, "received_at": base.at(43 + i if i < 2 else 46)} for i, r in enumerate(continuity_refs)
    ]
    if variant == "unsealed_continuity":
        phase["report_sha256s"].remove(current["sha256"])
    if variant == "stages_overlap":
        phase["started_at"] = base.at(35)
    if variant == "belated_original":
        phase["collection_index"][0]["received_at"] = base.at(47)
    for role, phase in sealed["new"].items():
        manifest["review_waves"][1]["seals"][role] = base.save(root, "new/seal-v2-" + role + ".json", phase)
    manifest["issue_reports"].extend([*technical_refs, *reader_refs, *continuity_refs])
    dispositions = base.review.document(root, manifest["dispositions"])
    dispositions["at"] = base.at(47)
    manifest["dispositions"] = base.save(root, "current/dispositions-v2.json", dispositions)
    manifest["dispositions_collection"] = {"wave": "new", "role": "continuity", "received_at": base.at(48)}
    return manifest, live, tasks, sealed


def check(root, built):
    manifest, live, tasks, sealed = built
    return base.review.check_dispositions(root, manifest, ("reader", "technical", "continuity"), live, tasks, sealed)


def test_continuity_recheck_needs_real_complete_three_stage_evidence(tmp_path):
    built = fixture(tmp_path)
    prior = (tmp_path / "prior/dispositions.json").read_bytes()
    old_report = (tmp_path / "prior/fixed.json").read_bytes()
    assert check(tmp_path, built) == {}
    assert (tmp_path / "prior/dispositions.json").read_bytes() == prior
    assert (tmp_path / "prior/fixed.json").read_bytes() == old_report


def test_composite_keeps_legacy_checks(tmp_path):
    built = complete_page_original(tmp_path, base.fixture(tmp_path))
    built[0]["historical_fix_rechecks"] = base.save(
        tmp_path,
        "current/composite.json",
        {
            "schema_version": 2,
            "chains": [built[0]["historical_fix_rechecks"]],
        },
    )
    assert base.check(tmp_path, built) == {}


def test_one_changed_page_does_not_erase_two_page_dependency(tmp_path):
    assert check(tmp_path, fixture(tmp_path, multi=True)) == {"publishing": set()}


def test_honest_revise_original_can_keep_its_authenticated_earlier_version(tmp_path):
    assert check(tmp_path, fixture(tmp_path, "technical_original_honest_revise")) == {}


def add_original_program_snapshot(root, built, variant=None):
    """真舊bytes只供原判查核；當前報告仍必須引用當前程式。"""
    manifest, _, _, sealed = built
    proof = manifest["pages"][0]["technical"]
    original = base.review.document(root, proof["original_report"])
    old_raw = b"print(1)\n"
    snapshot = base.save(root, "current/snapshots/program-before.py", old_raw)
    base.save(root, "scripts/demo.py", b"print(2)\n")
    original["sources"].append(
        {
            "id": "old-program",
            "kind": "repository_code",
            "title": "Fictional program actually read before correction",
            "path": "scripts/demo.py",
            "sha256": base.sha(old_raw),
            "version": "Fictional immutable original version",
            "verified": True,
            "inspection_note": "Fictional reviewer read the complete old print statement.",
        }
    )
    original_ref = base.save(root, "current/original-with-program.json", original)
    proof["original_report"] = original_ref
    mapping = {"source": "scripts/demo.py", "snapshot": snapshot}
    proof["original_source_snapshots"] = [mapping]
    if variant == "absent":
        del proof["original_source_snapshots"]
    elif variant == "wrong_reference_hash":
        mapping["snapshot"] = {**snapshot, "sha256": "f" * 64}
    elif variant == "wrong_bytes":
        mapping["snapshot"] = base.save(root, "current/snapshots/wrong.py", b"print(3)\n")
    elif variant == "unregistered_path":
        mapping["source"] = "scripts/other.py"
    elif variant == "duplicate":
        proof["original_source_snapshots"].append(copy.deepcopy(mapping))
    elif variant == "extra_field":
        mapping["earlier_files"] = {"scripts/demo.py": base.sha(old_raw)}
    elif variant == "escape":
        mapping["source"] = "../outside.py"
    elif variant == "as_current":
        current = base.review.document(root, proof["report"])
        current["sources"].append(copy.deepcopy(original["sources"][-1]))
        proof["report"] = base.save(root, "current/stale-current-program.json", current)
        manifest["issue_reports"].append(proof["report"])
        sealed["new"]["technical"]["report_sha256s"].append(proof["report"]["sha256"])
        sealed["new"]["technical"]["collection_index"].append({**proof["report"], "received_at": base.at(23)})
    manifest["issue_reports"].append(original_ref)
    sealed["new"]["technical"]["report_sha256s"].append(original_ref["sha256"])
    sealed["new"]["technical"]["collection_index"].append({**original_ref, "received_at": base.at(23)})
    return built


def test_original_program_requires_registered_fingerprint_and_saved_bytes(tmp_path):
    built = add_original_program_snapshot(tmp_path, fixture(tmp_path))
    assert check(tmp_path, built) == {}


@pytest.mark.parametrize(
    "variant",
    [
        "absent",
        "wrong_reference_hash",
        "wrong_bytes",
        "unregistered_path",
        "duplicate",
        "extra_field",
        "escape",
        "as_current",
    ],
)
def test_original_program_snapshot_cannot_admit_unbound_or_stale_current_code(tmp_path, variant):
    with pytest.raises((ValueError, KeyError)):
        check(tmp_path, add_original_program_snapshot(tmp_path, fixture(tmp_path), variant))


@pytest.mark.parametrize("bad_value", [False, {}, "", "scripts/demo.py"])
def test_original_program_snapshots_require_list_even_when_empty(tmp_path, bad_value):
    built = fixture(tmp_path)
    built[0]["pages"][0]["technical"]["original_source_snapshots"] = bad_value
    with pytest.raises(ValueError):
        check(tmp_path, built)


@pytest.mark.parametrize(
    "variant",
    [
        "missing_page_evidence",
        "quote_absent",
        "peer_quote_absent",
        "peer_evidence_absent",
        "continuity_stale",
        "outside_assignment",
    ],
)
def test_two_page_dependency_requires_both_current_pages(tmp_path, variant):
    with pytest.raises((ValueError, KeyError)):
        check(tmp_path, fixture(tmp_path, variant, multi=True))


def test_two_authenticated_generations_can_preserve_distinct_obligations(tmp_path):
    built = complete_page_original(tmp_path, base.fixture(tmp_path))
    manifest = built[0]
    sidecar = base.review.document(tmp_path, manifest["historical_fix_rechecks"])
    prior = base.review.document(tmp_path, sidecar["prior_active_pointer"])
    later = {**prior, "activated_at": "2026-09-03T00:00:00+00:00", "previous_pointer": sidecar["prior_active_pointer"]}
    base.save(tmp_path, "docs/course-reviews-active.json", later)
    later_ref = base.save(
        tmp_path, "current/later-pointer.json", (tmp_path / "docs/course-reviews-active.json").read_bytes()
    )
    first = base.save(tmp_path, "current/generation-one.json", {**sidecar, "rechecks": sidecar["rechecks"][:1]})
    second = base.save(
        tmp_path,
        "current/generation-two.json",
        {**sidecar, "prior_active_pointer": later_ref, "rechecks": sidecar["rechecks"][1:]},
    )
    manifest["historical_fix_rechecks"] = base.save(
        tmp_path, "current/two-generations.json", {"schema_version": 2, "chains": [first, second]}
    )
    assert base.check(tmp_path, built) == {}


@pytest.mark.parametrize("variant", [None, "honest_revise"])
def test_v2_page_retains_complete_original_even_when_original_requests_revision(tmp_path, variant):
    built = complete_page_original(tmp_path, base.fixture(tmp_path), variant)
    built[0]["historical_fix_rechecks"] = base.save(
        tmp_path, "current/page-composite.json", {"schema_version": 2, "chains": [built[0]["historical_fix_rechecks"]]}
    )
    assert base.check(tmp_path, built) == {}


@pytest.mark.parametrize(
    "variant",
    [
        "shell",
        "claims",
        "sources",
        "artifacts",
        "checks",
        "created_at",
        "source",
        "source_sha",
        "figures",
        "created_after_current",
        "late_receipt",
        "missing_freeze",
        "original_pointer",
    ],
)
def test_v2_page_rejects_incomplete_unbound_or_reversed_originals(tmp_path, variant):
    built = complete_page_original(tmp_path, base.fixture(tmp_path), variant)
    built[0]["historical_fix_rechecks"] = base.save(
        tmp_path, "current/page-composite.json", {"schema_version": 2, "chains": [built[0]["historical_fix_rechecks"]]}
    )
    with pytest.raises(ValueError):
        base.check(tmp_path, built)


@pytest.mark.parametrize(
    "variant",
    [
        "reader_absent",
        "reader_missing_note",
        "reader_stale",
        "technical_not_whole",
        "technical_late_phase",
        "technical_original_shell",
        "technical_original_created_after_current",
        "technical_original_header_wrong_source",
        "continuity_current_created_before_original",
        "continuity_original_unbound_source",
        "continuity_original_schema_missing",
        "continuity_original_checks_empty",
        "continuity_original_quote_absent",
        "continuity_stale",
        "continuity_revise",
        "continuity_shared",
        "outside_assignment",
        "author_is_continuity",
        "quote_absent",
        "issue_source_stale",
        "issue_reason_empty",
        "downgrade_necessity",
        "missing_page_evidence",
        "missing_evidence",
        "historical_verification_replaced",
        "duplicate_row",
        "appendix_as_full",
        "wrong_collection",
        "false_receipt",
        "duplicate_chain",
        "nested_chain",
        "boolean_schema",
        "unsealed_continuity",
        "stages_overlap",
        "belated_original",
        "unavailable_missing",
        "unavailable_listing_nonempty",
        "unanchored_prior",
        "prior_gate_failed",
        "prior_source_differs",
        "old_fix_not_pass",
        "wrong_old_role",
    ],
)
def test_incomplete_or_forged_continuity_history_is_rejected(tmp_path, variant):
    with pytest.raises((ValueError, KeyError)):
        check(tmp_path, fixture(tmp_path, variant))
