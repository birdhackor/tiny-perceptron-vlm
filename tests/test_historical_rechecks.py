"""虛構舊 closed 修正與新 fresh 重審；不使用真實 reviewer 報告當測試答案。

核對 stale 的預設拒絕、歷史身分鏈、完整新主 proof、逐項 appendix、工具防護
與實際收件。真正的 technical _validate 會執行；不 mock 成永遠 pass。
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


review = load("historical_test_gate", ROOT / "docs/review-tools/incremental_revision.py")
inventory = load("historical_test_inventory", ROOT / "docs/course-review-20261019/tools/issue_inventory.py")
OLD_PAGE = "/root/old_page"
OLD_TOOL = "/root/old_tool-with-hyphen"
NEW_PAGE = "/root/new_page"
NEW_TOOL = "/root/new_tool"
NEW_READER = "/root/new_reader"
TOOL_FILES = {
    "docs/review-tools/incremental_revision.py",
    "docs/review-tools/baseline_replay.py",
    "docs/review-tools/phase7_review.py",
    "docs/course-revision-20261016/tools/reader.py",
    "docs/course-revision-20261016/tools/freeze_sources.py",
    "tests/test_incremental_revision_review.py",
} | review.HISTORICAL_INPUTS
AFFECTED = ["docs/review-tools/incremental_revision.py", "tests/test_incremental_revision_review.py"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(root, name, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False) + "\n").encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": name, "sha256": sha(raw)}


def at(minute, old=False):
    return f"2026-{'09' if old else '10'}-01T00:{minute:02d}:00+00:00"


def assignment(task, lesson, old=False):
    return {
        "task": task,
        "role": "technical",
        "wave": "old" if old else "new",
        "context": "fresh",
        "lesson_id": lesson,
        "dispatch_window": {"not_before": at(21, old), "not_after": at(22, old)},
    }


def decision(issue, verification, tool=False):
    return {
        "issue_id": "GUARD" if tool else "MATH",
        "report": issue,
        "json_pointer": "/issues/0",
        "final_necessity": "necessary",
        "action": "fixed",
        **{
            name: "虛構測試：保留以前已修正的必要問題"
            for name in ("source_support", "learning_benefit", "change_burden", "reason")
        },
        "verification": [verification],
        **({"affected_files": AFFECTED} if tool else {"affected_pages": ["readme"]}),
    }


def fresh_reader(root, page, freeze, variant):
    """使用真實切段與 closure validator 的完整虛構 reader trace。"""
    save(
        root, "docs/review-tools/incremental_reader.py", (ROOT / "docs/review-tools/incremental_reader.py").read_bytes()
    )
    reader = load("historical_fixture_reader", root / "docs/course-revision-20261016/tools/reader.py")
    plan = save(
        root,
        "current/reader-plan.json",
        {
            "reviewer_task": NEW_READER,
            "author_task": "/root",
            "created_at": at(0),
            "main_pages": ["readme"],
            "context_pages": [],
            "source_freeze": freeze,
        },
    )
    opening = save(
        root,
        "current/reader-opening.json",
        {
            "reviewer_task": NEW_READER,
            "page_id": "readme",
            "started_at": at(3),
            "source_sha256": page["source_sha256"],
            "figures_sha256": page["figures_sha256"],
            "reading_plan": plan,
            "source_freeze": freeze,
        },
    )
    notes = []
    for index, chunk in enumerate(reader.units((root / page["snapshot"]).read_text())):
        note = {
            "reviewer_task": NEW_READER,
            "page_id": "readme",
            "unit_index": index,
            "unit_sha256": sha(chunk.encode()),
            "saved_at": at(4 + index),
            **{
                field: "虛構完整逐段觀察"
                for field in (
                    "understanding",
                    "materials_and_labels",
                    "expected_change",
                    "confusion_and_quote",
                    "missing_visuals",
                )
            },
            "concept_checks": [
                {
                    **{
                        field: "虛構概念核對"
                        for field in (
                            "concept",
                            "identity",
                            "mechanism",
                            "need",
                            "usage",
                            "example_scope",
                            "source_quote",
                        )
                    },
                    "verdict": "已足夠",
                }
            ],
        }
        if variant == "fresh_reader_unclosed_checkpoint":
            note["concept_checks"][0]["verdict"] = "需補說明"
        notes.append(save(root, f"current/reader-note-{index}.json", note))
    original_doc = {
        "reviewer_task": OLD_PAGE if variant == "fresh_reader_original_owner" else NEW_READER,
        "created_at": at(6),
        "page_id": "readme",
        "source_sha256": page["source_sha256"],
        "figures_sha256": page["figures_sha256"],
        "verdict": "pass",
        "issues": [],
    }
    original = save(root, "current/reader-original.json", original_doc)
    current_doc = {
        **original_doc,
        "reviewer_task": NEW_TOOL if variant == "fresh_reader_wrong_owner" else NEW_READER,
        "created_at": at(8),
        "basis_report": original,
        "ai_estimate": {"minutes_min": 1, "minutes_max": 2, "reason": "虛構全文閱讀估計"},
    }
    if variant == "fresh_reader_stale":
        current_doc["source_sha256"] = sha((root / "prior/readme.md").read_bytes())
    if variant == "fresh_reader_figures":
        current_doc["figures_sha256"] = {}
    current = save(root, "current/reader-current.json", current_doc)
    if variant == "fresh_reader_missing_note":
        notes.pop()
    frozen = json.loads((root / freeze["path"]).read_bytes())
    proof = {
        "mode": "inherit" if variant == "fresh_reader_inherited" else "review",
        "wave": "old" if variant == "fresh_reader_wrong_wave" else "new",
        "report": current,
        "json_pointer": "",
        "original_report": original,
        "original_json_pointer": "",
        "basis": {"report": original, "json_pointer": ""},
        "sessions": [
            {
                "wave": "new",
                "collection_role": "reader",
                "opening": opening,
                "notes": notes,
                "source": {"path": page["snapshot"], "sha256": page["source_sha256"]},
                "figures": frozen["figure_snapshots"],
            }
        ],
    }
    task = {
        "task": NEW_READER,
        "role": "reader",
        "context": "shared" if variant == "fresh_reader_shared" else "fresh",
        "wave": "new",
        "pages": ["readme"],
        "dispatch_window": {
            "not_before": at(11 if variant == "fresh_reader_bad_dispatch" else 1),
            "not_after": at(12 if variant == "fresh_reader_bad_dispatch" else 2),
        },
    }
    return proof, task, [original, current]


def fixture(root, variant=None, only_tool=False, old_role="technical"):
    for name in TOOL_FILES | {"scripts/check_technical_reviews.py"}:
        save(root, name, (ROOT / name).read_bytes())
    figure = save(root, "figure.svg", b"<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    figures = {"figure.svg": figure["sha256"]}
    old_figure = save(root, "prior/figure.svg", (root / "figure.svg").read_bytes())
    before = save(root, "prior/readme.md", b"312 notebooks; local rendering remains fixed.\n\n![figure](figure.svg)\n")
    after = save(root, "current/readme.md", b"313 notebooks; local rendering remains fixed.\n\n![figure](figure.svg)\n")
    save(root, "README.md", (root / after["path"]).read_bytes())
    common = {"page_id": "readme", "source": "README.md", "selector": "whole-file", "figures_sha256": figures}
    old_page = {**common, "source_sha256": before["sha256"], "snapshot": before["path"]}
    current_page = {**common, "source_sha256": after["sha256"], "snapshot": after["path"]}
    prior_freeze = save(
        root, "prior/freeze.json", {"pages": [old_page], "figure_snapshots": {"figure.svg": old_figure}}
    )
    current_freeze = save(
        root,
        "current/freeze.json",
        {"created_at": at(0), "pages": [current_page], "figure_snapshots": {"figure.svg": figure}},
    )
    original = save(
        root,
        "prior/original.json",
        {"reviewer_task": OLD_PAGE, "issues": [] if only_tool else [{"id": "MATH", "necessary": old_role != "reader"}]},
    )
    old_fix = save(
        root,
        "prior/fixed.json",
        {
            "reviewer_task": OLD_PAGE,
            "lesson_id": "readme",
            "verdict": "revise" if variant == "old_fix_revise" else "pass",
            "source_sha256": before["sha256"],
            "figure_sha256": {} if variant == "old_figure" else figures,
        },
    )
    old_tool_hashes = {name: sha(("historical code " + name).encode()) for name in AFFECTED}
    if variant == "unused_tool_recheck":
        old_tool_hashes = {name: sha((root / name).read_bytes()) for name in AFFECTED}
    old_tool = save(
        root,
        "prior/tool.json",
        {
            "reviewer_task": OLD_TOOL,
            "verdict": "pass",
            "resolved_issue_ids": ["GUARD"],
            "checked_files": [{"path": name, "sha256": value} for name, value in old_tool_hashes.items()],
            "issues": [{"id": "GUARD", "status": "resolved"}],
        },
    )
    page_decision = decision(original, {"report": old_fix, "json_pointer": ""})
    if old_role == "reader":
        page_decision["final_necessity"] = "optional"
    tool_decision = decision(old_tool, {"report": old_tool, "kind": "tool_audit"}, True)
    old_items = ([page_decision] if not only_tool else []) + [tool_decision]
    if variant == "prior_unclosed":
        old_items[0]["action"] = "defer"
    prior_dispositions = save(
        root, "prior/dispositions.json", {"at": at(45, True), "pending_count": 0, "items": old_items}
    )
    old_tasks = [assignment(OLD_PAGE, "readme", True), assignment(OLD_TOOL, "__review_tools__", True)]
    old_tasks[0]["role"] = old_role
    if old_role != "technical":
        old_tasks[0]["pages"] = [old_tasks[0].pop("lesson_id")]
    if variant == "tool_prior_reader":
        old_tasks[1]["role"] = "reader"
    prior_manifest = save(
        root,
        "prior/manifest.json",
        {
            "schema_version": 2,
            "review_policy": review.POLICY,
            "source_freeze": prior_freeze,
            "dispositions": prior_dispositions,
            "implementation": old_tool_hashes,
            "assignments": save(root, "prior/assignments.json", {"tasks": old_tasks}),
        },
    )
    prior_gate = save(
        root,
        "prior/gate.json",
        {
            "schema_version": 2,
            "review_policy": review.POLICY,
            "stage": "all",
            "status": "failed" if variant == "prior_gate_failed" else "passed_metadata_checks",
            "failures": [],
            "records": [
                {
                    "page_id": "readme",
                    "source_sha256": after["sha256"] if variant == "prior_gate_source" else before["sha256"],
                }
            ],
        },
    )
    pointer = {
        "schema_version": 2,
        "review_policy": review.POLICY,
        "manifest": prior_manifest["path"],
        "manifest_sha256": prior_manifest["sha256"],
        "actual_completed_gate": prior_gate,
        "activated_at": "2026-09-02T00:00:00+00:00",
    }
    save(root, "docs/course-reviews-active.json", pointer)
    backup = save(
        root,
        "current/previous-pointer.json",
        {**pointer, "extra": "forged"}
        if variant == "unanchored_prior"
        else (root / "docs/course-reviews-active.json").read_bytes(),
    )
    if variant == "prior_bad_hash":
        backup["sha256"] = "0" * 64
    unavailable = []
    for old_task in (OLD_PAGE, OLD_TOOL):
        response = (
            f"live agent path `{old_task}` not found"
            if old_task == OLD_PAGE
            else "agent_name must use only lowercase letters, digits, and underscores"
        )
        unavailable.append(
            {
                "reviewer_task": old_task,
                "unavailable": variant != "not_unavailable",
                "recorded_at": at(15),
                "reason": "虛構的實際失敗回應與空列表；兩種回應保持不同",
                "events": [
                    {"tool": "collaboration.followup_task", "request": {"target": old_task}, "response": response},
                    {
                        "tool": "collaboration.list_agents",
                        "request": {"path_prefix": "/root/old_"},
                        "response": {"agents": ["still present"] if variant == "nonempty_listing" else []},
                    },
                ],
            }
        )
    unavailability = save(root, "current/unavailable.json", {"author_task": "/root", "records": unavailable})
    calculation = save(
        root, "docs/technical-reviews/artifacts/calculation.txt", b"Synthetic fixture calculation: 312 + 1 = 313.\n"
    )
    source = {"id": "sum", "kind": "derivation", "title": "虛構推導", "verified": True, "details": "312+1=313"}
    evidence = {"source_id": "sum", "locator": "calculation", "supports": "新增一節後為 313"}
    claim = {
        "id": "count",
        "kind": "numeric",
        "statement": "312+1=313",
        "location": "first line",
        "scope": "虛構樣例",
        "status": "unresolved" if variant == "unverified_full_claim" else "verified",
        "evidence": [evidence],
        "artifact_ids": ["calculation"],
        "verification": {
            "method": "hand_calculation",
            "expected": "313",
            "observed": "313",
            "details": "整數加法",
            "tolerance": "exact",
        },
    }
    full_doc = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "readme",
        "source": "README.md",
        "source_sha256": after["sha256"],
        "figure_sha256": figures,
        "reviewer_task": NEW_PAGE,
        "reviewer_context": "fresh",
        "verdict": "pass",
        "issues": [],
        "claims": [claim],
        "sources": [source],
        "artifacts": [{"id": "calculation", "kind": "derivation", **calculation, "description": "虛構測試推導"}],
        "checks": {
            name: {"status": "pass", "details": "虛構逐項核對", "claim_ids": ["count"]}
            for name in (
                "factual_accuracy",
                "numeric_verification",
                "figure_consistency",
                "source_verification",
                "limitations",
            )
        },
        "review_scope": {"read_complete": True},
    }
    full = save(root, "current/full.json", full_doc)
    item = {
        "issue_id": "MATH",
        "disposition_ref": {**prior_dispositions, "json_pointer": "/items/0"},
        "original_report": original,
        "json_pointer": "/issues/0",
        "historical_fix_verification": {"report": old_fix, "json_pointer": ""},
        "current_source_sha256": before["sha256"] if variant == "item_stale" else after["sha256"],
        "current_figure_sha256": figures,
        "verdict": "pass",
        "status": "still_fixed",
        "necessity": page_decision["final_necessity"],
        "reason": "虛構問題逐筆複查",
        "current_evidence": [{**evidence, "source_id": "unknown" if variant == "unknown_evidence" else "sum"}],
        "artifact_ids": ["calculation"],
    }
    if variant == "unknown_prior_issue":
        item["disposition_ref"]["json_pointer"] = "/items/99"
    appendix_doc = {**copy.deepcopy(full_doc), "at": at(25), "historical_issue_rechecks": [item]}
    if variant == "different_owner":
        appendix_doc["reviewer_task"] = NEW_TOOL
    if variant == "stale_appendix":
        appendix_doc["source_sha256"] = before["sha256"]
    if variant == "figure_mismatch":
        appendix_doc["figure_sha256"] = {}
    if variant == "duplicate_recheck":
        appendix_doc["historical_issue_rechecks"].append(copy.deepcopy(item))
    appendix = save(root, "current/appendix.json", appendix_doc)
    if variant == "appendix_as_full":
        full = appendix
    current_tool_hashes = {name: sha((root / name).read_bytes()) for name in TOOL_FILES}
    full_tool = save(
        root,
        "current/tool-audit.json",
        {
            "reviewer_task": NEW_TOOL,
            "verdict": "pass",
            "issues": [],
            "checked_files": [{"path": name, "sha256": value} for name, value in current_tool_hashes.items()],
            "resolved_issue_ids": [] if variant == "tool_missing_issue" else ["GUARD"],
        },
    )
    tool_item = {
        "issue_id": "GUARD",
        "disposition_ref": {**prior_dispositions, "json_pointer": f"/items/{0 if only_tool else 1}"},
        "original_report": old_tool,
        "json_pointer": "/issues/0",
        "historical_fix_verification": {"report": old_tool, "kind": "tool_audit"},
        "verdict": "pass",
        "status": "still_fixed",
        "reason": "虛構防護反例仍拒絕",
        "current_files_sha256": {name: current_tool_hashes[name] for name in AFFECTED},
        "current_evidence": [calculation],
    }
    if variant == "tool_stale_files":
        tool_item["current_files_sha256"][AFFECTED[0]] = "0" * 64
    tool_appendix = save(
        root,
        "current/tool-appendix.json",
        {
            "reviewer_task": NEW_TOOL,
            "verdict": "pass",
            "at": at(25),
            "issues": [],
            "historical_issue_rechecks": [tool_item],
        },
    )
    original_tool = full_tool
    if variant == "tool_appendix_as_full":
        merged = json.loads((root / tool_appendix["path"]).read_bytes())
        merged.update(
            {
                "checked_files": [{"path": name, "sha256": value} for name, value in current_tool_hashes.items()],
                "resolved_issue_ids": ["GUARD"],
            }
        )
        tool_appendix = save(root, tool_appendix["path"], merged)
        full_tool = tool_appendix
    if variant == "tool_appendix_as_original":
        original_tool = tool_appendix
    if variant == "tool_incomplete_original":
        original_tool = save(
            root,
            "current/incomplete-tool-original.json",
            {"reviewer_task": NEW_TOOL, "verdict": "pass", "checked_files": []},
        )
    groups = []
    if not only_tool:
        groups.append(
            {
                "kind": "page",
                "page_id": "readme",
                "full_review": full,
                "appendix": appendix,
                "unavailability": {"report": unavailability, "json_pointer": "/records/0"},
                "collection": {"wave": "new", "collection_role": "technical", "received_at": at(26)},
            }
        )
    groups.append(
        {
            "kind": "tool_audit",
            "full_review": full_tool,
            "original_review": original_tool,
            "appendix": tool_appendix,
            "unavailability": {"report": unavailability, "json_pointer": "/records/1"},
            "collection": {"wave": "new", "collection_role": "technical", "received_at": at(26)},
        }
    )
    if variant == "missing_page_group":
        groups = groups[1:]
    if variant == "tool_missing_original":
        del groups[-1]["original_review"]
    if variant == "wrong_collection":
        groups[0]["collection"]["received_at"] = at(19)
    if variant == "false_receipt":
        groups[0]["collection"]["received_at"] = at(27)
    sidecar = save(
        root,
        "current/historical-rechecks.json",
        {"schema_version": 1, "prior_active_pointer": backup, "rechecks": groups},
    )
    current_items = copy.deepcopy(old_items)
    if variant == "changed_disposition":
        current_items[0]["final_necessity"] = "optional"
    tasks = old_tasks + [assignment(NEW_PAGE, "readme"), assignment(NEW_TOOL, "__review_tools__")]
    reader_proof, reader_refs = None, []
    if old_role == "reader" and variant != "fresh_reader_absent":
        reader_proof, reader_task, reader_refs = fresh_reader(root, current_page, current_freeze, variant)
        tasks.append(reader_task)
    if variant == "not_fresh":
        tasks[-2]["context"] = "shared"
    if variant == "wrong_page_assignment":
        tasks[-2]["lesson_id"] = "other"
    waves, sealed = [], {}
    collected = list({ref["path"]: ref for ref in (full, appendix, full_tool, tool_appendix)}.values())
    for wave, old in (("old", True), ("new", False)):
        seals, sealed[wave] = {}, {}
        for role, start, end in (("reader", 0, 10), ("technical", 20, 40), ("continuity", 41, 50)):
            refs = collected if wave == "new" and role == "technical" else []
            if wave == "new" and role == "reader":
                refs = reader_refs
            phase = {
                "role": role,
                "status": "completed",
                "started_at": at(start, old),
                "completed_at": at(end, old),
                "report_sha256s": [ref["sha256"] for ref in refs],
                "collection_index": [
                    {
                        **ref,
                        "received_at": at(
                            (7 if "original" in ref["path"] else 9)
                            if role == "reader"
                            else (26 if "appendix" in ref["path"] else 23)
                        ),
                    }
                    for ref in refs
                ],
            }
            if variant == "fresh_reader_unsealed" and wave == "new" and role == "reader":
                phase["report_sha256s"].remove(reader_refs[-1]["sha256"])
            if variant == "not_sealed" and wave == "new" and role == "technical":
                phase["report_sha256s"].remove(appendix["sha256"])
            if variant == "full_predates_dispatch" and wave == "new" and role == "technical":
                for entry in phase["collection_index"]:
                    if entry["path"] == full["path"]:
                        entry["received_at"] = at(19)
            sealed[wave][role] = phase
            seals[role] = save(root, f"{wave}/seal-{role}.json", phase)
        waves.append({"id": wave, "seals": seals})
    proof = {
        "mode": "review",
        "wave": "new",
        "report": full,
        "json_pointer": "",
        "original_report": full,
        "original_json_pointer": "",
    }
    if variant == "not_main_proof":
        proof["report"] = save(root, "current/different-full.json", full_doc)
    if variant == "appendix_as_original":
        proof["original_report"] = appendix
    manifest = {
        "schema_version": 2,
        "review_policy": review.POLICY,
        "author_tasks": ["/root"],
        "source_freeze": current_freeze,
        "assignments": save(root, "current/assignments.json", {"tasks": tasks}),
        "review_waves": waves,
        "pages": [{"page_id": "readme", "technical": proof}],
        "tool_audit": {"wave": "new", "report": full_tool},
        "issue_reports": [original, old_tool, full, appendix, full_tool, tool_appendix, *reader_refs],
        "dispositions": save(
            root, "current/dispositions.json", {"at": at(28), "pending_count": 0, "items": current_items}
        ),
        "dispositions_collection": {"wave": "new", "role": "technical", "received_at": at(29)},
        "historical_fix_rechecks": sidecar,
    }
    if reader_proof:
        manifest["pages"][0]["reader"] = reader_proof
    if variant == "author_is_reviewer":
        manifest["author_tasks"].append(NEW_PAGE)
    if variant == "missing_issue_report":
        manifest["issue_reports"].remove(appendix)
    if variant == "tamper_appendix":
        (root / appendix["path"]).write_text("{}")  # 刻意不更新已收件的 hash。
    return manifest, {"readme": current_page}, {a["task"]: a["role"] for a in tasks}, sealed


def check(root, built):
    manifest, live, tasks, sealed = built
    return review.check_dispositions(root, manifest, ("technical",), live, tasks, sealed)


def test_source_change_still_fails_without_sidecar(tmp_path):
    built = fixture(tmp_path)
    del built[0]["historical_fix_rechecks"]
    with pytest.raises(ValueError, match="fix recheck is stale"):
        check(tmp_path, built)


def test_tool_change_still_fails_without_sidecar(tmp_path):
    built = fixture(tmp_path, only_tool=True)
    del built[0]["historical_fix_rechecks"]
    with pytest.raises(ValueError, match="fix does not cover affected pages/files"):
        check(tmp_path, built)


def test_complete_fresh_rechecks_preserve_old_closures(tmp_path):
    built = fixture(tmp_path)
    original = (tmp_path / "prior/dispositions.json").read_bytes()
    original_report = (tmp_path / "prior/fixed.json").read_bytes()
    assert check(tmp_path, built) == {}
    assert (tmp_path / "prior/dispositions.json").read_bytes() == original
    assert (tmp_path / "prior/fixed.json").read_bytes() == original_report
    assert review.document(tmp_path, built[0]["dispositions"])["items"] == json.loads(original)["items"]


def test_old_reader_fix_needs_both_fresh_full_reader_and_technical(tmp_path):
    built = fixture(tmp_path, old_role="reader")
    prior = (tmp_path / "prior/dispositions.json").read_bytes()
    assert check(tmp_path, built) == {}
    assert (tmp_path / "prior/dispositions.json").read_bytes() == prior
    assert review.document(tmp_path, built[0]["dispositions"])["items"][0]["final_necessity"] == "optional"


@pytest.mark.parametrize(
    "variant, message",
    [
        ("fresh_reader_absent", "missing same-wave fresh reader"),
        ("fresh_reader_inherited", "missing same-wave fresh reader"),
        ("fresh_reader_wrong_wave", "missing same-wave fresh reader"),
        ("fresh_reader_stale", "fresh reader current source/figure/verdict stale"),
        ("fresh_reader_figures", "fresh reader current source/figure/verdict stale"),
        ("fresh_reader_wrong_owner", "independent fresh assigned reader"),
        ("fresh_reader_shared", "independent fresh assigned reader"),
        ("fresh_reader_original_owner", "original/basis reviewer mismatch"),
        ("fresh_reader_missing_note", "missing sequential checkpoint"),
        ("fresh_reader_bad_dispatch", "dispatch outside reader phase"),
        ("fresh_reader_unsealed", "not actually sealed/collected"),
        ("not_unavailable", "not explicitly unavailable"),
        ("old_fix_revise", "original fix was not verified"),
        ("prior_unclosed", "prior issue was not fixed"),
    ],
)
def test_old_reader_fix_rejects_incomplete_fresh_reader(tmp_path, variant, message):
    with pytest.raises(ValueError, match=message):
        check(tmp_path, fixture(tmp_path, variant, old_role="reader"))


def test_old_reader_fix_rejects_unclosed_checkpoint(tmp_path):
    with pytest.raises(KeyError, match="checkpoint_closure"):
        check(tmp_path, fixture(tmp_path, "fresh_reader_unclosed_checkpoint", old_role="reader"))


def test_old_continuity_fix_is_outside_fallback_scope(tmp_path):
    with pytest.raises(ValueError, match="original closure owner/role invalid"):
        check(tmp_path, fixture(tmp_path, old_role="continuity"))


def test_old_reader_tool_fix_is_outside_fallback_scope(tmp_path):
    with pytest.raises(ValueError, match="original closure owner/role invalid"):
        check(tmp_path, fixture(tmp_path, "tool_prior_reader", only_tool=True))


@pytest.mark.parametrize(
    "variant, message",
    [
        ("prior_bad_hash", "artifact changed"),
        ("unanchored_prior", "prior backup is not in active pointer history"),
        ("prior_gate_failed", "prior gate did not complete"),
        ("prior_gate_source", "prior gate source differs"),
        ("prior_unclosed", "prior issue was not fixed"),
        ("changed_disposition", "original closed disposition was changed"),
        ("old_fix_revise", "original fix was not verified"),
        ("old_figure", "original fix was not verified"),
        ("different_owner", "owner differ"),
        ("not_fresh", "independent fresh technical"),
        ("author_is_reviewer", "independent fresh technical"),
        ("wrong_page_assignment", "own original page review"),
        ("appendix_as_full", "appendix cannot replace full review"),
        ("not_main_proof", "current whole-page technical proof"),
        ("unverified_full_claim", "technical validation failed"),
        ("stale_appendix", "technical validation failed"),
        ("figure_mismatch", "current figure identity differs"),
        ("item_stale", "issue-specific current source/figure stale"),
        ("unknown_evidence", "unknown current evidence source"),
        ("duplicate_recheck", "duplicate historical replacement"),
        ("not_unavailable", "not explicitly unavailable"),
        ("nonempty_listing", "missing empty original-task-prefix listing"),
        ("wrong_collection", "collection outside actual phase"),
        ("false_receipt", "claimed appendix receipt differs"),
        ("full_predates_dispatch", "full review receipt outside actual phase/dispatch"),
        ("appendix_as_original", "appendix cannot be original full-page review"),
        ("not_sealed", "report not sealed"),
        ("tool_missing_issue", "tool audit omits resolved historical issue"),
        ("tool_stale_files", "issue-specific current files differ"),
        ("tool_appendix_as_full", "appendix cannot replace full tool audit"),
        ("tool_appendix_as_original", "appendix cannot be original full tool audit"),
        ("tool_incomplete_original", "original tool audit lacks complete file scope"),
        ("tool_missing_original", "missing original tool audit"),
        ("unused_tool_recheck", "unused historical replacement"),
        ("missing_page_group", "fix recheck is stale"),
        ("missing_issue_report", "issue inventory omits"),
        ("tamper_appendix", "artifact changed"),
    ],
)
def test_invalid_historical_rechecks_are_rejected(tmp_path, variant, message):
    with pytest.raises(ValueError, match=message):
        check(tmp_path, fixture(tmp_path, variant))


def test_unknown_prior_issue_cannot_be_created_by_appendix(tmp_path):
    with pytest.raises(IndexError):
        check(tmp_path, fixture(tmp_path, "unknown_prior_issue"))


def test_saved_prior_pointer_still_works_after_activation(tmp_path):
    built = fixture(tmp_path)
    backup = review.document(tmp_path, built[0]["historical_fix_rechecks"])["prior_active_pointer"]
    save(tmp_path, "docs/course-reviews-active.json", {"previous_pointer": backup})
    assert check(tmp_path, built) == {}


def test_issue_inventory_uses_same_recheck_rules(tmp_path):
    manifest, live, _, _ = fixture(tmp_path)
    result = inventory.inventory(tmp_path, manifest, live, "sealed", "technical")
    assert result["failing"] == [], result
    del manifest["historical_fix_rechecks"]
    result = inventory.inventory(tmp_path, manifest, live, "sealed", "technical")
    failures = result["dispositions"]["failing"]
    assert any("fix recheck is stale" in text for item in failures for text in item["errors"])
    assert any("fix does not cover affected pages/files" in text for item in failures for text in item["errors"])


def test_issue_inventory_rejects_invalid_sidecar_without_skipping_verification(tmp_path):
    manifest, live, _, _ = fixture(tmp_path, "different_owner")
    result = inventory.inventory(tmp_path, manifest, live, "sealed", "technical")
    assert result["failing"]
    assert any("owner differ" in error for error in result["errors"])
    assert any(
        "fix recheck is stale" in error for item in result["dispositions"]["failing"] for error in item["errors"]
    )
