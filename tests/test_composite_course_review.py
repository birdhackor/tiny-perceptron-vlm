"""Synthetic integrity fixtures only; these are never actual course review evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "synthetic_composite_review", Path(__file__).resolve().parents[1] / "docs/review-tools/composite_review.py"
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Two pages exercise unchanged baseline and repaired/exposed callback paths."""
    root = tmp_path / "synthetic-review-only"
    root.mkdir()
    monkeypatch.setattr(tool, "EXPECTED_CANONICAL", 2)
    artifacts = {}

    def put(name, value, *, raw=False):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw:
            path.write_bytes(value)
        else:
            write(path, value)
        ref = {"path": name, "sha256": tool.sha(path.read_bytes())}
        artifacts[name] = ref
        return ref

    criteria = {}
    for name in tool.CRITERIA:
        live = ".agents/skills/clear-tutorial/" + name
        ref = put(live, b"Synthetic criterion, not curriculum.\n", raw=True)
        criteria[name] = {**ref, "live_path": live, "live_sha256": ref["sha256"]}
    hashes = {name: ref["sha256"] for name, ref in criteria.items()}
    implementation = put("lib/synthetic.py", b"# Synthetic fixture code; no course execution.\n", raw=True)
    result_input = put("assets/synthetic-result.json", {"synthetic_fixture_not_execution": True})
    technical_inputs = put(
        "technical-inputs.json",
        {"files_sha256": {result_input["path"]: result_input["sha256"]}, "limits": ["Synthetic fixture only."]},
    )
    current = []
    for pid, body in [("1.1", b"Synthetic unchanged material.\n"), ("5.4", b"Synthetic repaired material.\n")]:
        source = put(f"course/{pid}.md", body, raw=True)
        nb = put(f"notebooks/{pid}.ipynb", {"synthetic_fixture_not_execution": True})
        current.append(
            {
                "page_id": pid,
                "source": source["path"],
                "selector": "whole-file",
                "source_sha256": source["sha256"],
                "figures_sha256": {},
                "notebook": nb["path"],
                "notebook_sha256": nb["sha256"],
            }
        )
    monkeypatch.setattr(tool, "current_inventory", lambda _: {"pages": current})
    tasks = {
        "ar": "/root/synthetic_a_reader",
        "at": "/root/synthetic_a_technical",
        "ac": "/root/synthetic_a_continuity",
        "br": "/root/synthetic_b_reader",
        "replacement": "/root/synthetic_replacement",
        "bt": "/root/synthetic_b_technical",
        "bc": "/root/synthetic_b_continuity",
    }
    batches = {}
    for stage in ("original", "freeze01", "callback02"):
        items = []
        for page in current if stage != "callback02" else current[1:]:
            p = {k: v for k, v in page.items() if k not in {"notebook", "notebook_sha256"}}
            body = (root / p["source"]).read_bytes()
            if p["page_id"] == "5.4" and stage != "callback02":
                body = ("Synthetic " + stage + " material.\n").encode()
            snapshot = put(f"{stage}/sources/{p['page_id']}.md", body, raw=True)
            p.update(source_sha256=snapshot["sha256"], snapshot=snapshot["path"])
            items.append(p)
        groups = {"all": {"pages": [p["page_id"] for p in items], "reviewer": tasks["ar"]}}
        if stage == "freeze01":
            groups = {
                "all": {"pages": ["5.4"], "reviewer": tasks["br"]},
                "callbacks": {"pages": ["5.4"], "reviewer": tasks["replacement"]},
            }
        value = {"criteria_sha256": hashes, "inventory": {"pages": items}, "groups": groups}
        if stage == "freeze01":
            value["implementation_sha256"] = {implementation["path"]: implementation["sha256"]}
        batches[stage] = (value, put(stage + "/manifest.json", value))
    pages_by_stage = {stage: tool.page_map(value[0]) for stage, value in batches.items()}

    def report(stage, role, task, ids, name=None):
        rows = []
        for pid in ids:
            rows.append(
                {
                    "page_id": pid,
                    "source_sha256": pages_by_stage[stage][pid]["source_sha256"],
                    "figures_sha256": {},
                    "understanding": "Synthetic statement to exercise integrity, not an actual reader answer.",
                    "four_questions": {"q1": "synthetic", "q2": "synthetic", "q3": "synthetic", "q4": "synthetic"},
                }
            )
        value = {"reviewer": task, "pages": rows, "new_necessary_findings": []}
        if stage == "callback02" and role == "technical":
            value.update(reader_gate="callback02/reader-gate.json", manifest_sha256=batches[stage][1]["sha256"])
        return put(name or f"{stage}/{role}.json", value)

    a_reports = {role: report("original", role, tasks["a" + role[0]], ["1.1", "5.4"]) for role in tool.ROLES}
    b_reports = {
        "reader": report("freeze01", "reader", tasks["replacement"], ["5.4"]),
        "technical": report("freeze01", "technical", tasks["bt"], ["5.4"]),
        "continuity": report("freeze01", "continuity", tasks["bc"], ["5.4"]),
    }
    c_reports = {
        "reader": report("callback02", "reader", tasks["replacement"], ["5.4"]),
        "technical": report("callback02", "technical", tasks["bt"], ["5.4"]),
        "continuity": report("callback02", "continuity", tasks["bc"], ["5.4"]),
    }
    c_data = json.loads((root / c_reports["continuity"]["path"]).read_bytes())
    c_data.update(started="2026-10-08T10:10:00+00:00", completed="2026-10-08T10:11:00+00:00")
    c_reports["continuity"] = put(c_reports["continuity"]["path"], c_data)
    technical_seal = put(
        "callback02/technical.seal.json",
        {"sha256": c_reports["technical"]["sha256"], "at": "2026-10-08T10:09:00+00:00"},
    )
    continuity_seal = put(
        "callback02/continuity.seal.json",
        {"sha256": c_reports["continuity"]["sha256"], "at": "2026-10-08T10:12:00+00:00"},
    )
    reader_gate = put("callback02/reader-gate.json", {"at": "2026-10-08T10:08:00+00:00"})
    cross = put("synthetic-cross.json", {"fixture": "group cross, not invented page answers"})

    def candidate(ref, pid, role, task, stage):
        report_data = json.loads((root / ref["path"]).read_bytes())
        index = next(i for i, p in enumerate(report_data["pages"]) if p["page_id"] == pid)
        return {
            "role": role,
            "reviewer": task,
            "recorded_source_sha256": pages_by_stage[stage][pid]["source_sha256"],
            "report": {**ref, "json_pointer": "/pages/" + str(index)},
        }

    traces, trace_pages = [], {}
    for stage in ("original", "freeze01"):
        for group, metadata in batches[stage][0]["groups"].items():
            rows = []
            for pid in metadata["pages"]:
                page = pages_by_stage[stage][pid]
                note = {key: "Synthetic fixture only." for key in tool.NOTE_FIELDS}
                note["reviewer"] = metadata["reviewer"]
                rows.append(
                    {
                        "kind": "checkpoint",
                        "phase": "main",
                        "page_id": pid,
                        "unit_index": 0,
                        "source_sha256": page["source_sha256"],
                        "unit_sha256": page["source_sha256"],
                        "note": note,
                        "at": "2026-10-08T10:00:00+00:00",
                    }
                )
                rows.append({"kind": "page_end", "phase": "main", "page_id": pid, "at": "2026-10-08T10:00:00+00:00"})
            ref = put(f"{stage}/{group}.jsonl", b"".join((json.dumps(row) + "\n").encode() for row in rows), raw=True)
            traces.append({"stage": stage, "group": group, "ref": ref})
            for pid in metadata["pages"]:
                trace_pages[(stage, group, pid)] = {
                    "trace": ref,
                    "raw_checkpoint_lines": [
                        next(i + 1 for i, r in enumerate(rows) if r.get("page_id") == pid and r["kind"] == "checkpoint")
                    ],
                    "early_full_page_exposure": False,
                }
    pages = []
    for page in current:
        pid = page["page_id"]
        original = {
            role: candidate(a_reports[role], pid, role, tasks["a" + role[0]], "original") for role in tool.ROLES
        }
        initial = original["reader"]
        selected, mode, sequential = original, "original", trace_pages[("original", "all", pid)]
        bounded = {}
        if pid == "5.4":
            bounded = {
                role: candidate(
                    b_reports[role],
                    pid,
                    role,
                    tasks["replacement"] if role == "reader" else tasks["b" + role[0]],
                    "freeze01",
                )
                for role in tool.ROLES
            }
            bounded["unprompted_replacement_reader_main"] = bounded["reader"]
            bounded["cross"] = bounded["continuity"]
            initial = bounded["reader"]
            sequential = trace_pages[("freeze01", "callbacks", pid)]
            selected = {
                role: candidate(
                    c_reports[role],
                    pid,
                    role,
                    tasks["replacement"] if role == "reader" else tasks["b" + role[0]],
                    "callback02",
                )
                for role in tool.ROLES
            }
            mode = "callback02"
        pages.append(
            {
                "page_id": pid,
                "current_source": page,
                "selected_mode": mode,
                "selected_roles": selected,
                "reader_source_evidence": initial,
                "sequential_evidence": sequential,
                "page_ending_evidence": initial,
                "original_role_candidates": original,
                "original_cross": cross,
                "bounded_role_candidates": bounded,
            }
        )
    freeze_pages = []
    notebooks = {}
    source_files = {}
    for page in current:
        nb = artifacts[page["notebook"]]
        notebooks[nb["path"]] = {**nb, "live_path": nb["path"]}
        source_files[page["source"]] = {**artifacts[page["source"]], "live_path": page["source"]}
        freeze_pages.append({**page, "snapshot": page["source"], "notebook": notebooks[nb["path"]]})
    freeze = put(
        "current-freeze.json",
        {
            "pages": freeze_pages,
            "source_files": source_files,
            "figures": {},
            "criteria": criteria,
            "notebooks": notebooks,
        },
    )
    history_note = put("history-note.json", {"synthetic": True})
    archive = put(
        "archive-index.json",
        {"schema_version": 1, "artifacts": [{"original_path": "/synthetic/history-note.json", **history_note}]},
    )
    adjudication = put(
        "adjudication.json",
        {
            "original_relation_closures": [{"id": "fixture-necessary", "original_grade": "necessary"}],
            "new_bounded_review_dispositions": [],
        },
    )
    diagnosis = put("diagnosis.json", {"decisions": [{"id": "fixture-necessary", "root_grade": "necessary"}]})
    closure = {
        "relation_id": "fixture-necessary",
        "coordinator_record": {**adjudication, "json_pointer": "/original_relation_closures/0"},
        "pages": ["5.4"],
    }
    entries = [
        {"stage": stage, "name": name, "expected_sha256": hashes[name], "current": ref, "frozen": ref}
        for stage in batches
        for name, ref in criteria.items()
    ]
    catalog = {
        "review_policy": tool.POLICY,
        "pages": pages,
        "manifest_references": {k: v[1] for k, v in batches.items()},
        "criteria": entries,
        "trace_validation": traces,
        "artifact_registry": {history_note["path"]: {"sha256": history_note["sha256"]}},
        "extras": [],
        "necessary_closures": [closure],
        "limitations": ["Synthetic fixture; no AI/human review performed."],
    }
    catalog["finding_sources"] = {
        "adjudication": adjudication,
        "original_diagnosis": diagnosis,
        "bounded_diagnosis": {**adjudication, "json_pointer": "/new_bounded_review_dispositions"},
    }
    catalog["technical_inputs_manifest"] = technical_inputs
    assignments = [
        {
            "reviewer_task": task,
            "role": "reader"
            if key in {"ar", "br", "replacement"}
            else "technical"
            if key.endswith("t")
            else "continuity",
            "groups": ["synthetic"],
            "evidence_class": "coordinator_attested_history",
            "basis": "Synthetic testing declaration only.",
            "limits": ["Not actual orchestration."],
        }
        for key, task in tasks.items()
    ]
    attestation = {
        "kind": "coordinator_attested_history",
        "coordinator": "/root",
        "recorded_at": "2026-10-08T11:00:00+00:00",
        "reviewer_assignments": assignments,
        "finding_dispositions": [
            {
                "id": "fixture-necessary",
                "necessary": True,
                "status": "resolved",
                "pages": ["5.4"],
                "reason": "Synthetic closure fixture only.",
                "current_evidence": [pages[1]["selected_roles"]["technical"]],
            }
        ],
        "limitations": ["Synthetic fixture, not actual evidence."],
    }
    order = []
    for stage, offset in [("original", 1), ("freeze01", 5)]:
        times = {
            role + "_completed_at": f"2026-10-08T10:0{offset + i}:00+00:00"
            for i, role in enumerate((*tool.ROLES, "cross"))
        }
        evidence = [
            put(f"{stage}/{key}-synthetic-gate.json", {"at": value, "synthetic_fixture_only": True})
            for key, value in times.items()
        ]
        order.append(
            {
                "batch": stage,
                **times,
                "independence_before_cross": True,
                "chronology_basis": "Synthetic testing data only.",
                "limitations": ["Not actual review chronology."],
                "evidence": evidence,
            }
        )
    manifest = {
        "schema_version": 1,
        "review_policy": tool.POLICY,
        "catalog": put("catalog.json", catalog),
        "archive_index": archive,
        "coordinator_attestation": put("attestation.json", attestation),
        "source_freeze": freeze,
        "criteria": criteria,
        "author_tasks": ["/root"],
        "stage_order": order,
        "latest_continuity": {
            "report": c_reports["continuity"],
            "seal": continuity_seal,
            "technical_report": c_reports["technical"],
            "technical_seal": technical_seal,
            "reader_gate": reader_gate,
            "started_at_pointer": "/started",
            "completed_at_pointer": "/completed",
        },
    }
    manifest_path = root / "manifest.json"
    manifest["current_technical_inputs"] = technical_inputs
    write(manifest_path, manifest)

    def save():
        manifest["catalog"] = put("catalog.json", catalog)
        manifest["coordinator_attestation"] = put("attestation.json", attestation)
        write(manifest_path, manifest)

    return root, manifest_path, manifest, catalog, attestation, save, put


def test_valid_fixture_is_metadata_only(fixture):
    root, path, *_ = fixture
    result = tool.audit_manifest(root, path)
    assert result["failures"] == []
    assert result["status"] == "passed_metadata_checks"
    assert "does not prove comprehension" in result["scope_note"]
    assert len(result["records"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "criteria",
        "source",
        "artifact",
        "missing_page",
        "missing_role",
        "invented_identity",
        "same_role_owner",
        "author",
        "unclosed",
        "missing_trace",
        "bad_order",
        "early_callback",
        "removed_closure",
    ],
)
def test_rejects_broken_review_evidence(fixture, mutation):
    root, path, manifest, catalog, attestation, save, put = fixture
    if mutation == "criteria":
        (root / manifest["criteria"]["SKILL.md"]["path"]).write_text("stale criterion\n")
    elif mutation == "source":
        (root / "course/1.1.md").write_text("tampered source\n")
    elif mutation == "artifact":
        (root / "history-note.json").write_text("tampered artifact\n")
    elif mutation == "missing_page":
        catalog["pages"].pop()
    elif mutation == "missing_role":
        catalog["pages"][0]["selected_roles"].pop("technical")
    elif mutation == "invented_identity":
        for p in catalog["pages"]:
            p["selected_roles"]["reader"]["reviewer"] = "/root/invented_many_pages"
    elif mutation == "same_role_owner":
        catalog["pages"][0]["selected_roles"]["technical"]["reviewer"] = catalog["pages"][0]["selected_roles"][
            "reader"
        ]["reviewer"]
    elif mutation == "author":
        manifest["author_tasks"].append(catalog["pages"][0]["selected_roles"]["reader"]["reviewer"])
    elif mutation == "unclosed":
        attestation["finding_dispositions"][0]["status"] = "deferred_optional"
    elif mutation == "missing_trace":
        catalog["trace_validation"].pop()
    elif mutation == "bad_order":
        manifest["stage_order"][0]["technical_completed_at"] = "2026-10-08T09:00:00+00:00"
    elif mutation == "early_callback":
        ref = manifest["latest_continuity"]["report"]
        report = json.loads((root / ref["path"]).read_bytes())
        report["started"] = "2026-10-08T10:08:00+00:00"
        manifest["latest_continuity"]["report"] = put(ref["path"], report)
    elif mutation == "removed_closure":
        catalog["necessary_closures"] = []
    save()
    assert tool.audit_manifest(root, path)["failures"], mutation


def test_active_router_rejects_unknown_policy_without_legacy_fallback(fixture):
    root, _, *_ = fixture
    spec = importlib.util.spec_from_file_location(
        "synthetic_active_router", tool.ROOT / "docs/review-tools/phase7_review.py"
    )
    phase7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase7)
    write(root / "docs/course-reviews-active.json", {"schema_version": 1, "review_policy": "invented_other_policy"})
    assert "legacy fallback forbidden" in phase7.check_active(root)["failures"][0]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("explicit_figure_version", "figure versions stale"),
        ("unsealed_technical", "technical not sealed"),
        ("unrelated_ending", "page-ending record differs"),
        ("unrelated_reader_gate", "different reader gate"),
        ("missing_source_registry", "omits canonical source file"),
        ("missing_finding_source", "finding sources missing"),
        ("omitted_adjudication_document", "diagnosis relations omitted"),
        ("empty_closure_pages", "no required current pages"),
        ("unrelated_closure_page", "scope differs"),
        ("unbound_order_time", "not bound to actual"),
        ("early_history", "attestation predates"),
        ("retained_necessary_callback", "retains necessary"),
        ("missing_callback_extra", "callback missing page"),
        ("notebook_tamper", "tampered"),
        ("live_code_drift", "implementation drift"),
        ("live_result_drift", "technical input drift"),
        ("author_owned_closure", "unselected role evidence"),
    ],
)
def test_rejects_rebound_evidence_with_valid_artifact_hashes(fixture, mutation, expected):
    root, path, manifest, catalog, attestation, save, put = fixture
    if mutation == "explicit_figure_version":
        candidate = catalog["pages"][0]["selected_roles"]["reader"]
        ref = candidate["report"]
        report = json.loads((root / ref["path"]).read_bytes())
        report["pages"][0]["figures_sha256"] = {"course/figures/unrelated.svg": "0" * 64}
        candidate["report"] = {**put("rebound-reader.json", report), "json_pointer": ref["json_pointer"]}
    elif mutation == "unsealed_technical":
        candidate = catalog["pages"][1]["selected_roles"]["technical"]
        report = json.loads((root / candidate["report"]["path"]).read_bytes())
        report["synthetic_unsealed_copy"] = True
        candidate["report"] = {**put("unsealed-technical.json", report), "json_pointer": "/pages/0"}
    elif mutation == "unrelated_ending":
        catalog["pages"][0]["page_ending_evidence"] = {"report": put("unrelated-ending.json", {"unrelated": True})}
    elif mutation == "unrelated_reader_gate":
        manifest["latest_continuity"]["reader_gate"] = put(
            "unrelated-reader-gate.json", {"at": "2026-10-08T10:08:00+00:00"}
        )
    elif mutation == "missing_source_registry":
        freeze = json.loads((root / manifest["source_freeze"]["path"]).read_bytes())
        freeze["source_files"] = {}
        manifest["source_freeze"] = put("incomplete-source-freeze.json", freeze)
    elif mutation == "missing_finding_source":
        catalog["finding_sources"].pop("original_diagnosis")
    elif mutation == "omitted_adjudication_document":
        incomplete = put(
            "incomplete-adjudication.json", {"original_relation_closures": [], "new_bounded_review_dispositions": []}
        )
        catalog["finding_sources"]["adjudication"] = incomplete
        catalog["finding_sources"]["bounded_diagnosis"] = {
            **incomplete,
            "json_pointer": "/new_bounded_review_dispositions",
        }
    elif mutation == "empty_closure_pages":
        catalog["necessary_closures"][0]["pages"] = []
    elif mutation == "unrelated_closure_page":
        catalog["necessary_closures"][0]["pages"] = ["1.1"]
    elif mutation == "unbound_order_time":
        manifest["stage_order"][0]["reader_completed_at"] = "2026-10-08T10:01:01+00:00"
    elif mutation == "early_history":
        attestation["recorded_at"] = "2026-10-08T10:05:00+00:00"
    elif mutation == "retained_necessary_callback":
        ref = manifest["latest_continuity"]["report"]
        report = json.loads((root / ref["path"]).read_bytes())
        report["unresolved_necessary_findings"] = ["synthetic necessary finding"]
        new = put("callback-with-unresolved.json", report)
        manifest["latest_continuity"]["report"] = new
        manifest["latest_continuity"]["seal"] = put(
            "unresolved.seal.json", {"sha256": new["sha256"], "at": "2026-10-08T10:12:00+00:00"}
        )
    elif mutation == "missing_callback_extra":
        ref = catalog["manifest_references"]["callback02"]
        batch = json.loads((root / ref["path"]).read_bytes())
        batch["inventory"]["pages"].append({**batch["inventory"]["pages"][0], "page_id": "synthetic-extra"})
        new = put("callback-with-extra.json", batch)
        catalog["manifest_references"]["callback02"] = new
        ref = manifest["latest_continuity"]["technical_report"]
        report = json.loads((root / ref["path"]).read_bytes())
        report["manifest_sha256"] = new["sha256"]
        new = put("technical-with-extra.json", report)
        manifest["latest_continuity"]["technical_report"] = new
        manifest["latest_continuity"]["technical_seal"] = put(
            "technical-with-extra.seal.json", {"sha256": new["sha256"], "at": "2026-10-08T10:09:00+00:00"}
        )
    elif mutation == "notebook_tamper":
        (root / "notebooks/1.1.ipynb").write_text("changed notebook bytes")
    elif mutation == "live_code_drift":
        (root / "lib/synthetic.py").write_text("# Changed live code\n")
    elif mutation == "live_result_drift":
        (root / "assets/synthetic-result.json").write_text('{"changed": true}\n')
    elif mutation == "author_owned_closure":
        candidate = copy.deepcopy(attestation["finding_dispositions"][0]["current_evidence"][0])
        report = json.loads((root / candidate["report"]["path"]).read_bytes())
        report["reviewer"] = "/root"
        candidate["reviewer"] = "/root"
        candidate["report"] = {**put("author-closure.json", report), "json_pointer": "/pages/0"}
        attestation["finding_dispositions"][0]["current_evidence"] = [candidate]
    save()
    result = tool.audit_manifest(root, path)
    assert result["failures"] and expected in result["failures"][0], result


def test_json_pointer_requires_actual_page_not_group_summary(fixture):
    root, path, _, catalog, _, save, _ = fixture
    catalog["pages"][0]["selected_roles"]["reader"]["report"]["json_pointer"] = ""
    save()
    assert tool.audit_manifest(root, path)["failures"]


def test_paper_receipt_cannot_replace_trace(fixture):
    root, _, _, _, _, _, put = fixture
    source_sha = tool.sha(b"An absent synthetic published-paper cache.")
    receipt = put(
        "paper-receipt.json",
        {
            "kind": "external_paper_receipt",
            "source_sha256": source_sha,
            "url": "https://arxiv.org/pdf/2309.00071v3",
            "version": "v3",
            "locators": ["Synthetic test locator"],
            "verified_at": "2026-10-08T10:00:00+00:00",
            "limitations": ["Receipt fixture only."],
        },
    )
    entry = {
        "original_path": "/history/notes/reader.txt",
        **receipt,
        "kind": "external_paper_receipt",
        "source_sha256": source_sha,
    }
    index = put("invalid-paper-index.json", {"schema_version": 1, "artifacts": [entry]})
    with pytest.raises(ValueError, match="cannot replace"):
        tool.Artifacts(root, index)


def test_fixed_paper_receipt_checks_both_receipt_and_original_hash(fixture):
    root, _, _, _, _, _, put = fixture
    source_sha = tool.sha(b"Absent synthetic external paper; never course source.")
    receipt = put(
        "paper-receipt.json",
        {
            "kind": "external_paper_receipt",
            "source_sha256": source_sha,
            "url": "https://arxiv.org/pdf/2309.00071v3",
            "version": "v3",
            "locators": ["Synthetic test locator"],
            "verified_at": "2026-10-08T10:00:00+00:00",
            "limitations": ["No CI download or actual paper reading."],
        },
    )
    entry = {
        "original_path": "/history/original-papers/2309.00071v3.pdf",
        **receipt,
        "kind": "external_paper_receipt",
        "source_sha256": source_sha,
    }
    index = put("paper-index.json", {"schema_version": 1, "artifacts": [entry]})
    store = tool.Artifacts(root, index)
    store.raw({"path": entry["original_path"], "sha256": source_sha}, allow_external=True)
    with pytest.raises(ValueError, match="original hash changed"):
        store.raw({"path": entry["original_path"], "sha256": "0" * 64}, allow_external=True)
    bad = copy.deepcopy(entry)
    bad["sha256"] = "0" * 64
    bad_index = put("bad-paper-index.json", {"schema_version": 1, "artifacts": [bad]})
    with pytest.raises(ValueError, match="tampered"):
        tool.Artifacts(root, bad_index)


@pytest.mark.parametrize("reference_kind", ["original_alias", "receipt_path"])
def test_paper_receipt_with_report_fields_cannot_become_review_evidence(fixture, reference_kind):
    root, path, manifest, catalog, _, save, put = fixture
    candidate = catalog["pages"][0]["selected_roles"]["reader"]
    report = json.loads((root / candidate["report"]["path"]).read_bytes())
    source_sha = tool.sha(b"Synthetic absent external paper bytes")
    receipt = put(
        "structured-paper-receipt.json",
        {
            **report,
            "kind": "external_paper_receipt",
            "source_sha256": source_sha,
            "url": "https://arxiv.org/pdf/2309.00071v3",
            "version": "v3",
            "locators": ["Synthetic fixture only"],
            "verified_at": "2026-10-08T10:00:00+00:00",
            "limitations": ["No original paper included."],
        },
    )
    entry = {
        "original_path": "/history/original-papers/2309.00071v3.pdf",
        **receipt,
        "kind": "external_paper_receipt",
        "source_sha256": source_sha,
    }
    index = json.loads((root / manifest["archive_index"]["path"]).read_bytes())
    index["artifacts"].append(entry)
    manifest["archive_index"] = put("archive-with-paper.json", index)
    reference = (
        {"path": entry["original_path"], "sha256": source_sha} if reference_kind == "original_alias" else receipt
    )
    candidate["report"] = {**reference, "json_pointer": "/pages/0"}
    save()
    assert "paper receipt cannot substitute" in tool.audit_manifest(root, path)["failures"][0]


def test_callback_extra_requires_selected_ordered_continuity(fixture):
    root, path, manifest, catalog, _, save, put = fixture
    initial = catalog["pages"][1]
    extra_id = "synthetic-extra"
    extra = {
        "stage": "callback02",
        "page_id": extra_id,
        "source_matches_current_file": True,
        "source": {**initial["current_source"], "page_id": extra_id},
        "selected_roles": {},
    }
    # Extend the synthetic callback scope truthfully; no fixture is actual evidence.
    ref = catalog["manifest_references"]["callback02"]
    batch = json.loads((root / ref["path"]).read_bytes())
    batch["inventory"]["pages"].append({**batch["inventory"]["pages"][0], "page_id": extra_id})
    batch_ref = put("extended-callback.json", batch)
    catalog["manifest_references"]["callback02"] = batch_ref
    for role in tool.ROLES:
        candidate = initial["selected_roles"][role]
        report = json.loads((root / candidate["report"]["path"]).read_bytes())
        report["pages"].append({**report["pages"][0], "page_id": extra_id})
        if role == "technical":
            report["manifest_sha256"] = batch_ref["sha256"]
        new_ref = put(f"extended-{role}.json", report)
        candidate["report"] = {**new_ref, "json_pointer": "/pages/0"}
        extra["selected_roles"][role] = {**candidate, "report": {**new_ref, "json_pointer": "/pages/1"}}
        if role in {"technical", "continuity"}:
            key = "technical_report" if role == "technical" else "report"
            seal_key = "technical_seal" if role == "technical" else "seal"
            manifest["latest_continuity"][key] = new_ref
            seal = json.loads((root / manifest["latest_continuity"][seal_key]["path"]).read_bytes())
            seal["sha256"] = new_ref["sha256"]
            manifest["latest_continuity"][seal_key] = put(f"extended-{role}.seal.json", seal)
    catalog["extras"] = [extra]
    save()
    assert tool.audit_manifest(root, path)["failures"] == []
    old = json.loads((root / extra["selected_roles"]["continuity"]["report"]["path"]).read_bytes())
    old["prior_parallel_callback_fixture"] = True
    extra["selected_roles"]["continuity"]["report"] = {
        **put("earlier-extra-continuity.json", old),
        "json_pointer": "/pages/1",
    }
    save()
    assert "extra not actual sealed callback" in tool.audit_manifest(root, path)["failures"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "extra_exclusion",
        "removed_old",
        "reordered_old",
        "other_toml",
        "baseline_rebound",
        "caller_allowed_path",
        "unbound_current",
    ],
)
def test_only_exact_preserved_ruff_archive_delta_is_allowed(fixture, mutation):
    root, path, manifest, catalog, _, save, put = fixture
    baseline_raw = b'[project]\nname = "synthetic-fixture"\n[tool.ruff]\nextend-exclude = ["keep-first", "keep-last"]\n'
    baseline = put("freeze-01/freeze/implementation/pyproject.toml", baseline_raw, raw=True)
    exclusions = ["keep-first", *tool.RUFF_ARCHIVE_EXCLUDES, "keep-last"]
    if mutation == "extra_exclusion":
        exclusions.append("lib/**")
    elif mutation == "removed_old":
        exclusions.remove("keep-last")
    elif mutation == "reordered_old":
        exclusions = ["keep-last", *tool.RUFF_ARCHIVE_EXCLUDES, "keep-first"]
    project_name = "changed-dependency-config" if mutation == "other_toml" else "synthetic-fixture"
    current_raw = (
        f'[project]\nname = "{project_name}"\n[tool.ruff]\nextend-exclude = ' + json.dumps(exclusions) + "\n"
    ).encode()
    put("pyproject.toml", current_raw, raw=True)
    ref = catalog["manifest_references"]["freeze01"]
    batch = json.loads((root / ref["path"]).read_bytes())
    batch["implementation_sha256"]["pyproject.toml"] = baseline["sha256"]
    catalog["manifest_references"]["freeze01"] = put("freeze-01/manifest-with-config.json", batch)
    manifest["configuration_delta"] = {
        "path": "pyproject.toml",
        "baseline_snapshot": baseline,
        "current_sha256": tool.sha(current_raw),
        "allowed_added_ruff_archive_paths": list(tool.RUFF_ARCHIVE_EXCLUDES),
        "purpose": "Synthetic fixture of exact archival config delta only.",
    }
    if mutation == "baseline_rebound":
        manifest["configuration_delta"]["baseline_snapshot"] = {**baseline, "sha256": "0" * 64}
    elif mutation == "caller_allowed_path":
        manifest["configuration_delta"]["allowed_added_ruff_archive_paths"].append("lib/**")
    elif mutation == "unbound_current":
        manifest["configuration_delta"]["current_sha256"] = "0" * 64
    save()
    result = tool.audit_manifest(root, path)
    assert bool(result["failures"]) is (mutation is not None), result
    if mutation:
        assert "configuration" in result["failures"][0], result
