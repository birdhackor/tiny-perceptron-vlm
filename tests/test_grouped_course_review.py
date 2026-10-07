"""Synthetic integrity fixtures only: never actual agents, reader answers or formal course-pass reports."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "phase7_fixture_tool", Path(__file__).resolve().parents[1] / "docs/review-tools/phase7_review.py"
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-repo"
    chapter = root / "course/chapters/01.md"
    chapter.parent.mkdir(parents=True)
    chapter.write_text(
        "# Fixture chapter\n\nFixture introduction.\n\n"
        "## 1.1 Fixture one\n\n" + "Fixture material. " * 50 + "\n\n"
        "UNREAD_FUTURE_FIXTURE\n\n## 1.2 Fixture two\n\nSecond fixture material.\n",
        encoding="utf-8",
    )
    index = []
    for pid in ("1.1", "1.2"):
        notebook = f"notebooks/01/{pid}.ipynb"
        put(root / notebook, {"fixture": True})
        index.append({"id": pid, "title": "fixture", "source": "course/chapters/01.md", "notebook": notebook})
    put(root / "course/lesson-index.json", index)
    for name in tool.CRITERIA:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Synthetic criterion fixture only.\n", encoding="utf-8")
    monkeypatch.setattr(tool.export_course, "DOCUMENTS", {})
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *args, **kwargs: "synthetic-fixture-revision\n")
    plan = root / "fixture-groups.json"
    put(
        plan,
        {"groups": [{"group": "a", "primary_page_ids": ["index", "chapter-01", "1.1", "1.2"], "context_page_ids": []}]},
    )
    folder = root / "docs/fixture-review/freeze-01"
    tool.freeze(root, folder, plan, expected_pages=4, batch_id="fixture-only")
    return root, folder / "manifest.json", plan


def declare(root, manifest, stage, *, task=None, ready=None):
    task = task or "/root/fixture_" + stage
    path = manifest.parent / "dispatch.json"
    data = (
        tool.read_json(path)
        if path.exists()
        else {"schema_version": 1, "batch_id": "fixture-only", "dispatches": [], "followups": []}
    )
    row = {
        "stage": stage,
        "group": "a",
        "reviewer_task": task,
        "fork_turns": "none",
        "dispatched_at": tool.now(),
        "spawn_receipt": {
            "tool": "collaboration.spawn_agent",
            "task_name": task,
            "agent_id": "synthetic-test-not-a-real-agent",
        },
    }
    if ready:
        row["ready_receipt"] = ready
    data["dispatches"].append(row)
    put(path, data)
    return task


def test_refresh_keeps_same_batch_owners_and_excludes_real_legacy_reviewers(setup):
    root, manifest, plan = setup
    owner = declare(root, manifest, "reader")
    put(
        root / "docs/reader-reviews/current.json",
        {
            "review_policy": tool.POLICY,
            "batch_id": "fixture-only",
            "reviewer_task": owner,
        },
    )
    put(
        root / "docs/reader-reviews/legacy.json",
        {
            "review_policy": tool.POLICY,
            "batch_id": "another-batch",
            "reviewer_task": "/root/legacy_fixture",
        },
    )
    refreshed = root / "docs/fixture-review/freeze-02"
    tool.freeze(root, refreshed, plan, expected_pages=4, batch_id="fixture-only", previous=manifest)
    data = tool.read_json(refreshed / "manifest.json")
    assert owner not in data["excluded_reviewers"]
    assert "/root/legacy_fixture" in data["excluded_reviewers"]
    tool.dispatch(root, refreshed / "manifest.json", data, "reader", "a", owner)


def test_refresh_additively_corrects_old_self_report_exclusion(setup):
    root, manifest, plan = setup
    owner = declare(root, manifest, "reader")
    explicit = "/root/explicit_fixture"
    # Synthetic old-bug fixture only; never mutate a real review manifest.
    second = root / "docs/fixture-review/freeze-02"
    tool.freeze(root, second, plan, expected_pages=4, batch_id="fixture-only", previous=manifest, excluded=(explicit,))
    bad = tool.read_json(second / "manifest.json")
    bad["excluded_reviewers"].append(owner)
    put(second / "manifest.json", bad)
    original_bad = (second / "manifest.json").read_bytes()
    for task, name in [(owner, "current"), (explicit, "explicit")]:
        put(
            root / f"docs/reader-reviews/{name}.json",
            {
                "review_policy": tool.POLICY,
                "batch_id": "fixture-only",
                "reviewer_task": task,
            },
        )
    third = root / "docs/fixture-review/freeze-03"
    tool.freeze(root, third, plan, expected_pages=4, batch_id="fixture-only", previous=second / "manifest.json")
    fixed = tool.read_json(third / "manifest.json")
    assert owner not in fixed["excluded_reviewers"]
    assert explicit in fixed["excluded_reviewers"]
    assert set(tool.EXCLUDED) <= set(fixed["excluded_reviewers"])
    assert fixed["exclusion_correction"]["reviewer_tasks"] == [owner]
    assert (second / "manifest.json").read_bytes() == original_bad


def test_same_batch_report_does_not_override_author_exclusion(setup):
    root, manifest, plan = setup
    owner = declare(root, manifest, "reader")
    put(
        root / "docs/reader-reviews/current.json",
        {
            "review_policy": tool.POLICY,
            "batch_id": "fixture-only",
            "reviewer_task": owner,
        },
    )
    second = root / "docs/fixture-review/freeze-02"
    tool.freeze(root, second, plan, expected_pages=4, batch_id="fixture-only", previous=manifest, authors=(owner,))
    with pytest.raises(ValueError, match="author/researcher/legacy reviewer"):
        tool.dispatch(root, second / "manifest.json", tool.read_json(second / "manifest.json"), "reader", "a", owner)


def followup(manifest, stage, task):
    path = manifest.parent / "dispatch.json"
    data = tool.read_json(path)
    data["followups"].append(
        {
            "stage": stage,
            "group": "a",
            "reviewer_task": task,
            "called_at": tool.now(),
            "followup_receipt": {
                "tool": "collaboration.followup_task",
                "task_name": task,
                "receipt": "synthetic fixture, not actual dispatch",
            },
        }
    )
    put(path, data)


def note(packet, *, method=False, issue=None, rechecks=()):
    quote = next(line for line in packet["text"].splitlines() if line.strip())[:30]
    result = {
        "page_id": packet["page_id"],
        "unit_index": packet["unit_index"],
        **{field: "Synthetic integrity fixture; not a student's understanding." for field in tool.reader.FIELDS},
    }
    if packet["four_questions_required"] or method:
        at = (
            ["first_use", "section_end"]
            if method and packet["section_end"]
            else "first_use"
            if method
            else "section_end"
        )
        claim = {"claim": "synthetic claim", "status": "explicit", "quote": quote}
        result["four_questions"] = [
            {
                "at": at,
                "scope": packet["required_section_ends"] or packet["page_id"],
                "subject": "fixture method" if method else "fixture subject",
                "questions": [
                    {"number": n, "answer": "synthetic fixture answer", "claims": [copy.deepcopy(claim)]}
                    for n in range(1, 5)
                ],
            }
        ]
        if method:
            result["introduced_methods"] = ["fixture method"]
            point = {"assessment": "sufficient", "reason": "fixture assessment", "evidence": [copy.deepcopy(claim)]}
            result["method_introductions"] = [
                {
                    "name": "fixture method",
                    **{key: copy.deepcopy(point) for key in ("identity", "purpose", "step_relation")},
                }
            ]
    if issue:
        result["issues"] = [
            {
                "id": issue,
                "type": "concept",
                "severity": "burden",
                "location": "fixture location",
                "quote": quote,
                "understanding": "synthetic uncertainty",
                "missing": "synthetic connection",
            }
        ]
    if rechecks:
        result["rechecks"] = list(rechecks)
    return result


def read_fixture(root, manifest, stage, task, **options):
    packet = tool.start_session(root, manifest, stage, "a", task, **options)
    session, trace = packet["session"], packet["trace_file"]
    while not packet.get("complete"):
        packet = tool.next_unit(root, session, note(packet, rechecks=options.get("issues", ())))
    return trace


def report_fixture(root, manifest, stage, task, trace, *, old=None, suffix="initial"):
    frozen = tool.read_json(manifest)
    report = (
        copy.deepcopy(old)
        if old
        else {
            "schema_version": 1,
            "review_policy": tool.POLICY,
            "batch_id": "fixture-only",
            "stage": stage,
            "group": "a",
            "reviewer_task": task,
            "reviewer_context": "fresh",
            "verdict": "pass",
            "trace_files": [],
            "pages": [],
        }
    )
    report["manifest_sha256"] = tool.sha(manifest.read_bytes())
    report["trace_files"].append({"path": trace, "sha256": tool.sha((root / trace).read_bytes())})
    rows = tool.raw_trace(root, trace)
    for source in frozen["pages"]:
        pid = source["page_id"]
        notes = [row for row in rows[1:-1] if row["page_id"] == pid]
        if not notes:
            continue
        page = {
            "page_id": pid,
            "source_sha256": source["source_sha256"],
            "figures_sha256": source["figures_sha256"],
            "verdict": "pass",
            "summary": "Synthetic fixture, never a real content review",
            "trace_file": trace,
            "question_refs": [row["event_index"] for row in notes if row["four_questions"]],
            "issues": [],
            "missing_visuals": "No fixture image",
            "visual_checks": [],
            "page_visual_check": {
                "status": "unverified",
                "required": False,
                "details": "Fixture has no needed image/layout claim",
            },
        }
        if stage == "reader":
            page["variation"] = {"not_applicable": True, "reason": "synthetic navigation fixture"}
        elif stage == "technical":
            page.update(
                {
                    "artifacts": [],
                    "sources": [],
                    "claims": [],
                    "applicability": {"substantive_claims": False, "reason": "synthetic metadata fixture only"},
                    "checks": {
                        key: {"status": "not_applicable", "details": "fixture applicability", "claim_ids": []}
                        for key in tool.technical.CHECKS
                    },
                }
            )
        else:
            page["continuity_checks"] = {
                key: "Synthetic continuity metadata fixture"
                for key in ("prior_relation", "transition", "new_prerequisites", "terminology", "example_change")
            }
        previous = next((p for p in report["pages"] if p["page_id"] == pid), None)
        if previous:
            report["pages"][report["pages"].index(previous)] = page
        else:
            report["pages"].append(page)
    order = frozen["groups"][0]["primary_page_ids"]
    report["pages"].sort(key=lambda page: order.index(page["page_id"]))
    report_path = manifest.parent / "fixture-reports" / f"{stage}-{suffix}.json"
    put(report_path, report)
    receipt = manifest.parent / "fixture-reports" / f"{stage}-{suffix}-receipt.json"
    put(
        receipt,
        {
            "final_received": True,
            "reviewer_task": task,
            "received_at": tool.now(),
            "receipt": "Synthetic test receipt; no actual agent",
        },
    )
    tool.collect(root, manifest, report_path, receipt)
    return report_path, report


def complete(root, manifest, stage, ready=None):
    task = declare(root, manifest, stage, ready=ready)
    trace = read_fixture(root, manifest, stage, task)
    path, report = report_fixture(root, manifest, stage, task, trace)
    return task, trace, path, report


def active(root, manifest):
    put(
        root / tool.ACTIVE,
        {
            "schema_version": 1,
            "review_policy": tool.POLICY,
            "manifest": tool.relative(root, manifest),
            "manifest_sha256": tool.sha(manifest.read_bytes()),
        },
    )


def test_no_active_preserves_legacy_but_invalid_active_never_falls_back(setup):
    root, manifest, _ = setup
    assert tool.check_active(root) is None
    put(root / tool.ACTIVE, {"schema_version": 1, "review_policy": "wrong-policy"})
    assert tool.check_active(root)["failures"]
    with pytest.raises(SystemExit):
        tool.route_active(root)


def test_active_uncollected_or_stale_is_not_a_pass(setup):
    root, manifest, _ = setup
    active(root, manifest)
    assert tool.check_active(root, "reader")["failures"]
    declare(root, manifest, "reader")
    data = tool.read_json(root / tool.ACTIVE)
    data["manifest_sha256"] = "0" * 64
    put(root / tool.ACTIVE, data)
    assert "artifact missing or stale" in tool.check_active(root)["failures"][0]


@pytest.mark.parametrize(
    "task",
    [
        "/root",
        "/root/p7_distillation_sources",
        "/root/p7_review_infrastructure_inventory",
        "/root/p7_mtp_sources_and_capstone",
    ],
)
def test_authors_researchers_and_tool_author_cannot_be_readers(setup, task):
    root, manifest, _ = setup
    declare(root, manifest, "reader", task=task)
    with pytest.raises(ValueError):
        tool.start_session(root, manifest, "reader", "a", task)


def test_start_requires_real_dispatch_shape_and_next_cannot_reveal_a_future_unit(setup):
    root, manifest, _ = setup
    with pytest.raises(OSError):
        tool.start_session(root, manifest, "reader", "a", "/root/fixture_reader")
    task = declare(root, manifest, "reader")
    packet = tool.start_session(root, manifest, "reader", "a", task)
    trace = root / packet["trace_file"]
    before = trace.read_bytes()
    bad = note(packet)
    bad["unit_index"] += 1
    with pytest.raises(ValueError, match="current unit"):
        tool.next_unit(root, packet["session"], bad)
    assert trace.read_bytes() == before
    while packet["page_id"] != "1.1":
        packet = tool.next_unit(root, packet["session"], note(packet))
    assert "UNREAD_FUTURE_FIXTURE" not in packet["text"]
    packet = tool.next_unit(root, packet["session"], note(packet))
    assert "UNREAD_FUTURE_FIXTURE" in packet["text"]


def test_section_end_and_first_use_need_individual_evidence_and_method_role(setup):
    root, manifest, _ = setup
    task = declare(root, manifest, "reader")
    packet = tool.start_session(root, manifest, "reader", "a", task)
    while not packet["section_end"]:
        packet = tool.next_unit(root, packet["session"], note(packet))
    bad = note(packet, method=True)
    bad["method_introductions"][0].pop("step_relation")
    with pytest.raises(ValueError, match="three separate"):
        tool.next_unit(root, packet["session"], bad)
    bad = note(packet)
    bad["four_questions"][0]["questions"][3]["claims"][0]["quote"] = "not actually revealed"
    with pytest.raises(ValueError, match="quotation is absent"):
        tool.next_unit(root, packet["session"], bad)
    good = note(packet, method=True)
    tool.next_unit(root, packet["session"], good)


def test_necessary_unknown_cannot_be_hidden_as_optional(setup):
    root, manifest, _ = setup
    task = declare(root, manifest, "reader")
    packet = tool.start_session(root, manifest, "reader", "a", task)
    bad = note(packet, method=True, issue="gap")
    bad["four_questions"][0]["questions"][0]["claims"] = [
        {
            "claim": "fixture missing role",
            "status": "unknown",
            "needed_now": True,
            "reason": "fixture need",
            "issue_id": "gap",
        }
    ]
    bad["issues"][0]["severity"] = "optional"
    with pytest.raises(ValueError, match="misclassified"):
        tool.next_unit(root, packet["session"], bad)


def test_same_count_unit_tampering_and_missing_question_refs_fail(setup):
    root, manifest, _ = setup
    task, trace, path, report = complete(root, manifest, "reader")
    assert not tool.audit_manifest(root, manifest, "reader")["failures"]
    raw = (root / trace).read_bytes()
    (root / trace).write_bytes(raw.replace(b"synthetic", b"synthetiX", 1))
    assert tool.audit_manifest(root, manifest, "reader")["failures"]
    (root / trace).write_bytes(raw)
    report["pages"][0]["question_refs"] = []
    changed = manifest.parent / "fixture-reports/missing-question-refs.json"
    put(changed, report)
    receipt = manifest.parent / "fixture-reports/missing-question-refs-receipt.json"
    put(
        receipt,
        {"final_received": True, "reviewer_task": task, "received_at": tool.now(), "receipt": "synthetic fixture"},
    )
    tool.collect(root, manifest, changed, receipt)
    assert tool.audit_manifest(root, manifest, "reader")["failures"]


def test_complete_stages_use_real_group_owners_and_initial_readiness(setup):
    root, manifest, _ = setup
    complete(root, manifest, "reader")
    reader_ready = tool.seal_stage(root, manifest, "reader")
    complete(root, manifest, "technical", reader_ready)
    technical_ready = tool.seal_stage(root, manifest, "technical")
    complete(root, manifest, "continuity", technical_ready)
    result = tool.audit_manifest(root, manifest)
    assert not result["failures"], result
    assert len({record["reviewer_task"] for record in result["records"]}) == 3
    assert result["unverified"], "fixture layout must remain explicitly unverified"


def test_missing_readiness_and_reused_owner_fail(setup):
    root, manifest, _ = setup
    reader_task, _, _, _ = complete(root, manifest, "reader")
    declare(root, manifest, "technical", task=reader_task)
    with pytest.raises(ValueError, match="reused"):
        tool.start_session(root, manifest, "technical", "a", reader_task)
    declaration = tool.read_json(manifest.parent / "dispatch.json")
    declaration["dispatches"].pop()
    put(manifest.parent / "dispatch.json", declaration)
    task = declare(root, manifest, "technical")
    with pytest.raises(ValueError, match="artifact reference"):
        tool.start_session(root, manifest, "technical", "a", task)


def test_revision_retains_unchanged_true_reads_and_callbacks_do_not_invalidate_initial_stage_order(setup):
    root, manifest, plan = setup
    completed = {"reader": complete(root, manifest, "reader")}
    ready = tool.seal_stage(root, manifest, "reader")
    completed["technical"] = complete(root, manifest, "technical", ready)
    ready = tool.seal_stage(root, manifest, "technical")
    completed["continuity"] = complete(root, manifest, "continuity", ready)
    old_bytes = manifest.read_bytes()
    chapter = root / "course/chapters/01.md"
    chapter.write_text(chapter.read_text().replace("UNREAD_FUTURE_FIXTURE", "REVISED_FUTURE_FIXTURE"), encoding="utf-8")
    new = root / "docs/fixture-review/freeze-02/manifest.json"
    tool.freeze(root, new.parent, plan, expected_pages=4, batch_id="fixture-only", previous=manifest)
    assert manifest.read_bytes() == old_bytes
    assert tool.audit_manifest(root, new)["failures"], "changed page cannot inherit old pass"
    for stage in tool.STAGES:
        task, original_trace, _, original_report = completed[stage]
        followup(new, stage, task)
        new_trace = read_fixture(root, new, stage, task, pages=["1.1"], recheck_of=[original_trace])
        _, updated = report_fixture(root, new, stage, task, new_trace, old=original_report, suffix="callback")
        unchanged = next(page for page in updated["pages"] if page["page_id"] == "1.2")
        assert unchanged["trace_file"] == original_trace
    result = tool.audit_manifest(root, new)
    assert not result["failures"], result
    assert manifest.read_bytes() == old_bytes


def test_callback_requires_actual_followup_and_cannot_drop_early_trace(setup):
    root, manifest, _ = setup
    task, trace, _, report = complete(root, manifest, "reader")
    with pytest.raises(ValueError, match="followup"):
        tool.start_session(root, manifest, "reader", "a", task, pages=["1.1"], recheck_of=[trace])
    followup(manifest, "reader", task)
    new_trace = read_fixture(root, manifest, "reader", task, pages=["1.1"], recheck_of=[trace])
    _, updated = report_fixture(root, manifest, "reader", task, new_trace, old=report, suffix="callback")
    assert not tool.audit_manifest(root, manifest, "reader")["failures"]
    updated["trace_files"] = [updated["trace_files"][-1]]
    bad_path = manifest.parent / "fixture-reports/dropped-history.json"
    put(bad_path, updated)
    receipt = manifest.parent / "fixture-reports/dropped-history-receipt.json"
    put(
        receipt,
        {"final_received": True, "reviewer_task": task, "received_at": tool.now(), "receipt": "synthetic fixture"},
    )
    tool.collect(root, manifest, bad_path, receipt)
    assert any("omitted original" in error for error in tool.audit_manifest(root, manifest, "reader")["failures"])


def test_render_png_can_stay_ignored_but_receipt_must_bind_actual_owner_and_hash(setup):
    root, _, _ = setup
    image = {"path": "outputs/fixture-640.png", "sha256": "a" * 64, "width": 640}
    mobile = {"path": "outputs/fixture-360.png", "sha256": "b" * 64, "width": 360}
    source = root / "fixture.svg"
    source.write_bytes(b"<svg>synthetic fixture only</svg>")
    receipt_path = root / "docs/fixture-review/view-receipt.json"
    receipt = {
        "tool": "view_image",
        "reviewer_task": "/root/fixture_reader",
        "artifacts": [image, mobile],
        "received_at": tool.now(),
        "source_sha256": tool.sha(source.read_bytes()),
    }
    put(receipt_path, receipt)
    value = {
        "status": "verified",
        "details": "synthetic contract fixture",
        "observation": "synthetic fixture observation",
        "figure": "fixture.svg",
        "source_sha256": tool.sha(source.read_bytes()),
        "artifacts": [image, mobile],
        "receipt": {"path": tool.relative(root, receipt_path), "sha256": tool.sha(receipt_path.read_bytes())},
    }
    assert tool.visual_check(root, value, "/root/fixture_reader", figure="fixture.svg")
    with pytest.raises(ValueError, match="reviewer"):
        tool.visual_check(root, value, "/root/another_fixture", figure="fixture.svg")
    value["artifacts"][0]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="rendered image"):
        tool.visual_check(root, value, "/root/fixture_reader", figure="fixture.svg")


def test_corrected_report_view_receipt_precedes_immutable_trace_candidate(setup, monkeypatch):
    root, manifest, _ = setup
    task, trace, report_path, report = complete(root, manifest, "reader")
    frozen = tool.read_json(manifest)
    source = next(page for page in frozen["pages"] if page["page_id"] == "1.1")
    page = next(page for page in report["pages"] if page["page_id"] == "1.1")
    figure = root / "fixture.svg"
    figure.write_bytes(b"<svg>synthetic correction fixture only</svg>")
    source["figures_sha256"] = page["figures_sha256"] = {"fixture.svg": tool.sha(figure.read_bytes())}
    images = [
        {"path": "outputs/fixture-640.png", "sha256": "a" * 64, "width": 640},
        {"path": "outputs/fixture-360.png", "sha256": "b" * 64, "width": 360},
    ]

    def checked_receipt(name, source_sha, artifacts, tool_name):
        target = root / "docs/fixture-review" / name
        put(
            target,
            {
                "tool": tool_name,
                "reviewer_task": task,
                "received_at": tool.now(),
                "source_sha256": source_sha,
                "artifacts": artifacts,
            },
        )
        return {"path": tool.relative(root, target), "sha256": tool.sha(target.read_bytes())}

    visual = {
        "figure": "fixture.svg",
        "source_sha256": source["figures_sha256"]["fixture.svg"],
        "status": "verified",
        "details": "synthetic additive correction",
        "observation": "synthetic fixture only",
        "artifacts": images,
    }
    stale = copy.deepcopy(visual)
    stale["receipt"] = checked_receipt(
        "old-view.json", visual["source_sha256"], images, "functions.exec -> tools.view_image"
    )
    visual["receipt"] = checked_receipt("corrected-view.json", visual["source_sha256"], images, "tools.view_image")
    page["visual_checks"] = [visual]
    layout_images = [
        {"path": "outputs/fixture-desktop.png", "sha256": "c" * 64, "viewport": [1280, 800]},
        {"path": "outputs/fixture-mobile.png", "sha256": "d" * 64, "viewport": [390, 844]},
    ]
    page["page_visual_check"] = {
        "status": "verified",
        "details": "synthetic fixture layout",
        "observation": "synthetic fixture only",
        "source_sha256": source["source_sha256"],
        "artifacts": layout_images,
        "receipt": checked_receipt("layout.json", source["source_sha256"], layout_images, "tools.view_image"),
    }
    original_rows = tool.raw_trace(root, trace)
    synthetic_rows = copy.deepcopy(original_rows)
    next(row for row in synthetic_rows[1:-1] if row["page_id"] == "1.1")["visual_checks"] = [stale]
    # This monkeypatch represents immutable historical metadata; no real review
    # or trace is altered and all receipt validation uses the real checker.
    monkeypatch.setattr(tool, "verify_trace", lambda *args: synthetic_rows)
    entry = tool.collection_entries(root, manifest)[("reader", "a")]

    def check():
        put(report_path, report)
        entry["sha256"] = tool.sha(report_path.read_bytes())
        return tool.group_report(root, manifest, frozen, frozen["groups"][0], "reader", entry, [])

    assert check()[0] == task
    assert tool.raw_trace(root, trace) == original_rows
    page["visual_checks"] = []
    with pytest.raises(ValueError, match="reviewer/tool"):
        check()
    page["visual_checks"] = [visual]
    visual["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source figure"):
        check()


def test_criteria_change_invalidates_old_results_and_same_batch_refresh(setup):
    root, manifest, plan = setup
    criterion = root / tool.CRITERIA[1]
    original = criterion.read_bytes()
    criterion.write_bytes(original + b"Changed synthetic criterion.\n")
    with pytest.raises(ValueError, match="criteria changed"):
        tool.load_manifest(root, manifest)
    with pytest.raises(ValueError, match="criteria changed"):
        tool.freeze(
            root,
            root / "docs/fixture-review/freeze-02",
            plan,
            expected_pages=4,
            batch_id="fixture-only",
            previous=manifest,
        )
    assert tool.audit_manifest(root, manifest, "reader")["failures"]


def test_context_is_actual_five_field_reading_without_formal_four_questions(setup):
    root, manifest, _ = setup
    task = declare(root, manifest, "reader")
    packet = tool.start_session(root, manifest, "reader", "a", task, context_only=True, context=["1.1"])
    trace = packet["trace_file"]
    assert packet["primary"] is False
    while not packet.get("complete"):
        assert packet["four_questions_required"] is False
        current = note(packet)
        assert "four_questions" not in current
        packet = tool.next_unit(root, packet["session"], current)
    rows = tool.verify_trace(
        root, {"path": trace, "sha256": tool.sha((root / trace).read_bytes())}, task, "reader", "a", "fixture-only"
    )
    assert all(not row["four_questions"] for row in rows[1:-1])
    primary = tool.start_session(root, manifest, "reader", "a", task)
    future_claim = {
        "claim": "fixture prerequisite",
        "status": "prior",
        "trace_file": trace,
        "event_index": rows[-2]["event_index"],
        "quote": "not actually in read context",
    }
    bad = note(primary)
    bad["claims"] = [future_claim]
    with pytest.raises(ValueError, match="actually read unit"):
        tool.next_unit(root, primary["session"], bad)
    bad["claims"][0]["quote"] = "UNREAD_FUTURE_FIXTURE"
    tool.next_unit(root, primary["session"], bad)


def test_technical_and_continuity_have_own_records_without_reader_four_questions(setup):
    root, manifest, _ = setup
    complete(root, manifest, "reader")
    ready = tool.seal_stage(root, manifest, "reader")
    _, trace, _, report = complete(root, manifest, "technical", ready)
    assert all(not row["four_questions"] for row in tool.raw_trace(root, trace)[1:-1])
    assert all(page["question_refs"] == [] for page in report["pages"])
    ready = tool.seal_stage(root, manifest, "technical")
    _, trace, _, report = complete(root, manifest, "continuity", ready)
    assert all(not row["four_questions"] for row in tool.raw_trace(root, trace)[1:-1])
    assert all(page["question_refs"] == [] for page in report["pages"])
    assert not tool.audit_manifest(root, manifest)["failures"]


def test_necessary_issue_requires_retained_original_and_a_new_owner_session(setup):
    root, manifest, _ = setup
    task = declare(root, manifest, "reader")
    packet = tool.start_session(root, manifest, "reader", "a", task)
    trace = packet["trace_file"]
    first_page = packet["page_id"]
    packet = tool.next_unit(root, packet["session"], note(packet, issue="fixture-gap"))
    while not packet.get("complete"):
        packet = tool.next_unit(root, packet["session"], note(packet))
    _, original = report_fixture(root, manifest, "reader", task, trace)
    assert any("early issue" in failure for failure in tool.audit_manifest(root, manifest, "reader")["failures"])
    followup(manifest, "reader", task)
    new_trace = read_fixture(
        root, manifest, "reader", task, pages=[first_page], recheck_of=[trace], issues=["fixture-gap"]
    )
    _, updated = report_fixture(root, manifest, "reader", task, new_trace, old=original, suffix="callback")
    issue = copy.deepcopy(tool.raw_trace(root, trace)[1]["issues"][0])
    issue.update(
        {
            "status": "resolved",
            "resolution": "Synthetic actual-recheck contract fixture only",
            "recheck": {"trace_file": new_trace, "event_index": 1},
        }
    )
    next(page for page in updated["pages"] if page["page_id"] == first_page)["issues"] = [issue]
    resolved = manifest.parent / "fixture-reports/issue-resolved.json"
    put(resolved, updated)
    receipt = manifest.parent / "fixture-reports/issue-resolved-receipt.json"
    put(
        receipt,
        {"final_received": True, "reviewer_task": task, "received_at": tool.now(), "receipt": "synthetic fixture"},
    )
    tool.collect(root, manifest, resolved, receipt)
    assert not tool.audit_manifest(root, manifest, "reader")["failures"]
    issue["recheck"] = {"trace_file": trace, "event_index": 1}
    bad = manifest.parent / "fixture-reports/same-session-recheck.json"
    put(bad, updated)
    tool.collect(root, manifest, bad, receipt)
    assert any("recheck" in failure for failure in tool.audit_manifest(root, manifest, "reader")["failures"])
