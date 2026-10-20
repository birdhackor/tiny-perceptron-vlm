"""協調者的逐頁 preflight 與問題清單，在一頁虛構的讀者流程上必須抓到連錯的判斷、漏收的概念與過期的複查。

流程照新一輪的讀者交件：原始判斷 O 有必要問題；新凍結檔、新閱讀計畫下重讀得到 R；目前判斷 C 的 basis 指向 R；
收束 L 的頂層 basis_report 是 C 的 {path, sha256}；一筆 fixed 處置由 R 複查。全部是虛構資料。

驗證：preflight 在 intake 與 sealed 兩種模式下，對 C 的 basis 指向舊版、L 漏收或收錯概念、L 的 basis_report 連錯、
seal 漏列原始判斷、inherit 條件不符的反應，以及一頁的錯不影響另一頁；三項協調者核對（basis 那一版沒有讀完的
session、閱讀計畫不含該頁、technical 的 figure_sha256 多列圖）只記在 coordinator_checks；issue_inventory 對複查者
不是原提出者、複查版本過期、兩份報告各有同 id 的問題、issue_reports 漏列必要報告的反應；草稿帶 author_patches 時，
作者補之前的讀者報告、技術報告與它引用的頁檔、修正的複查照樣通過，靠補丁的頁列在 author_patches，清單倒退不回
現行檔時記錯、不承認任何較早版本；CLI 的說明文字與 exit code；搬寫的段落與現行閘門原始碼相同。
不涵蓋：真實 reader.py 的單元切法（以 monkeypatch 的固定切法代替）、section-composite 分支、tool_audit 型的
複查、多個 wave 之間的先後；這些在這裡沒有測試守著。
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "docs/course-review-20261019/tools"
READER = "/root/fixture_reader"
TECHNICAL = "/root/fixture_technical"
WAVE = "fixture-wave"
PID = "fixture"
FIRST = "first unit, first version\n\nsecond unit, first version\n"
REVISED = "first unit, revised\n\nsecond unit, revised\n"
NOTE_FIELDS = ("understanding", "materials_and_labels", "expected_change", "confusion_and_quote", "missing_visuals")
CONCEPT_FIELDS = ("concept", "identity", "mechanism", "need", "usage", "example_scope", "source_quote")
NEEDS_DETAIL = "需補說明：沒說為什麼要分成兩份"
LATER = "合理待教：下一頁才教"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("course_review_preflight", TOOLS / "preflight.py")
inventory = load("course_review_issue_inventory", TOOLS / "issue_inventory.py")
validator = load("course_review_technical_validator", ROOT / "scripts/check_technical_reviews.py")


def save(root, name, value, raw=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def at(minute):
    return f"2026-10-01T01:{minute:02d}:00+00:00"


def units(text):
    return text.strip().split("\n\n")


@pytest.fixture(autouse=True)
def fixed_units(monkeypatch):
    # 單元切法由這裡指定；測試核對紀錄之間的連結，不複製 reader.py 的分段演算法。
    monkeypatch.setattr(preflight.review, "module", lambda *_: SimpleNamespace(units=units))


def reading(root, name, source, freeze, plan, start, verdicts):
    """reader.py 會留下的一個 session：opening、state 與逐單元 note；verdicts 是每個單元裡各概念的判定。"""
    chunks = units((root / source["path"]).read_bytes().decode())
    opening = {
        "schema_version": 1,
        "session": name,
        "page_id": PID,
        "source_sha256": source["sha256"],
        "figures_sha256": {},
        "reviewer_task": READER,
        "next_index": 0,
        "started_at": at(start),
        "reading_plan": plan,
        "source_freeze": freeze,
    }
    reference = {"opening": save(root, f"sessions/{name}/opening.json", opening), "source": source}
    save(root, f"sessions/{name}/state.json", {**opening, "next_index": len(chunks)})
    reference["notes"] = []
    reference["figures"] = {}
    for i, (chunk, concepts) in enumerate(zip(chunks, verdicts)):
        note = {
            "unit_index": i,
            "page_id": PID,
            "reviewer_task": READER,
            "unit_sha256": hashlib.sha256(chunk.encode()).hexdigest(),
            "saved_at": at(start + 1 + i),
            **{field: "虛構觀察" for field in NOTE_FIELDS},
            "concept_checks": [{**{field: "虛構回答" for field in CONCEPT_FIELDS}, "verdict": v} for v in concepts],
        }
        reference["notes"].append(save(root, f"sessions/{name}/note-{i:03}.json", note))
    return reference


def judgment(root, name, created, source, verdict, **extra):
    page = {"page_id": PID, "source_sha256": source, "figures_sha256": {}, "verdict": verdict, **extra}
    return save(root, name, {"schema_version": 1, "reviewer_task": READER, "created_at": at(created), "pages": [page]})


def resolution(note, index, verdict, evidence):
    return {
        "note": note,
        "concept_index": index,
        "original_verdict": verdict,
        "final_status": "resolved",
        "reason": "重讀時已讀到補上的理由",
        "evidence": [{"report": evidence, "json_pointer": "/pages/0"}],
    }


def disposition(report, verification, issue_id="R-1"):
    return {
        "issue_id": issue_id,
        "report": report,
        "json_pointer": "/pages/0/issues/0",
        "final_necessity": "necessary",
        "action": "fixed",
        "source_support": "原文沒說為什麼要分成兩份",
        "learning_benefit": "讀者知道分開的用途",
        "change_burden": "補一句理由",
        "reason": "照原問題補上理由",
        "affected_pages": [PID],
        "verification": [{"report": verification, "json_pointer": "/pages/0"}],
    }


def pilot(root, variant=None):
    """虛構一頁讀者流程的全部紀錄；variant 指定要打進去的一個錯。"""
    first = save(root, "freeze-1/fixture.md", FIRST.encode(), raw=True)
    revised = save(root, "freeze-2/fixture.md", REVISED.encode(), raw=True)
    neighbor = save(root, "freeze-2/neighbor.md", b"neighbor page\n", raw=True)
    page = {"page_id": PID, "source": "course/fixture.md", "selector": "whole-file", "figures_sha256": {}}
    first_page = {**page, "source_sha256": first["sha256"], "snapshot": first["path"]}
    revised_page = {**page, "source_sha256": revised["sha256"], "snapshot": revised["path"]}
    neighbor_page = {**page, "page_id": "neighbor", "source": "course/neighbor.md", "source_sha256": neighbor["sha256"]}
    freeze_1 = save(root, "freeze-1.json", {"pages": [first_page], "figure_snapshots": {}})
    freeze_2 = save(root, "freeze-2.json", {"pages": [revised_page, {**neighbor_page, "snapshot": neighbor["path"]}]})
    plan = {"reviewer_task": READER, "author_task": "/root", "context_pages": []}
    plan_1 = save(root, "plan-1.json", {**plan, "source_freeze": freeze_1, "main_pages": [PID], "created_at": at(0)})
    main_pages = ["another-page"] if variant == "plan_lacks_page" else [PID]
    plan_2 = save(
        root, "plan-2.json", {**plan, "source_freeze": freeze_2, "main_pages": main_pages, "created_at": at(20)}
    )
    first_read = reading(root, "s1", first, freeze_1, plan_1, 5, [["已足夠", NEEDS_DETAIL], [LATER]])
    reread = reading(root, "s2", revised, freeze_2, plan_2, 25, [["已足夠"], ["已足夠"]])
    issue = {"id": "R-1", "unit_index": 0, "necessary": True, "problem": "沒說為什麼要分成兩份"}
    original = judgment(root, "reviews/original.json", 10, first["sha256"], "需補說明", issues=[issue])
    later_issue = {"id": "R-1", "unit_index": 1, "necessary": False, "problem": "選讀可以再給一例"}
    recheck_issues = [later_issue] if variant == "recheck_same_issue_id" else []
    recheck = judgment(root, "reviews/recheck.json", 30, revised["sha256"], "pass", issues=recheck_issues)
    basis = original if variant == "basis_original" else recheck
    estimate = {"minutes_min": 10, "minutes_max": 15, "reason": "虛構估計"}
    current = judgment(
        root, "reviews/current.json", 31, revised["sha256"], "pass", issues=[], ai_estimate=estimate, basis_report=basis
    )
    resolutions = [
        resolution(first_read["notes"][0], 1, NEEDS_DETAIL, recheck),
        resolution(first_read["notes"][1], 0, LATER, recheck),
    ]
    if variant == "closure_misses_concept":
        resolutions.pop()
    if variant == "closure_wrong_index":
        resolutions[0]["concept_index"] = 0  # 同一份 note 裡判「已足夠」的那個概念
    closure_basis = {"closure_basis_path": current["path"], "closure_basis_recheck": recheck}.get(variant, current)
    closure_page = {
        "page_id": PID,
        "source_sha256": revised["sha256"],
        "figures_sha256": {},
        "verdict": "pass",
        "checkpoint_resolutions": resolutions,
    }
    closure = save(
        root,
        "reviews/closure.json",
        {
            "schema_version": 1,
            "reviewer_task": READER,
            "created_at": at(32),
            "basis_report": closure_basis,
            "pages": [closure_page],
        },
    )
    collected = [current, recheck, closure] + ([] if variant == "seal_omits_original" else [original])
    seal = save(
        root,
        "reader-seal.json",
        {
            "role": "reader",
            "status": "completed",
            "started_at": at(0),
            "completed_at": at(40),
            "report_sha256s": [ref["sha256"] for ref in collected],
        },
    )
    reader_task = {
        "task": READER,
        "role": "reader",
        "wave": WAVE,
        "context": "fresh",
        "pages": [PID, "neighbor"],
        "dispatch_window": {"not_before": at(1), "not_after": at(2)},
    }
    # 技術 task 只用來當「別的 task 出具的複查」；stage reader 不核它的派工時間窗。
    technical_task = {
        "task": TECHNICAL,
        "role": "technical",
        "wave": WAVE,
        "context": "fresh",
        "lesson_id": PID,
        "dispatch_window": {"not_before": at(41), "not_after": at(42)},
    }
    assignments = save(root, "assignments.json", {"tasks": [reader_task, technical_task]})
    proof = {
        "mode": "review",
        "wave": WAVE,
        "report": current,
        "json_pointer": "/pages/0",
        "original_report": original,
        "original_json_pointer": "/pages/0",
        "basis": {"report": basis, "json_pointer": "/pages/0"},
        "sessions": [first_read] if variant == "unread_revision" else [first_read, reread],
        "checkpoint_closure": {
            "report": closure,
            "json_pointer": "/pages/0",
            "wave": WAVE,
            "collection_role": "reader",
            "received_at": at(33),
        },
    }
    dispositions = save(
        root,
        "dispositions.json",
        {"author_task": "/root", "at": at(34), "pending_count": 0, "items": [disposition(original, recheck)]},
    )
    common = {
        "source_freeze": freeze_2,
        "assignments": assignments,
        "author_tasks": ["/root"],
        "pages": [{"page_id": PID, "reader": proof}],
        "issue_reports": [current, original, recheck, closure],
        "dispositions": dispositions,
        "dispositions_collection": {"role": "reader", "wave": WAVE, "received_at": at(35)},
    }
    return SimpleNamespace(
        root=root,
        refs=SimpleNamespace(original=original, recheck=recheck, current=current, closure=closure, revised=revised),
        live={PID: {key: value for key, value in revised_page.items() if key != "snapshot"}, "neighbor": neighbor_page},
        drafts={
            "intake": {**common, "review_waves": [{"id": WAVE, "seals": {}}]},
            "sealed": {**common, "review_waves": [{"id": WAVE, "seals": {"reader": seal}}]},
        },
    )


def check(fixture, mode, draft=None, live=None):
    return preflight.check_draft(fixture.root, draft or fixture.drafts[mode], live or fixture.live, mode, "reader")


def issues(fixture, mode, draft=None, live=None):
    return inventory.inventory(fixture.root, draft or fixture.drafts[mode], live or fixture.live, mode, "reader")


def test_complete_flow_passes_and_intake_lists_seal_dependent_checks(tmp_path):
    fixture = pilot(tmp_path)
    intake = check(fixture, "intake")
    assert intake["failing"] == []
    pending = " ".join(intake["shared"]["pending_seal"] + intake["pages"][PID]["reader"]["pending_seal"])
    for expected in (
        f"{WAVE}/reader seal 尚未寫入",
        "派工時間窗 " + READER,
        "報告 reviews/current.json 是否列在",
        "原始判斷 reviews/original.json 是否列在",
        "session s1：opening 與最後 checkpoint",
        "session s2：opening 與最後 checkpoint",
        "closure 建立時間是否不早於",
        "closure 的 received_at 是否落在",
    ):
        assert expected in pending
    assert "局部 preflight，不是發布閘門" in intake["note"]
    sealed = check(fixture, "sealed")
    assert sealed["failing"] == []
    assert sealed["shared"]["pending_seal"] == [] and sealed["pages"][PID]["reader"]["pending_seal"] == []
    for mode in ("intake", "sealed"):
        result = issues(fixture, mode)
        assert result["failing"] == [], result
        assert result["issues"] == [{"report": fixture.refs.original, "json_pointer": "/pages/0/issues/0", "id": "R-1"}]
    assert issues(fixture, "intake")["pending_seal"] == [f"處置的 received_at 是否落在 {WAVE}/reader seal 期間"]


@pytest.mark.parametrize("mode", ["intake", "sealed"])
@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("basis_original", "current judgment lacks read revised source"),
        ("closure_misses_concept", "original non-pass concept lacks reader final closure"),
        ("closure_wrong_index", "original non-pass concept lacks reader final closure"),
        ("closure_basis_path", "checkpoint closure not linked to own current judgment"),
        ("closure_basis_recheck", "checkpoint closure not linked to own current judgment"),
    ],
)
def test_wrong_reader_links_are_rejected_in_both_modes(tmp_path, mode, variant, expected):
    result = check(pilot(tmp_path, variant), mode)
    assert result["pages"][PID]["reader"]["errors"] == [expected]
    assert "fixture/reader" in result["failing"]


def test_seal_omitting_original_is_rejected_once_sealed_and_deferred_at_intake(tmp_path):
    fixture = pilot(tmp_path, "seal_omits_original")
    assert check(fixture, "sealed")["pages"][PID]["reader"]["errors"] == ["original report not sealed"]
    intake = check(fixture, "intake")["pages"][PID]["reader"]
    assert intake["errors"] == []
    assert any(item.startswith("原始判斷 reviews/original.json") for item in intake["pending_seal"])


@pytest.mark.parametrize("mode", ["intake", "sealed"])
def test_basis_without_complete_reread_is_flagged_only_by_coordinator(tmp_path, mode):
    result = check(pilot(tmp_path, "unread_revision"), mode)
    assert result["pages"][PID]["reader"]["errors"] == []
    errors = result["coordinator_checks"][PID]["basis_read_by_complete_session"]["errors"]
    assert errors and "reviews/recheck.json#/pages/0" in errors[0]
    assert result["failing"] == ["coordinator fixture/basis_read_by_complete_session"]


def test_reading_plan_without_the_page_is_flagged_only_by_coordinator(tmp_path):
    result = check(pilot(tmp_path, "plan_lacks_page"), "sealed")
    assert result["pages"][PID]["reader"]["errors"] == []
    errors = result["coordinator_checks"][PID]["session_reading_plans"]["errors"]
    assert errors == ["session sessions/s2/opening.json: 閱讀計畫的 main_pages 不含這一頁"]


def test_one_broken_page_does_not_hide_another_pages_result(tmp_path):
    fixture = pilot(tmp_path)
    draft = dict(fixture.drafts["sealed"])
    broken = {"mode": "review", "wave": WAVE, "report": {"path": "reviews/missing.json", "sha256": "0" * 64}}
    draft["pages"] = [{"page_id": "neighbor", "reader": broken}] + draft["pages"]
    result = check(fixture, "sealed", draft)
    assert result["pages"]["neighbor"]["reader"]["errors"][0].startswith("FileNotFoundError")
    assert result["pages"][PID]["reader"]["errors"] == []
    # 讀不到報告時，協調者核對也記「無法核對」，不當成通過。
    assert result["failing"] == ["neighbor/reader", "coordinator neighbor/basis_read_by_complete_session"]


def test_fixed_disposition_rechecked_by_another_task_is_rejected(tmp_path):
    fixture = pilot(tmp_path)
    other = save(
        tmp_path,
        "reviews/technical-recheck.json",
        {
            "reviewer_task": TECHNICAL,
            "pages": [
                {
                    "page_id": PID,
                    "source_sha256": fixture.refs.revised["sha256"],
                    "figures_sha256": {},
                    "verdict": "pass",
                }
            ],
        },
    )
    draft = dict(fixture.drafts["sealed"])
    draft["dispositions"] = save(
        tmp_path,
        "dispositions-other.json",
        {"at": at(34), "pending_count": 0, "items": [disposition(fixture.refs.original, other)]},
    )
    failing = issues(fixture, "sealed", draft)["dispositions"]["failing"]
    assert len(failing) == 1
    assert (
        "verification reviews/technical-recheck.json: fix was not rechecked by original reviewer"
        in failing[0]["errors"]
    )


def test_fixed_disposition_on_outdated_page_version_is_rejected(tmp_path):
    fixture = pilot(tmp_path)
    edited_again = {**fixture.live, PID: {**fixture.live[PID], "source_sha256": "c" * 64}}
    failing = issues(fixture, "sealed", live=edited_again)["dispositions"]["failing"]
    assert "verification reviews/recheck.json: fix recheck is stale" in failing[0]["errors"]


def test_same_issue_id_in_two_reports_needs_its_own_disposition(tmp_path):
    fixture = pilot(tmp_path, "recheck_same_issue_id")
    result = issues(fixture, "sealed")
    later = {"report": fixture.refs.recheck, "json_pointer": "/pages/0/issues/0", "id": "R-1"}
    assert result["undisposed"] == [later]
    keep = {
        **disposition(fixture.refs.recheck, fixture.refs.recheck),
        "final_necessity": "optional",
        "action": "keep",
        "verification": [],
    }
    draft = dict(fixture.drafts["sealed"])
    draft["dispositions"] = save(
        tmp_path,
        "dispositions-both.json",
        {"at": at(34), "pending_count": 0, "items": [disposition(fixture.refs.original, fixture.refs.recheck), keep]},
    )
    assert issues(fixture, "sealed", draft)["failing"] == []


def test_issue_reports_must_list_every_required_report(tmp_path):
    fixture = pilot(tmp_path)
    draft = dict(fixture.drafts["intake"])
    draft["issue_reports"] = [fixture.refs.current, fixture.refs.original, fixture.refs.recheck]
    result = issues(fixture, "intake", draft)
    assert result["missing_issue_reports"] == [fixture.refs.closure]
    assert "missing_issue_reports" in result["failing"]


def technical_page(root, figure_sha256):
    """一頁沒有實質主張、能通過 _validate 的技術報告；figure_sha256 由呼叫者指定。"""
    main = save(root, "course/figures/main.svg", b"<svg/>", raw=True)
    snapshot = save(root, "freeze/fixture.md", b"# Fixture\n\n![diagram](figures/main.svg)\n", raw=True)
    figures = {main["path"]: main["sha256"]}
    page = {"page_id": PID, "source": "course/fixture.md", "selector": "whole-file", "figures_sha256": figures}
    page["source_sha256"] = snapshot["sha256"]
    freeze = save(root, "freeze.json", {"pages": [{**page, "snapshot": snapshot["path"]}]})
    checks = {
        name: {"status": "pass" if name == "figure_consistency" else "not_applicable", "details": "虛構核對"}
        for name in validator.CHECKS
    }
    report = save(
        root,
        "reviews/technical.json",
        {
            "schema_version": 1,
            "review_stage": "technical",
            "lesson_id": PID,
            "source": "course/fixture.md",
            "source_sha256": snapshot["sha256"],
            "verdict": "pass",
            "reviewer_task": TECHNICAL,
            "reviewer_context": "fresh",
            "figures_sha256": figures,
            "figure_sha256": figure_sha256(root, figures),
            "artifacts": [],
            "sources": [],
            "claims": [],
            "applicability": {"substantive_claims": False, "reason": "虛構頁沒有實質主張"},
            "issues": [],
            "checks": checks,
        },
    )
    assignment = {"task": TECHNICAL, "role": "technical", "wave": WAVE, "context": "fresh", "lesson_id": PID}
    assignment["dispatch_window"] = {"not_before": at(1), "not_after": at(2)}
    proof = {"mode": "review", "wave": WAVE, "report": report, "json_pointer": "", "original_report": report}
    draft = {
        "source_freeze": freeze,
        "assignments": save(root, "assignments.json", {"tasks": [assignment]}),
        "author_tasks": ["/root"],
        "review_waves": [{"id": WAVE, "seals": {}}],
        "pages": [
            {"page_id": PID, "technical": {**proof, "original_json_pointer": "", "collection_role": "technical"}}
        ],
    }
    return preflight.check_draft(root, draft, {PID: page}, "intake", "technical")


def with_prerequisite(root, figures):
    extra = save(root, "course/figures/prerequisite.svg", b"<svg></svg>", raw=True)
    return {**figures, extra["path"]: extra["sha256"]}


@pytest.mark.parametrize(("figure_sha256", "flagged"), [(lambda _, figures: figures, False), (with_prerequisite, True)])
def test_extra_figure_in_technical_report_is_flagged_only_by_coordinator(tmp_path, figure_sha256, flagged):
    result = technical_page(tmp_path, figure_sha256)
    # 這一頁沒有 reader 證據，stage technical 會在 reader 記錯；這裡只看 technical 與協調者核對。
    assert result["pages"][PID]["technical"]["errors"] == []
    assert bool(result["coordinator_checks"][PID]["technical_figure_inventory"]["errors"]) is flagged


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (None, []),
        ("source", ["changed source cannot inherit old verdict: fixture"]),
        ("note", ["inheritance needs scope/route disposition"]),
    ],
)
def test_inherit_needs_unchanged_baseline_page_and_route_note(tmp_path, change, expected):
    old = save(tmp_path, "baseline/fixture.md", b"unchanged page\n", raw=True)
    page = {"page_id": PID, "source": "course/fixture.md", "selector": "whole-file", "figures_sha256": {}}
    baseline = save(
        tmp_path, "baseline.json", {"pages": [{**page, "source_sha256": old["sha256"], "snapshot": old["path"]}]}
    )
    save(tmp_path, "docs/course-publish-20261008/review-manifest.json", {"source_freeze": baseline})
    current = save(tmp_path, "freeze/fixture.md", b"edited page\n", raw=True) if change == "source" else old
    live = {**page, "source_sha256": current["sha256"]}
    freeze = save(tmp_path, "freeze.json", {"pages": [{**live, "snapshot": current["path"]}]})
    proof = {"mode": "inherit", "baseline_page_id": PID, "dependency_note": "" if change == "note" else "先備沒變"}
    draft = {
        "source_freeze": freeze,
        "assignments": save(tmp_path, "assignments.json", {"tasks": []}),
        "author_tasks": ["/root"],
        "review_waves": [{"id": WAVE, "seals": {}}],
        "pages": [{"page_id": PID, "reader": proof}],
    }
    result = preflight.check_draft(tmp_path, draft, {PID: live}, "intake", "reader")
    assert result["pages"][PID]["page"]["errors"] == []
    assert result["pages"][PID]["reader"]["errors"] == expected


def test_cli_prints_local_scope_and_exits_nonzero_on_any_error(tmp_path, capsys):
    good = pilot(tmp_path / "good")
    unlinked = pilot(tmp_path / "unlinked", "closure_basis_path")
    undisposed = pilot(tmp_path / "undisposed", "recheck_same_issue_id")
    cases = [
        (preflight, good, 0, []),
        (preflight, unlinked, 1, ["fixture/reader"]),
        (inventory, good, 0, []),
        (inventory, undisposed, 1, ["undisposed"]),
    ]
    for tool, fixture, code, failing in cases:
        path = fixture.root / "draft.json"
        path.write_text(json.dumps(fixture.drafts["intake"]), encoding="utf-8")
        assert tool.run(path, "intake", "reader", fixture.root, fixture.live) == code
        output = json.loads(capsys.readouterr().out)
        assert "不是發布閘門" in output["note"]
        assert output["failing"] == failing


def test_transcribed_checks_still_match_the_current_gate():
    # 失敗時：閘門的對應段落改了，先對照兩支工具裡註明「搬自」的段落，再更新 preflight.TRANSCRIBED。
    assert preflight.drift() == []


def test_basis_with_only_new_figures_needs_a_session_that_saw_them(tmp_path):
    fixture = pilot(tmp_path)
    proof = fixture.drafts["sealed"]["pages"][0]["reader"]
    assert preflight.basis_read(fixture.root, proof) == []
    # 只改圖：basis 頁的正文還是同一版，圖換成新的一張；列出的 session 都只看過舊的圖清單。
    recheck = json.loads((tmp_path / fixture.refs.recheck["path"]).read_bytes())
    recheck["pages"][0]["figures_sha256"] = {"course/figures/new.svg": "0" * 64}
    basis = {"report": save(tmp_path, "reviews/recheck-figure.json", recheck), "json_pointer": "/pages/0"}
    errors = preflight.basis_read(fixture.root, {**proof, "basis": basis})
    assert errors and "reviews/recheck-figure.json#/pages/0" in errors[0]


# 以下測作者補丁：R、C、L 與複查都審過 REVISED 之後，作者直接補了半句，目前清單與凍結檔是補過的版本。
PATCHED = "first unit, revised\n\nsecond unit, revised by the author\n"
PATCH = {
    "id": "fixture-1",
    "path": "course/fixture.md",
    "old": "second unit, revised",
    "new": "second unit, revised by the author",
    "requested_by": "author",
    "requested_on": "2026-10-20",
    "reason": "作者直接要求補半句",
}


def patch_list(root, patches):
    data = {
        "schema_version": 1,
        "kind": preflight.review.PATCHES_KIND,
        "statement": preflight.review.PATCHES_STATEMENT,
        "patches": patches,
    }
    return save(root, "author-patches.json", data)


def author_patched(fixture, patches=(PATCH,)):
    """目前清單與新凍結檔換成補過的版本；patches 不是 None 時兩份草稿都帶作者補丁清單。"""
    root = fixture.root
    save(root, "course/fixture.md", PATCHED.encode(), raw=True)
    snapshot = save(root, "freeze-3/fixture.md", PATCHED.encode(), raw=True)
    page = {**fixture.live[PID], "source_sha256": snapshot["sha256"]}
    neighbor = {**fixture.live["neighbor"], "snapshot": "freeze-2/neighbor.md"}
    freeze = save(root, "freeze-3.json", {"pages": [{**page, "snapshot": snapshot["path"]}, neighbor]})
    extra = {} if patches is None else {"author_patches": patch_list(root, list(patches))}
    drafts = {mode: {**draft, "source_freeze": freeze, **extra} for mode, draft in fixture.drafts.items()}
    return drafts, {**fixture.live, PID: page}


@pytest.mark.parametrize("mode", ["intake", "sealed"])
def test_reader_reports_before_an_author_patch_pass_and_are_listed(tmp_path, mode):
    fixture = pilot(tmp_path)
    drafts, live = author_patched(fixture)
    result = check(fixture, mode, drafts[mode], live)
    assert result["failing"] == []
    assert result["author_patches"] == {PID: ["fixture-1"]}


@pytest.mark.parametrize("mode", ["intake", "sealed"])
def test_draft_without_the_patch_list_is_checked_as_before_and_lists_nothing(tmp_path, mode):
    # 同一頁、同一份草稿不帶清單：和以前一樣是過期，輸出也沒有 author_patches 這個鍵。
    fixture = pilot(tmp_path)
    drafts, live = author_patched(fixture, None)
    result = check(fixture, mode, drafts[mode], live)
    assert result["pages"][PID]["reader"]["errors"] == ["report source is stale: fixture"]
    assert "author_patches" not in result
    assert "author_patches" not in check(fixture, mode)


def test_draft_without_patches_lists_nothing_even_when_the_field_is_present(tmp_path):
    fixture = pilot(tmp_path)
    draft = {**fixture.drafts["sealed"], "author_patches": patch_list(tmp_path, [])}
    result = check(fixture, "sealed", draft)
    assert result["failing"] == [] and result["author_patches"] == {}


def test_fix_recheck_before_an_author_patch_is_accepted_by_the_issue_inventory(tmp_path):
    fixture = pilot(tmp_path)
    drafts, live = author_patched(fixture)
    assert issues(fixture, "sealed", drafts["sealed"], live)["failing"] == []
    drafts, live = author_patched(fixture, None)
    failing = issues(fixture, "sealed", drafts["sealed"], live)["dispositions"]["failing"]
    assert "verification reviews/recheck.json: fix recheck is stale" in failing[0]["errors"]


def test_patch_list_that_does_not_lead_back_is_an_error_and_accepts_no_older_version(tmp_path):
    fixture = pilot(tmp_path)
    drafts, live = author_patched(fixture, [{**PATCH, "new": "second unit, revised by someone else"}])
    undo = (
        "作者補丁清單的 patches[0]（id 'fixture-1'）：new 在倒退到這一筆時的 course/fixture.md 出現 0 次，必須恰好一次"
    )
    result = check(fixture, "sealed", drafts["sealed"], live)
    assert result["shared"]["errors"] == ["author_patches: " + undo]
    assert result["pages"][PID]["reader"]["errors"] == ["report source is stale: fixture"]
    assert result["author_patches"] == {}
    listed = issues(fixture, "sealed", drafts["sealed"], live)
    assert listed["errors"] == ["author_patches: " + undo]
    assert "verification reviews/recheck.json: fix recheck is stale" in listed["dispositions"]["failing"][0]["errors"]


def technical_patched(root, listed):
    """一頁技術報告與它當 repository_code 引用的頁檔都是作者補之前的版本；listed 時草稿帶作者補丁清單。"""
    main = save(root, "course/figures/main.svg", b"<svg/>", raw=True)
    reviewed = b"# Fixture\n\n![diagram](figures/main.svg)\n\nweight grows four times\n"
    patched = b"# Fixture\n\n![diagram](figures/main.svg)\n\nweight becomes four times as large\n"
    save(root, "course/fixture.md", patched, raw=True)
    snapshot = save(root, "freeze/fixture.md", patched, raw=True)
    figures = {main["path"]: main["sha256"]}
    page = {"page_id": PID, "source": "course/fixture.md", "selector": "whole-file", "figures_sha256": figures}
    page["source_sha256"] = snapshot["sha256"]
    freeze = save(root, "freeze.json", {"pages": [{**page, "snapshot": snapshot["path"]}]})
    checks = {
        name: {"status": "pass" if name == "figure_consistency" else "not_applicable", "details": "虛構核對"}
        for name in validator.CHECKS
    }
    cited = {
        "id": "page",
        "kind": "repository_code",
        "title": "本頁原文",
        "path": "course/fixture.md",
        "sha256": hashlib.sha256(reviewed).hexdigest(),
        "version": "審閱時的工作樹",
        "inspection_note": "虛構讀取紀錄",
        "verified": True,
    }
    report = save(
        root,
        "reviews/technical.json",
        {
            "schema_version": 1,
            "review_stage": "technical",
            "lesson_id": PID,
            "source": "course/fixture.md",
            "source_sha256": hashlib.sha256(reviewed).hexdigest(),
            "verdict": "pass",
            "reviewer_task": TECHNICAL,
            "reviewer_context": "fresh",
            "figure_sha256": figures,
            "artifacts": [],
            "sources": [cited],
            "claims": [],
            "applicability": {"substantive_claims": False, "reason": "虛構頁沒有實質主張"},
            "issues": [],
            "checks": checks,
        },
    )
    assignment = {"task": TECHNICAL, "role": "technical", "wave": WAVE, "context": "fresh", "lesson_id": PID}
    assignment["dispatch_window"] = {"not_before": at(1), "not_after": at(2)}
    proof = {"mode": "review", "wave": WAVE, "report": report, "json_pointer": "", "original_report": report}
    draft = {
        "source_freeze": freeze,
        "assignments": save(root, "assignments.json", {"tasks": [assignment]}),
        "author_tasks": ["/root"],
        "review_waves": [{"id": WAVE, "seals": {}}],
        "pages": [
            {"page_id": PID, "technical": {**proof, "original_json_pointer": "", "collection_role": "technical"}}
        ],
    }
    if listed:
        patch = {**PATCH, "old": "weight grows four times", "new": "weight becomes four times as large"}
        draft["author_patches"] = patch_list(root, [patch])
    return preflight.check_draft(root, draft, {PID: page}, "intake", "technical")


def test_technical_report_and_cited_file_before_an_author_patch_pass_and_are_listed(tmp_path):
    # 這一頁沒有 reader 證據，stage technical 會在 reader 記錯；這裡只看 technical 與作者補丁的列表。
    result = technical_patched(tmp_path / "listed", True)
    assert result["pages"][PID]["technical"]["errors"] == []
    assert result["author_patches"] == {PID: ["fixture-1"]}
    result = technical_patched(tmp_path / "unlisted", False)
    assert result["pages"][PID]["technical"]["errors"] == ["report source is stale: fixture"]
