"""審閱閘門遇到遺漏、改檔與假順序必須拒絕；此處全是虛構測試資料。"""

import builtins
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("incremental_review", ROOT / "docs/review-tools/incremental_revision.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
REPLAY_SPEC = importlib.util.spec_from_file_location("replay_review", ROOT / "docs/review-tools/baseline_replay.py")
replay_tool = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(replay_tool)


def save(root, name, value, raw=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def session(root, monkeypatch):
    # 單位切法由獨立fixture指定；測試保存完整性，而非複製分段演算法。
    monkeypatch.setattr(review, "module", lambda *_: SimpleNamespace(units=lambda _: ["first", "second"]))
    source = save(root, "source.md", b"fictional source", raw=True)
    opening = save(
        root,
        "opening.json",
        {
            "page_id": "fixture",
            "reviewer_task": "/root/fixture_reader",
            "source_sha256": source["sha256"],
            "figures_sha256": {},
            "started_at": "2026-10-16T01:00:00+00:00",
        },
    )
    notes = []
    for i, unit in enumerate(("first", "second")):
        note = {
            "page_id": "fixture",
            "reviewer_task": "/root/fixture_reader",
            "unit_index": i,
            "unit_sha256": hashlib.sha256(unit.encode()).hexdigest(),
            "saved_at": f"2026-10-16T01:00:0{i + 1}+00:00",
            **{
                name: "fictional fixture"
                for name in (
                    "understanding",
                    "materials_and_labels",
                    "expected_change",
                    "confusion_and_quote",
                    "missing_visuals",
                )
            },
            "concept_checks": [
                {
                    name: "fictional fixture"
                    for name in (
                        "concept",
                        "identity",
                        "mechanism",
                        "need",
                        "usage",
                        "example_scope",
                        "source_quote",
                        "verdict",
                    )
                }
            ],
        }
        notes.append(save(root, f"note-{i}.json", note))
    return {"opening": opening, "source": source, "notes": notes, "figures": {}}


def test_artifact_tamper_and_escape_are_rejected(tmp_path):
    ref = save(tmp_path, "proof.txt", b"original", raw=True)
    (tmp_path / "proof.txt").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        review.artifact(tmp_path, ref)
    with pytest.raises(ValueError, match="escapes"):
        review.local(tmp_path, "../outside.txt")


def test_complete_fixture_preserves_page_and_order(tmp_path, monkeypatch):
    proof = session(tmp_path, monkeypatch)
    page, completed = review.reader_session(tmp_path, proof, "/root/fixture_reader")
    assert page == "fixture"
    assert completed.second == 2


@pytest.mark.parametrize(
    "attack", ["missing_note", "future_time", "wrong_unit", "missing_need", "wrong_owner", "missing_figure"]
)
def test_reader_trace_rejects_incomplete_or_inconsistent_evidence(tmp_path, monkeypatch, attack):
    proof = session(tmp_path, monkeypatch)
    if attack == "missing_note":
        proof["notes"].pop()
    elif attack == "missing_figure":
        opening = json.loads((tmp_path / "opening.json").read_bytes())
        opening["figures_sha256"] = {"figure.svg": "0" * 64}
        proof["opening"] = save(tmp_path, "opening.json", opening)
    else:
        note = json.loads((tmp_path / "note-1.json").read_bytes())
        if attack == "future_time":
            note["saved_at"] = "2026-10-16T00:59:59+00:00"
        elif attack == "wrong_unit":
            note["unit_index"] = 0
        elif attack == "missing_need":
            del note["concept_checks"][0]["need"]
        elif attack == "wrong_owner":
            note["reviewer_task"] = "/root/other_fixture"
        proof["notes"][1] = save(tmp_path, "note-1.json", note)
    with pytest.raises(ValueError):
        review.reader_session(tmp_path, proof, "/root/fixture_reader")


def test_issue_inventory_includes_nested_structured_notes_without_counting_id_links():
    value = {
        "issues": [{"id": "one"}],
        "pages": [{"issues": ["one"]}],
        "note": {"concept_checks": [{"issues": [{"id": "two"}]}]},
    }
    assert review.reported_issues(value) == ["/issues/0", "/note/concept_checks/0/issues/0"]


def test_original_judgment_must_belong_to_same_reader_and_page():
    report = {"reviewer_task": "/root/fixture_reader", "pages": [{"page_id": "fixture", "source_sha256": "f" * 64}]}
    assert review.page_member(report, "/pages/0", "fixture", "/root/fixture_reader")["page_id"] == "fixture"
    with pytest.raises(ValueError, match="reviewer mismatch"):
        review.page_member(report, "/pages/0", "fixture", "/root/author")
    with pytest.raises(ValueError, match="page mismatch"):
        review.page_member(report, "/pages/0", "different", "/root/fixture_reader")


def collection_fixture(root):
    seals = {}
    for i, role in enumerate(review.ROLES):
        seals[role] = save(
            root,
            role + "-seal.json",
            {
                "role": role,
                "status": "completed",
                "started_at": f"2026-10-16T0{i + 1}:00:00+00:00",
                "completed_at": f"2026-10-16T0{i + 1}:30:00+00:00",
                "report_sha256s": [role + "-report"],
            },
        )
    return [{"id": "fixture", "seals": seals}]


def test_late_section_or_tool_report_uses_actual_later_seal_even_in_technical_check(tmp_path):
    waves = collection_fixture(tmp_path)
    sealed = review.review_seals(tmp_path, waves, review.ROLES[:2])
    proof = {"wave": "fixture", "collection_role": "continuity"}
    review.require_collected(proof, "technical", sealed, [{"sha256": "continuity-report"}])
    with pytest.raises(ValueError, match="not collected"):
        review.require_collected(proof, "technical", sealed, [{"sha256": "uncollected-report"}])
    # 不得把後補報告謊稱已在較早technical階段收到。
    with pytest.raises(ValueError, match="not collected"):
        review.require_collected({"wave": "fixture"}, "technical", sealed, [{"sha256": "continuity-report"}])
    with pytest.raises(ValueError, match="invalid followup"):
        review.require_collected({"wave": "fixture", "collection_role": "reader"}, "technical", sealed, [])


def test_provided_later_seal_is_not_skipped_when_checking_earlier_roles(tmp_path):
    waves = collection_fixture(tmp_path)
    seal = json.loads((tmp_path / "continuity-seal.json").read_bytes())
    seal["started_at"] = "2026-10-16T02:15:00+00:00"
    waves[0]["seals"]["continuity"] = save(tmp_path, "continuity-seal.json", seal)
    with pytest.raises(ValueError, match="overlapped"):
        review.review_seals(tmp_path, waves, review.ROLES[:2])


def test_reader_only_manifest_does_not_invent_unprovided_future_stages(tmp_path):
    waves = collection_fixture(tmp_path)
    waves[0]["seals"] = {"reader": waves[0]["seals"]["reader"]}
    sealed = review.review_seals(tmp_path, waves, ("reader",))
    assert set(sealed["fixture"]) == {"reader"}
    with pytest.raises(ValueError, match="missing actual"):
        review.require_collected({"wave": "fixture", "collection_role": "continuity"}, "reader", sealed, [])


# 以下測重放舊輪次與補充還原清單：迷你repo放最小的重放基準結構，只把composite_review換成小型驗證器替身；
# 兩張舊雜湊表照樣由replay自己的窄用途讀取取得。

FREEZE = "docs/course-publish-20261008/current-source-freeze.json"
OVERLAY = "docs/course-revision-20261016/baseline-overlay.json"
REVIEW_MANIFEST = "docs/course-publish-20261008/review-manifest.json"
FREEZE01_ORIGINAL = "/workspace/old/freeze-01/manifest.json"
STOP_AFTER_REPLAY = "baseline identity is not the replayed source freeze"
CURRENT = {
    "scripts/fetch.py": b"fetch current\n",
    "docs/v2.json": b"v2 current\n",
    "course/page.md": b"page current\n",
    "course/figure.svg": b"figure current\n",
    "notebooks/page.ipynb": b"notebook current\n",
    ".agents/skills/clear-tutorial/SKILL.md": b"criteria current\n",
    "README_en.md": b"readme current\n",
    "pyproject.toml": b"pyproject current\n",
    "both.txt": b"both current\n",
    "shared.txt": b"shared current\n",
    "untouched.txt": b"untouched current\n",
}
ARCHIVED = {
    "old/fetch.py": b"fetch archived\n",
    "old/v2.json": b"v2 archived\n",
    "old/page.md": b"page archived\n",
    "old/figure.svg": b"figure archived\n",
    "old/page.ipynb": b"notebook archived\n",
    "old/SKILL.md": b"criteria archived\n",
    "old/README_en.md": b"readme overlay\n",
    "old/pyproject.toml": b"pyproject overlay\n",
    "old/pyproject-reviewed.toml": b"pyproject reviewed\n",
    "old/both.txt": b"both reviewed\n",
    "old/shared.txt": b"shared reviewed\n",
    "old/vendor-data.txt": b"vendor reviewed\n",
}
FREEZE_ENTRIES = {
    "source_files": {"course/page.md": ("course/page.md", "old/page.md")},
    "criteria": {"SKILL.md": (".agents/skills/clear-tutorial/SKILL.md", "old/SKILL.md")},
    "figures": {"course/figure.svg": ("course/figure.svg", "old/figure.svg")},
    "notebooks": {"notebooks/page.ipynb": ("notebooks/page.ipynb", "old/page.ipynb")},
}
OVERLAY_ENTRIES = [("README_en.md", "old/README_en.md"), ("pyproject.toml", "old/pyproject.toml")]
EXISTING_RESTORES = {
    live: ARCHIVED[path]
    for live, path in [*(p for kind in FREEZE_ENTRIES.values() for p in kind.values()), *OVERLAY_ENTRIES]
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exactly(message):
    return "^" + re.escape(message) + "$"


def restore(live_path, path, data=None):
    return {"live_path": live_path, "path": path, "sha256": digest(ARCHIVED[path] if data is None else data)}


FETCH = restore("scripts/fetch.py", "old/fetch.py")
V2 = restore("docs/v2.json", "old/v2.json")
# 兩張表都有both.txt卻記了不同的值；shared.txt則兩邊一致。
IMPLEMENTATION = {
    "scripts/fetch.py": FETCH["sha256"],
    "pyproject.toml": digest(ARCHIVED["old/pyproject-reviewed.toml"]),
    "both.txt": digest(ARCHIVED["old/both.txt"]),
    "shared.txt": digest(ARCHIVED["old/shared.txt"]),
    "vendor/linked/data.txt": digest(ARCHIVED["old/vendor-data.txt"]),
}
INPUTS = {
    "docs/v2.json": V2["sha256"],
    "both.txt": digest(b"both reviewed elsewhere\n"),
    "shared.txt": digest(ARCHIVED["old/shared.txt"]),
}
VALIDATOR = '''"""小型驗證器替身：核對重放傳來的參數，回報暫存副本每個檔的內容，以及是否仍和原檔是同一個檔。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit_manifest(root, manifest_path, stage="all"):
    root = Path(root)
    failures = []
    if root.parent != ROOT / "outputs/review-baseline-replay" or not root.name.startswith("immutable-baseline-"):
        failures.append("unexpected replay root: " + str(root))
    if Path(manifest_path) != root / "docs/course-publish-20261008/review-manifest.json":
        failures.append("unexpected manifest: " + str(manifest_path))
    if stage != "all":
        failures.append("unexpected stage: " + str(stage))
    mode = ROOT / "validator-mode.txt"
    mode = mode.read_text() if mode.exists() else "pass"
    if mode == "raise":
        raise RuntimeError("stand-in validator raised")
    if mode == "fail":
        failures.append("stand-in validator rejected")
    observed = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            name = path.relative_to(root).as_posix()
            live = ROOT / name
            same = live.exists() and os.path.samefile(path, live)
            observed[name] = {"bytes": path.read_bytes().decode(), "samefile": same}
    return {"status": "failed" if failures else "passed_metadata_checks", "failures": failures, "observed": observed}
'''
RECORDING_REPLAY = '''"""重放替身：記下收到的參數再回報失敗，讓審核停在重放這一步。"""

import json
from pathlib import Path


def replay(root, supplements=()):
    with (Path(root) / "replay-calls.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"root": str(root), "supplements": list(supplements)}) + "\\n")
    return {"failures": ["recorded replay call"]}
'''
TOOL_FILES = (
    "docs/review-tools/incremental_revision.py",
    "docs/review-tools/baseline_replay.py",
    "docs/review-tools/phase7_review.py",
    "docs/course-revision-20261016/tools/reader.py",
    "docs/course-revision-20261016/tools/freeze_sources.py",
    "tests/test_incremental_revision_review.py",
)


def git(repo, *args):
    """在暫存repo執行git；不讀使用者與系統設定，也不複製template（比照test_course_commit_pins的run_git）。"""
    template = repo.parent / "empty-git-template"
    template.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TEMPLATE_DIR": str(template),
    }
    done = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=environment)
    return done.stdout


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    """迷你重放基準：現行檔、舊副本、凍結檔、overlay、通到兩張舊雜湊表的引用鏈與小型驗證器替身。"""
    if shutil.which("git") is None:
        pytest.skip("找不到 git 指令")
    repo = tmp_path / "repo"
    for name, data in {**CURRENT, **ARCHIVED}.items():
        save(repo, name, data, raw=True)
    freeze = {kind: {name: restore(*pair) for name, pair in rows.items()} for kind, rows in FREEZE_ENTRIES.items()}
    freeze_ref = save(repo, FREEZE, freeze)
    save(repo, OVERLAY, {"files": [restore(*pair) for pair in OVERLAY_ENTRIES]})
    freeze01 = save(repo, "old/freeze01.json", {"implementation_sha256": IMPLEMENTATION})
    index = {
        "schema_version": 1,
        "artifacts": [
            {"original_path": FREEZE01_ORIGINAL, **freeze01},
            # 無關項目指到已改過的現行檔：像composite那樣驗整份index一定失敗，窄用途讀取不能碰它。
            {"original_path": "/workspace/old/README_en.md", "path": "README_en.md", "sha256": digest(b"readme old\n")},
        ],
    }
    catalog = {"manifest_references": {"freeze01": {"path": FREEZE01_ORIGINAL, "sha256": freeze01["sha256"]}}}
    policy = {
        "archive_index": save(repo, "old/archive-index.json", index),
        "catalog": save(repo, "old/catalog.json", catalog),
        "current_technical_inputs": save(repo, "old/inputs.json", {"files_sha256": INPUTS}),
        "source_freeze": freeze_ref,
    }
    save(repo, REVIEW_MANIFEST, policy)
    save(repo, "docs/review-tools/composite_review.py", VALIDATOR.encode(), raw=True)
    save(
        repo,
        "docs/review-tools/baseline_replay.py",
        (ROOT / "docs/review-tools/baseline_replay.py").read_bytes(),
        raw=True,
    )
    outside = tmp_path / "outside"
    save(outside, "data.txt", b"outside sentinel\n", raw=True)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    return SimpleNamespace(repo=repo, outside=outside, track=lambda: git(repo, "add", "-A"))


@pytest.fixture
def kept_trees(monkeypatch):
    """讓重放的暫存目錄在失敗後留下，才看得到失敗前有沒有建樹或寫入。"""
    created = []

    class Kept:
        def __init__(self, **options):
            self.name = tempfile.mkdtemp(**options)
            created.append(Path(self.name))

        def __enter__(self):
            return self.name

        def __exit__(self, *error):
            return False

    monkeypatch.setattr(replay_tool, "tempfile", SimpleNamespace(TemporaryDirectory=Kept))
    return created


def files_state(repo):
    """repo內每個普通檔的位元組與權限位元，不含.git、重放暫存區與載入驗證器時產生的bytecode。"""
    state = {}
    for path in sorted(repo.rglob("*")):
        name = path.relative_to(repo).as_posix()
        if name.startswith((".git/", "outputs/")) or "__pycache__" in path.parts:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        state[name] = (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
    return state


def listing(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes() if path.is_file() else None for path in folder.rglob("*")
    }


def assert_no_replay_tree(baseline, kept):
    """A段失敗時還沒建暫存目錄，連重放暫存根都沒建。"""
    assert kept == []
    assert not (baseline.repo / "outputs").exists()


def require_symlinks(folder):
    probe = folder / "symlink-probe"
    try:
        probe.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("這個環境不能建 symlink")
    probe.unlink()


def require_tracked_symlink(baseline, name):
    if git(baseline.repo, "ls-files", "-s", "--", name).split()[:1] != [b"120000"]:
        pytest.skip("git 沒有把它記成 symlink")


def add_overlay(baseline, entry):
    overlay = json.loads((baseline.repo / OVERLAY).read_bytes())
    overlay["files"].append(entry)
    save(baseline.repo, OVERLAY, overlay)
    return f"{OVERLAY} files[{len(overlay['files']) - 1}]"


def place(baseline, listed_in, entry):
    """把entry放進補充還原清單或overlay；回傳交給replay的補充，以及錯誤訊息裡的位置。"""
    if listed_in == "supplement":
        return [entry], "補充還原清單 files[0]"
    return [], add_overlay(baseline, entry)


def rebind(repo, key, reference):
    policy = json.loads((repo / REVIEW_MANIFEST).read_bytes())
    policy[key] = reference
    save(repo, REVIEW_MANIFEST, policy)


def candidate(repo, supplement_list, reference=lambda listed: listed):
    """候選manifest帶補充還原清單；baseline_freeze故意指到別的檔，重放一通過就停在下一道檢查。"""
    raw = isinstance(supplement_list, bytes)
    listed = reference(save(repo, "round/baseline-supplements.json", supplement_list, raw=raw))
    inputs = (repo / "old/inputs.json").read_bytes()
    manifest = {
        "review_policy": review.POLICY,
        "schema_version": 2,
        "baseline_supplements": listed,
        "baseline_freeze": {"path": "old/inputs.json", "sha256": digest(inputs)},
    }
    save(repo, "round/manifest.json", manifest)
    return repo / "round/manifest.json"


def test_replay_restores_both_existing_lists_and_supplements_only_when_given(baseline):
    before = files_state(baseline.repo)
    plain = replay_tool.replay(baseline.repo)
    supplemented = replay_tool.replay(baseline.repo, [FETCH, V2])
    for result in (plain, supplemented):
        assert result["failures"] == []
        for live, data in EXISTING_RESTORES.items():
            assert result["observed"][live]["bytes"] == data.decode()
        assert result["observed"]["untouched.txt"]["bytes"] == CURRENT["untouched.txt"].decode()
    assert plain["observed"]["scripts/fetch.py"]["bytes"] == CURRENT["scripts/fetch.py"].decode()
    assert plain["observed"]["docs/v2.json"]["bytes"] == CURRENT["docs/v2.json"].decode()
    assert supplemented["observed"]["scripts/fetch.py"]["bytes"] == ARCHIVED["old/fetch.py"].decode()
    assert supplemented["observed"]["docs/v2.json"]["bytes"] == ARCHIVED["old/v2.json"].decode()
    assert files_state(baseline.repo) == before


def test_hard_link_branch_keeps_untouched_files_shared_and_detaches_restored_ones(baseline):
    source = baseline.repo.parent / "link-probe"
    source.write_bytes(b"probe")
    try:
        os.link(source, baseline.repo.parent / "link-probe-2")
    except OSError:
        pytest.skip("這個檔案系統不能建 hard link")
    before = files_state(baseline.repo)
    result = replay_tool.replay(baseline.repo, [FETCH, V2])
    assert result["failures"] == []
    assert result["observed"]["untouched.txt"] == {"bytes": CURRENT["untouched.txt"].decode(), "samefile": True}
    restored = {
        **EXISTING_RESTORES,
        "scripts/fetch.py": ARCHIVED["old/fetch.py"],
        "docs/v2.json": ARCHIVED["old/v2.json"],
    }
    for live, data in restored.items():
        assert result["observed"][live] == {"bytes": data.decode(), "samefile": False}
    assert files_state(baseline.repo) == before


def test_copy_branch_still_detaches_restored_files_and_keeps_originals(baseline, monkeypatch):
    def refuse(*_):
        raise OSError("hard link refused by test")

    monkeypatch.setattr(replay_tool.os, "link", refuse)
    before = files_state(baseline.repo)
    result = replay_tool.replay(baseline.repo, [FETCH, V2])
    assert result["failures"] == []
    assert result["observed"]["untouched.txt"] == {"bytes": CURRENT["untouched.txt"].decode(), "samefile": False}
    restored = {
        **EXISTING_RESTORES,
        "scripts/fetch.py": ARCHIVED["old/fetch.py"],
        "docs/v2.json": ARCHIVED["old/v2.json"],
    }
    for live, data in restored.items():
        assert result["observed"][live] == {"bytes": data.decode(), "samefile": False}
    assert files_state(baseline.repo) == before


@pytest.mark.parametrize("mode", ["fail", "raise"])
def test_validator_failure_or_exception_is_not_turned_into_success(baseline, mode):
    (baseline.repo / "validator-mode.txt").write_text(mode)
    if mode == "fail":
        result = replay_tool.replay(baseline.repo, [FETCH, V2])
        assert result["status"] == "failed"
        assert result["failures"] == ["stand-in validator rejected"]
    else:
        with pytest.raises(RuntimeError, match="stand-in validator raised"):
            replay_tool.replay(baseline.repo, [FETCH, V2])


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        ("/tmp/x", "以 / 開頭"),
        ("C:/x", "以磁碟代號開頭"),
        ("C:x", "以磁碟代號開頭"),
        ("docs\\x.json", "含反斜線"),
        ("", "含空段"),
        ("docs//x.json", "含空段"),
        ("docs/", "含空段"),
        (".", "含 . 段"),
        ("docs/./x.json", "含 . 段"),
        ("..", "含 .. 段"),
        ("docs/../x.json", "含 .. 段"),
        (None, "不是字串"),
    ],
)
def test_restore_path_check_names_the_rule_each_non_posix_form_breaks(value, rule):
    with pytest.raises(ValueError, match=exactly(f"標籤 {rule}：{value!r}")):
        replay_tool.posix_relative(value, "標籤")


@pytest.mark.parametrize(
    "value", ["README_en.md", ".agents/skills/clear-tutorial/SKILL.md", "docs/..notes/.x/a b.json"]
)
def test_restore_path_check_accepts_plain_posix_relative_paths(value):
    assert replay_tool.posix_relative(value, "標籤") == value


@pytest.mark.parametrize("listed_in", ["supplement", "overlay"])
def test_valid_destination_with_source_escaping_through_dotdot_is_rejected_before_any_tree(
    baseline, kept_trees, listed_in
):
    # 跳出去的檔位元組完全正確，只做字串前綴檢查擋不住。
    save(baseline.outside, "fetch.py", ARCHIVED["old/fetch.py"], raw=True)
    supplements, where = place(baseline, listed_in, {**FETCH, "path": "old/../../outside/fetch.py"})
    with pytest.raises(ValueError, match=exactly(f"{where} 的 path 含 .. 段：'old/../../outside/fetch.py'")):
        replay_tool.replay(baseline.repo, supplements)
    assert_no_replay_tree(baseline, kept_trees)


def test_supplement_matching_old_table_is_rejected_when_its_copy_bytes_changed(baseline, kept_trees):
    (baseline.repo / "old/fetch.py").write_bytes(b"fetch archived, then edited\n")
    with pytest.raises(ValueError, match=exactly("補充還原清單 files[0] 的來源位元組與 sha256 不符：'old/fetch.py'")):
        replay_tool.replay(baseline.repo, [FETCH])
    assert_no_replay_tree(baseline, kept_trees)


def test_self_consistent_supplement_still_has_to_equal_the_old_table(baseline):
    # 位元組與自填的sha256一致，清單的外層SHA也跟著更新，檢查才走得到舊身分比較。
    rewritten = b"fetch rewritten\n"
    save(baseline.repo, "old/fetch-rewritten.py", rewritten, raw=True)
    entry = restore("scripts/fetch.py", "old/fetch-rewritten.py", rewritten)
    path = candidate(baseline.repo, {"schema_version": 1, "files": [entry], "reason": "fictional"})
    assert review.audit_manifest(baseline.repo, path, "all")["failures"] == [
        "補充還原清單 files[0] 的 sha256 與舊雜湊表 implementation_sha256 記的值不同：'scripts/fetch.py'"
    ]


@pytest.mark.parametrize(
    ("entry", "rule"),
    [
        (restore(REVIEW_MANIFEST, "old/fetch.py"), f"的 live_path 不是舊雜湊表的 key：{REVIEW_MANIFEST!r}"),
        (
            restore("scripts/fetch.py", "old/v2.json"),
            "的 sha256 與舊雜湊表 implementation_sha256 記的值不同：'scripts/fetch.py'",
        ),
    ],
    ids=["old_manifest_as_destination", "digest_recorded_for_another_key"],
)
def test_supplement_cannot_borrow_a_digest_the_old_tables_record_for_another_key(baseline, kept_trees, entry, rule):
    with pytest.raises(ValueError, match=exactly("補充還原清單 files[0] " + rule)):
        replay_tool.replay(baseline.repo, [entry])
    assert_no_replay_tree(baseline, kept_trees)


def test_key_recorded_identically_in_both_old_tables_is_accepted(baseline):
    result = replay_tool.replay(baseline.repo, [restore("shared.txt", "old/shared.txt")])
    assert result["failures"] == []
    assert result["observed"]["shared.txt"]["bytes"] == ARCHIVED["old/shared.txt"].decode()


def test_key_in_both_old_tables_has_to_equal_both_values(baseline, kept_trees):
    # 這筆等於implementation_sha256記的值，卻不等於files_sha256記的值。
    rule = "補充還原清單 files[0] 的 sha256 與舊雜湊表 files_sha256 記的值不同：'both.txt'"
    with pytest.raises(ValueError, match=exactly(rule)):
        replay_tool.replay(baseline.repo, [restore("both.txt", "old/both.txt")])
    assert_no_replay_tree(baseline, kept_trees)


@pytest.mark.parametrize("case", ["supplement_repeats_overlay", "supplement_repeats_itself", "existing_case_alias"])
def test_restore_destinations_are_unique_across_all_three_lists_ignoring_case(baseline, kept_trees, case):
    if case == "supplement_repeats_overlay":
        # 舊表有pyproject.toml，這筆補充本身合法；但overlay已經還原它。
        supplements = [restore("pyproject.toml", "old/pyproject-reviewed.toml")]
        message = f"補充還原清單 files[0] 的 live_path 與 {OVERLAY} files[1] 重複（不分大小寫）：'pyproject.toml'"
    elif case == "supplement_repeats_itself":
        supplements = [FETCH, dict(FETCH)]
        message = "補充還原清單 files[1] 的 live_path 與 補充還原清單 files[0] 重複（不分大小寫）：'scripts/fetch.py'"
    else:
        # 既有兩份清單不套舊身分條件，才能合成只差大小寫的兩個目的。
        save(baseline.repo, "old/alias-upper.md", b"alias upper\n", raw=True)
        save(baseline.repo, "old/alias-lower.md", b"alias lower\n", raw=True)
        freeze = json.loads((baseline.repo / FREEZE).read_bytes())
        freeze["source_files"]["course/Alias.md"] = restore("course/Alias.md", "old/alias-upper.md", b"alias upper\n")
        save(baseline.repo, FREEZE, freeze)
        where = add_overlay(baseline, restore("course/alias.md", "old/alias-lower.md", b"alias lower\n"))
        supplements = []
        message = (
            f"{where} 的 live_path 與 {FREEZE} source_files['course/Alias.md'] 重複（不分大小寫）：'course/alias.md'"
        )
    with pytest.raises(ValueError, match=exactly(message)):
        replay_tool.replay(baseline.repo, supplements)
    assert_no_replay_tree(baseline, kept_trees)


@pytest.mark.parametrize("points_to", ["outside", "inside"])
def test_source_reached_through_a_symlink_is_rejected_before_any_tree(baseline, kept_trees, points_to):
    require_symlinks(baseline.repo.parent)
    save(baseline.outside, "fetch.py", ARCHIVED["old/fetch.py"], raw=True)
    target = baseline.outside if points_to == "outside" else baseline.repo / "old"
    (baseline.repo / "linked-old").symlink_to(target, target_is_directory=True)
    outside = listing(baseline.outside)
    with pytest.raises(ValueError, match=exactly("補充還原清單 files[0] 的 path 經過 symlink：'linked-old'")):
        replay_tool.replay(baseline.repo, [{**FETCH, "path": "linked-old/fetch.py"}])
    assert listing(baseline.outside) == outside
    assert_no_replay_tree(baseline, kept_trees)


@pytest.mark.parametrize("listed_in", ["supplement", "overlay"])
def test_symlinked_parent_inside_replay_tree_is_rejected_before_any_restore_write(baseline, kept_trees, listed_in):
    # 暫存樹裡的上層symlink由repo內追蹤的目錄symlink照原樣重建而來。
    require_symlinks(baseline.repo.parent)
    (baseline.repo / "vendor").mkdir()
    (baseline.repo / "vendor/linked").symlink_to(baseline.outside, target_is_directory=True)
    baseline.track()
    require_tracked_symlink(baseline, "vendor/linked")
    supplements, where = place(baseline, listed_in, restore("vendor/linked/data.txt", "old/vendor-data.txt"))
    outside = listing(baseline.outside)
    with pytest.raises(ValueError, match=exactly(f"{where} 的 live_path 上層 經過 symlink：'vendor/linked'")):
        replay_tool.replay(baseline.repo, supplements)
    assert listing(baseline.outside) == outside
    # 全部檢查做完才寫：排在前面的凍結檔與overlay項目也還沒換成舊位元組。
    (tree,) = kept_trees
    for live in EXISTING_RESTORES:
        assert (tree / live).read_bytes() == CURRENT[live]


@pytest.mark.parametrize("linked", ["outputs", "outputs/review-baseline-replay"])
def test_symlinked_replay_cache_is_rejected_before_any_tree(baseline, kept_trees, linked):
    require_symlinks(baseline.repo.parent)
    (baseline.outside / "cache").mkdir()
    (baseline.repo / linked).parent.mkdir(parents=True, exist_ok=True)
    (baseline.repo / linked).symlink_to(baseline.outside / "cache", target_is_directory=True)
    outside = listing(baseline.outside)
    with pytest.raises(ValueError, match=exactly(f"重放暫存根 經過 symlink：{linked!r}")):
        replay_tool.replay(baseline.repo, [FETCH, V2])
    assert listing(baseline.outside) == outside
    assert kept_trees == []


def test_redirect_hidden_from_is_symlink_is_still_caught_by_its_real_location(tmp_path, monkeypatch):
    # 模擬Windows junction這類is_symlink回報不到的重導向。
    require_symlinks(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "redirect").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(replay_tool.Path, "is_symlink", lambda _: False)
    with pytest.raises(ValueError, match=exactly("標籤 實際位置離開 暫存目錄：'redirect/x.txt'")):
        replay_tool.no_redirect(base, "redirect/x.txt", "標籤", "暫存目錄")
    assert replay_tool.no_redirect(base, "plain/x.txt", "標籤", "暫存目錄") == base / "plain/x.txt"


@pytest.fixture
def opened_files(monkeypatch):
    """記下經內建open或io.open以路徑開啟的每個檔的實際位置，才看得到拒絕之前有沒有讀過外部檔。"""
    opened = []
    original = io.open

    def recording(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            opened.append(Path(os.path.realpath(os.fsdecode(file))))
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", recording)
    monkeypatch.setattr(builtins, "open", recording)
    # Path.read_bytes必須經過這裡；哪天不經過，記錄會是空的，用到它的斷言就白白通過。
    Path(__file__).read_bytes()
    assert opened == [Path(os.path.realpath(__file__))]
    opened.clear()
    return opened


@pytest.mark.parametrize("linked", ["list", "parent"])
@pytest.mark.parametrize("name", [FREEZE, OVERLAY], ids=["freeze", "overlay"])
def test_existing_restore_list_behind_a_symlink_is_rejected_before_it_is_opened(
    baseline, kept_trees, opened_files, name, linked
):
    # 清單本身或它的上層目錄搬到repo外，再用symlink接回；外部檔是同一份合法JSON，照舊直接讀的實作會接受它。
    require_symlinks(baseline.repo.parent)
    redirected = name if linked == "list" else name.rsplit("/", 1)[0]
    moved = baseline.outside / "moved"
    (baseline.repo / redirected).rename(moved)
    (baseline.repo / redirected).symlink_to(moved, target_is_directory=linked == "parent")
    with pytest.raises(ValueError, match=exactly(f"還原清單 {name} 經過 symlink：{redirected!r}")):
        replay_tool.replay(baseline.repo)
    outside = Path(os.path.realpath(baseline.outside))
    assert [path for path in opened_files if path.is_relative_to(outside)] == []
    assert_no_replay_tree(baseline, kept_trees)


@pytest.mark.parametrize("branch", ["hardlink", "copy"])
@pytest.mark.parametrize(
    ("redirected", "first"), [("scripts", "scripts/fetch.py"), ("lib/nested", "lib/nested/inner.txt")]
)
def test_tracked_directory_redirected_outside_is_rejected_before_any_source_is_linked_or_copied(
    baseline, monkeypatch, redirected, first, branch
):
    # 追蹤的普通目錄搬到repo外再用symlink接回，外部內容與原檔相同。
    # lib/a.txt排在lib/nested/inner.txt之前：已核過的上層lib不能讓下一層的lib/nested跟著免核。
    require_symlinks(baseline.repo.parent)
    save(baseline.repo, "lib/a.txt", b"lib top\n", raw=True)
    save(baseline.repo, "lib/nested/inner.txt", b"lib nested\n", raw=True)
    baseline.track()
    moved = baseline.outside / "moved"
    (baseline.repo / redirected).rename(moved)
    (baseline.repo / redirected).symlink_to(moved, target_is_directory=True)
    sources = []
    link, copy = replay_tool.os.link, replay_tool.shutil.copy2

    def recording_link(source, destination, *args, **kwargs):
        sources.append(source)
        if branch == "copy":
            raise OSError("hard link refused by test")
        return link(source, destination, *args, **kwargs)

    def recording_copy(source, destination, *args, **kwargs):
        sources.append(source)
        return copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(replay_tool.os, "link", recording_link)
    monkeypatch.setattr(replay_tool.shutil, "copy2", recording_copy)
    with pytest.raises(ValueError, match=exactly(f"追蹤檔 {first!r} 的上層 經過 symlink：{redirected!r}")):
        replay_tool.replay(baseline.repo)
    repo = Path(os.path.realpath(baseline.repo))
    assert [source for source in sources if not Path(os.path.realpath(source)).is_relative_to(repo)] == []


# 從links/指回上一層AGENTS.md的相對target。Windows不把相對target裡的/當分隔符，那樣建的連結指不到檔，
# 所以用os.path.join組成各平台自己的分隔符。
AGENTS_FROM_LINKS = os.path.join("..", "AGENTS.md")


class RecordingModule:
    """代替重放模組裡的os或shutil：列出的函式先記下第一個引數再照原樣呼叫，其餘屬性照舊取自原模組。

    只換重放模組自己用的名稱；全域換掉os.readlink，連os.path.realpath內部的呼叫也會被記到。
    """

    def __init__(self, module, calls, *names):
        self.module = module
        for name in names:
            setattr(self, name, self.recording(getattr(module, name), calls))

    @staticmethod
    def recording(function, calls):
        def call(path, *args, **kwargs):
            calls.append(path)
            return function(path, *args, **kwargs)

        return call

    def __getattr__(self, name):
        return getattr(self.module, name)


@pytest.mark.parametrize("mirrored", [True, False], ids=["target_mirrored_outside", "target_missing_outside"])
def test_tracked_symlink_alone_in_a_redirected_folder_is_rejected_by_its_parent_check(
    baseline, kept_trees, monkeypatch, mirrored
):
    # links/裡只有一個追蹤中的symlink，沒有普通檔先替它觸發上層檢查：葉節點是symlink的名稱也要核上層。
    # links/整個搬到repo外再用symlink接回後，指回上一層AGENTS.md的相對連結改指向外部那一層；
    # 一個案例在那裡放同內容的AGENTS.md，另一個案例沒有，連結指不到檔，只核存在的名稱的實作會跳過它。
    require_symlinks(baseline.repo.parent)
    save(baseline.repo, "AGENTS.md", b"agents current\n", raw=True)
    (baseline.repo / "links").mkdir()
    (baseline.repo / "links/CLAUDE.md").symlink_to(AGENTS_FROM_LINKS)
    baseline.track()
    require_tracked_symlink(baseline, "links/CLAUDE.md")
    if mirrored:
        save(baseline.outside, "AGENTS.md", b"agents current\n", raw=True)
    moved = baseline.outside / "moved"
    (baseline.repo / "links").rename(moved)
    (baseline.repo / "links").symlink_to(moved, target_is_directory=True)
    calls = []
    monkeypatch.setattr(replay_tool, "os", RecordingModule(os, calls, "link", "readlink"))
    monkeypatch.setattr(replay_tool, "shutil", RecordingModule(shutil, calls, "copy2"))
    with pytest.raises(ValueError, match=exactly("追蹤檔 'links/CLAUDE.md' 的上層 經過 symlink：'links'")):
        replay_tool.replay(baseline.repo)
    # 看上層的實際位置：os.readlink不跟著葉節點走，link與copy收到的葉節點不是symlink。
    repo = Path(os.path.realpath(baseline.repo))
    assert [path for path in calls if not Path(os.path.realpath(Path(path).parent)).is_relative_to(repo)] == []
    assert_no_replay_tree(baseline, kept_trees)


@pytest.mark.parametrize(
    ("link", "target"),
    [("CLAUDE.md", "AGENTS.md"), ("links/CLAUDE.md", AGENTS_FROM_LINKS)],
    ids=["root", "alone_in_folder"],
)
def test_tracked_file_symlink_is_still_copied_as_a_link_and_the_replay_passes(baseline, kept_trees, link, target):
    # 比照根目錄的CLAUDE.md：追蹤中的檔案symlink指向repo內的檔，暫存樹裡照樣是同一個連結。
    # 根目錄的CLAUDE.md排在AGENTS.md之後；連葉節點一起核、每個上層又只核第一個檔的實作會跳過它。
    # links/裡只有這個連結，那種實作一定會核到它。
    require_symlinks(baseline.repo.parent)
    save(baseline.repo, "AGENTS.md", b"agents current\n", raw=True)
    (baseline.repo / link).parent.mkdir(parents=True, exist_ok=True)
    (baseline.repo / link).symlink_to(target)
    baseline.track()
    require_tracked_symlink(baseline, link)
    result = replay_tool.replay(baseline.repo)
    assert result["failures"] == []
    (tree,) = kept_trees
    assert (tree / link).is_symlink()
    assert os.readlink(tree / link) == target


ALIAS_COUNT = "重放基準 archive_index 裡 original_path 等於 freeze01 引用的項目有 {} 筆，必須恰好一筆：" + repr(
    FREEZE01_ORIGINAL
)


def test_old_table_reader_ignores_unrelated_archive_entries(baseline):
    assert replay_tool.preserved_identities(baseline.repo) == {
        "implementation_sha256": IMPLEMENTATION,
        "files_sha256": INPUTS,
    }


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("old/catalog.json", "重放基準 catalog 的 SHA-256 與引用不符：'old/catalog.json'"),
        ("old/archive-index.json", "重放基準 archive_index 的 SHA-256 與引用不符：'old/archive-index.json'"),
        ("old/inputs.json", "重放基準 current_technical_inputs 的 SHA-256 與引用不符：'old/inputs.json'"),
        ("old/freeze01.json", "重放基準 freeze01 的 SHA-256 與引用不符：'old/freeze01.json'"),
        ("alias_missing", ALIAS_COUNT.format(0)),
        ("alias_twice", ALIAS_COUNT.format(2)),
        ("alias_digest", f"freeze01 的 catalog 引用與 archive_index 那筆記的 SHA-256 不同：{FREEZE01_ORIGINAL!r}"),
    ],
)
def test_old_table_reader_rejects_a_broken_reference_chain(baseline, damage, message):
    repo = baseline.repo
    if damage.startswith("old/"):
        (repo / damage).write_bytes((repo / damage).read_bytes() + b" ")
    else:
        index = json.loads((repo / "old/archive-index.json").read_bytes())
        alias = index["artifacts"][0]
        if damage == "alias_missing":
            index["artifacts"] = index["artifacts"][1:]
        elif damage == "alias_twice":
            index["artifacts"].append(dict(alias))
        else:
            alias["sha256"] = digest(b"another manifest")
        rebind(repo, "archive_index", save(repo, "old/archive-index.json", index))
    with pytest.raises(ValueError, match=exactly(message)):
        replay_tool.preserved_identities(repo)


LISTED = {"schema_version": 1, "files": [FETCH, V2], "reason": "fictional"}
LIST_PATH = "round/baseline-supplements.json"
ENTRY_RULE = "補充還原清單 " + LIST_PATH + " 的 files[{}] 必須是只含 live_path、path、sha256 三個字串欄位的物件"


@pytest.mark.parametrize(
    ("supplement_list", "reference", "message"),
    [
        (LISTED, lambda ref: {**ref, "sha256": "0" * 64}, "artifact changed: " + LIST_PATH),
        (LISTED, lambda ref: None, "missing artifact reference"),
        (
            LISTED,
            lambda ref: {**ref, "path": "round/absent.json"},
            "補充還原清單讀不到（FileNotFoundError）：round/absent.json",
        ),
        (b"not json", None, "補充還原清單不是合法的 JSON：" + LIST_PATH),
        ([FETCH], None, "補充還原清單不是 JSON 物件：" + LIST_PATH),
        ({**LISTED, "schema_version": 2}, None, "補充還原清單的 schema_version 不是 1：" + LIST_PATH),
        ({**LISTED, "schema_version": True}, None, "補充還原清單的 schema_version 不是 1：" + LIST_PATH),
        ({**LISTED, "files": {"0": FETCH}}, None, "補充還原清單的 files 不是 list：" + LIST_PATH),
        ({**LISTED, "files": ["scripts/fetch.py"]}, None, ENTRY_RULE.format(0)),
        ({**LISTED, "files": [FETCH, {**V2, "sha256": 7}]}, None, ENTRY_RULE.format(1)),
        ({**LISTED, "files": [{**FETCH, "note": "x"}]}, None, ENTRY_RULE.format(0)),
        ({**LISTED, "files": [{"live_path": "a", "path": "b"}]}, None, ENTRY_RULE.format(0)),
    ],
    ids=[
        "list_digest",
        "null_reference",
        "missing_file",
        "not_json",
        "not_object",
        "schema_version",
        "schema_version_bool",
        "files_not_list",
        "entry_not_object",
        "field_not_string",
        "extra_field",
        "missing_field",
    ],
)
def test_supplement_list_is_bound_and_well_formed_before_replay_runs(baseline, supplement_list, reference, message):
    # 型別不符也要以ValueError收進failures，不能以AttributeError等漏出audit_manifest。
    path = candidate(baseline.repo, supplement_list, *([reference] if reference else []))
    result = review.audit_manifest(baseline.repo, path, "all")
    assert result["status"] == "failed"
    assert result["failures"] == [message]
    assert not (baseline.repo / "outputs").exists()


def test_audit_moves_past_replay_once_it_accepts_the_candidate_supplements(baseline):
    path = candidate(baseline.repo, LISTED)
    assert review.audit_manifest(baseline.repo, path, "all")["failures"] == [STOP_AFTER_REPLAY]


def test_audit_hands_replay_the_supplements_named_by_the_manifest_under_audit(tmp_path):
    # active pointer與候選manifest各指一份補充清單；審哪份manifest，重放就收到哪份的entries。
    save(tmp_path, "docs/review-tools/baseline_replay.py", RECORDING_REPLAY.encode(), raw=True)
    entries, manifests = {}, {}
    for name in ("active", "candidate"):
        entries[name] = {"live_path": name + ".txt", "path": "copies/" + name + ".txt", "sha256": "0" * 64}
        listed = save(tmp_path, name + "/baseline-supplements.json", {"schema_version": 1, "files": [entries[name]]})
        manifest = {"review_policy": review.POLICY, "schema_version": 2, "baseline_supplements": listed}
        manifests[name] = save(tmp_path, name + "/manifest.json", manifest)
    plain = save(tmp_path, "plain/manifest.json", {"review_policy": review.POLICY, "schema_version": 2})
    pointer = {
        "review_policy": review.POLICY,
        "schema_version": 2,
        "manifest": manifests["active"]["path"],
        "manifest_sha256": manifests["active"]["sha256"],
    }
    save(tmp_path, "docs/course-reviews-active.json", pointer)
    stopped = ["preserved baseline failed: ['recorded replay call']"]
    assert review.audit_manifest(tmp_path, tmp_path / manifests["candidate"]["path"], "all")["failures"] == stopped
    assert review.check_active(tmp_path, pointer, "all")["failures"] == stopped
    assert review.audit_manifest(tmp_path, tmp_path / plain["path"], "reader")["failures"] == stopped
    calls = [json.loads(line) for line in (tmp_path / "replay-calls.jsonl").read_text().splitlines()]
    assert [call["supplements"] for call in calls] == [[entries["candidate"]], [entries["active"]], []]
    assert {call["root"] for call in calls} == {str(tmp_path.resolve())}


@pytest.mark.parametrize(
    "case", ["covered", "no_supplements", "missing", "wrong_digest", "list_updated", "tool_changed"]
)
def test_tool_audit_has_to_cover_the_current_supplement_list(tmp_path, case):
    tools = [save(tmp_path, name, ("fictional " + name).encode(), raw=True) for name in TOOL_FILES]
    listed = save(tmp_path, LIST_PATH, {"schema_version": 1, "files": [], "reason": "fictional"})
    covered = {"missing": [], "no_supplements": [], "wrong_digest": [{**listed, "sha256": "0" * 64}]}.get(
        case, [listed]
    )
    report = save(
        tmp_path,
        "round/tool-audit.json",
        {"reviewer_task": "/root/tool_auditor", "verdict": "pass", "checked_files": tools + covered},
    )
    manifest = {"tool_audit": {"wave": "fixture", "report": report}, "baseline_supplements": listed}
    if case == "no_supplements":
        del manifest["baseline_supplements"]
    elif case == "list_updated":
        manifest["baseline_supplements"] = save(
            tmp_path, LIST_PATH, {"schema_version": 1, "files": [], "reason": "new"}
        )
    elif case == "tool_changed":
        save(tmp_path, TOOL_FILES[1], b"fictional changed tool", raw=True)
    seal = {
        "role": "technical",
        "status": "completed",
        "started_at": "2026-10-16T01:00:00+00:00",
        "completed_at": "2026-10-16T01:30:00+00:00",
        "report_sha256s": [report["sha256"]],
    }
    arguments = (tmp_path, manifest, {"/root/tool_auditor": "technical"}, {"fixture": {"technical": seal}})
    expected = {
        "missing": "工具審閱的 checked_files 沒有列補充還原清單：" + LIST_PATH,
        "wrong_digest": "工具審閱核對的補充還原清單不是現行版本：" + LIST_PATH,
        "list_updated": "工具審閱核對的補充還原清單不是現行版本：" + LIST_PATH,
        "tool_changed": "tool audit version stale: " + TOOL_FILES[1],
    }
    if case in expected:
        with pytest.raises(ValueError, match=exactly(expected[case])):
            review.check_tool_audit(*arguments)
    else:
        review.check_tool_audit(*arguments)


# 以下測作者補丁：manifest 的 author_patches 指向作者直接要求、沒有經過 AI 審閱的小補丁清單。閘門把現行檔倒退這些補丁，
# 得到的較早版本只在報告的 source_sha256、修正的複查與 technical 的 repository_code 來源承認。

PATCH_OLD = "第二項的重要程度增加四倍，其梯度也增加四倍。"
PATCH_NEW = "第二項的重要程度變成四倍，其梯度也變成四倍。"
CHAPTER = "course/chapters/99.md"
INTRO = "# 第九十九章\n\n導言。\n\n"
REVIEWED = "## 99.1 權重\n\n" + PATCH_OLD + "\n"
PATCHED = "## 99.1 權重\n\n" + PATCH_NEW + "\n"
OTHER = "## 99.1 權重\n\n第二項的重要程度增加三倍，其梯度也增加三倍。\n"
FILE_REVIEWED = INTRO + REVIEWED
FILE_PATCHED = INTRO + PATCHED
PLAIN = "# 沒有改過的頁\n"
PATCH_LIST = "round/author-patches.json"
ENTRY = "作者補丁清單 " + PATCH_LIST + " 的 patches[{}]（id {!r}）："
UNDO = "作者補丁清單的 patches[{}]（id {!r}）："
SOURCES = {
    "99.1": {"page_id": "99.1", "source": CHAPTER},
    "plain": {"page_id": "plain", "source": "course/plain.md"},
    # 首頁的 source 是產生它的 Python 程式。
    "index": {"page_id": "index", "source": "scripts/export_course.py"},
}


def patch_entry(**changes):
    entry = {
        "id": "ch99-1",
        "path": CHAPTER,
        "old": PATCH_OLD,
        "new": PATCH_NEW,
        "requested_by": "author",
        "requested_on": "2026-10-20",
        "reason": "作者要求把「增加四倍」改成「變成四倍」",
    }
    return {**entry, **changes}


def patch_list(patches, **changes):
    return {
        "schema_version": 1,
        "kind": review.PATCHES_KIND,
        "statement": review.PATCHES_STATEMENT,
        "patches": patches,
        **changes,
    }


def listed_patches(root, data, raw=False, reference=lambda listed: listed):
    listed = reference(save(root, PATCH_LIST, data, raw=raw))
    return review.author_patches(root, {"author_patches": listed}, SOURCES)


def test_author_patch_list_is_optional_and_returned_as_listed(tmp_path):
    # 欄位缺席時什麼都不讀，連目前清單都不碰。
    assert review.author_patches(tmp_path, {}, None) == []
    entries = [patch_entry(), patch_entry(id="ch99-2", old="導言。", new="這一章的導言。", requested_on="2026-10-21")]
    assert listed_patches(tmp_path, patch_list(entries)) == entries


def test_old_and_new_may_be_exactly_eighty_characters(tmp_path):
    entry = patch_entry(old="增" * 80, new="變" * 80)
    assert listed_patches(tmp_path, patch_list([entry])) == [entry]


FIELDS_RULE = (
    "作者補丁清單 " + PATCH_LIST + " 的 patches[0] 必須是恰好含 " + "、".join(review.PATCH_FIELDS) + " 七個欄位的物件"
)


@pytest.mark.parametrize(
    ("patches", "message"),
    [
        (
            [patch_entry(), patch_entry(old="導言。", new="這一章的導言。")],
            ENTRY.format(1, "ch99-1") + "id 和前面的補丁重複",
        ),
        ([patch_entry(id=" ")], ENTRY.format(0, " ") + "id 必須是非空字串"),
        ([patch_entry(path="scripts/tool.py")], ENTRY.format(0, "ch99-1") + "path 不是 manifest 任何一頁的 source"),
        (
            [patch_entry(path="scripts/export_course.py", old='return ""', new='return "\\t"')],
            ENTRY.format(0, "ch99-1") + "path 不是 Markdown（.md）原文；程式檔整份是程式碼，不能用作者補丁改",
        ),
        ([patch_entry(old="")], ENTRY.format(0, "ch99-1") + "old 必須是非空字串"),
        ([patch_entry(new=7)], ENTRY.format(0, "ch99-1") + "new 必須是非空字串"),
        ([patch_entry(new=PATCH_OLD)], ENTRY.format(0, "ch99-1") + "old 與 new 相同"),
        ([patch_entry(old="增" * 81)], ENTRY.format(0, "ch99-1") + "old 有 81 個字元，超過 80"),
        ([patch_entry(new="變" * 81)], ENTRY.format(0, "ch99-1") + "new 有 81 個字元，超過 80"),
        ([patch_entry(requested_by="coordinator")], ENTRY.format(0, "ch99-1") + "requested_by 必須是 'author'"),
        (
            [patch_entry(requested_on="2026/10/20")],
            ENTRY.format(0, "ch99-1") + "requested_on 必須是 YYYY-MM-DD 格式的日期",
        ),
        (
            [patch_entry(requested_on="20261020")],
            ENTRY.format(0, "ch99-1") + "requested_on 必須是 YYYY-MM-DD 格式的日期",
        ),
        (
            [patch_entry(requested_on="2026-02-30")],
            ENTRY.format(0, "ch99-1") + "requested_on 必須是 YYYY-MM-DD 格式的日期",
        ),
        ([patch_entry(reason=" ")], ENTRY.format(0, "ch99-1") + "reason 必須是非空字串"),
        ([{**patch_entry(), "reviewed_by": "nobody"}], FIELDS_RULE),
        ([{key: value for key, value in patch_entry().items() if key != "reason"}], FIELDS_RULE),
    ],
    ids=[
        "duplicate_id",
        "blank_id",
        "path_not_page_source",
        "path_is_python_page_source",
        "empty_old",
        "new_not_string",
        "same_text",
        "old_81",
        "new_81",
        "requested_by",
        "date_slashes",
        "date_compact",
        "date_impossible",
        "blank_reason",
        "extra_field",
        "missing_field",
    ],
)
def test_each_patch_that_breaks_a_rule_is_named_with_that_rule(tmp_path, patches, message):
    with pytest.raises(ValueError, match=exactly(message)):
        listed_patches(tmp_path, patch_list(patches))


def test_every_forbidden_mark_in_old_or_new_is_rejected(tmp_path):
    # 記號取自 PATCH_FORBIDDEN；規約至少要擋換行、兩種程式區塊的圍欄、圖與連結的語法。
    assert {"\n", "```", "~~~", "![", "]("} <= set(review.PATCH_FORBIDDEN)
    for token, label in review.PATCH_FORBIDDEN.items():
        for field in ("old", "new"):
            entry = patch_entry(**{field: "權重" + token + "四倍"})
            with pytest.raises(ValueError, match=exactly(ENTRY.format(0, "ch99-1") + f"{field} 不能含{label}")):
                listed_patches(tmp_path, patch_list([entry]))


@pytest.mark.parametrize(
    ("data", "reference", "message"),
    [
        (patch_list([patch_entry()], schema_version=2), None, "作者補丁清單的 schema_version 不是 1：" + PATCH_LIST),
        (patch_list([patch_entry()], schema_version=True), None, "作者補丁清單的 schema_version 不是 1：" + PATCH_LIST),
        (
            patch_list([patch_entry()], kind="author patches"),
            None,
            "作者補丁清單的 kind 不是 'author patches without AI review'：" + PATCH_LIST,
        ),
        (
            patch_list([patch_entry()], statement="作者要求的修改"),
            None,
            "作者補丁清單的 statement 不是規定的聲明：" + PATCH_LIST,
        ),
        (patch_list({"0": patch_entry()}), None, "作者補丁清單的 patches 不是 list：" + PATCH_LIST),
        ([patch_entry()], None, "作者補丁清單不是 JSON 物件：" + PATCH_LIST),
        (b"not json", None, "作者補丁清單不是合法的 JSON：" + PATCH_LIST),
        (patch_list([]), lambda ref: {**ref, "sha256": "0" * 64}, "artifact changed: " + PATCH_LIST),
        (
            patch_list([]),
            lambda ref: {**ref, "path": "round/absent.json"},
            "作者補丁清單讀不到（FileNotFoundError）：round/absent.json",
        ),
    ],
    ids=[
        "schema_version",
        "schema_version_bool",
        "kind",
        "statement",
        "not_list",
        "not_object",
        "not_json",
        "digest",
        "missing",
    ],
)
def test_patch_list_itself_is_bound_and_declares_no_ai_review(tmp_path, data, reference, message):
    options = {"reference": reference} if reference else {}
    with pytest.raises(ValueError, match=exactly(message)):
        listed_patches(tmp_path, data, raw=isinstance(data, bytes), **options)


def chapter_versions(root, text, patches):
    save(root, CHAPTER, text.encode(), raw=True)
    return review.file_versions(root, patches)


def test_file_versions_undo_the_list_from_its_last_patch(tmp_path):
    # 同一處先後補了兩次：先倒退第二筆，第一筆的 new 才會出現。
    patches = [
        patch_entry(id="p1", old="權重增加四倍。", new="權重變成四倍。"),
        patch_entry(id="p2", old="權重變成四倍。", new="權重變成原來的四倍。"),
    ]
    once = INTRO + "## 99.1 權重\n\n權重變成四倍。\n"
    reviewed = INTRO + "## 99.1 權重\n\n權重增加四倍。\n"
    versions = chapter_versions(tmp_path, INTRO + "## 99.1 權重\n\n權重變成原來的四倍。\n", patches)
    assert versions == {CHAPTER: {digest(once.encode()): ("p2",), digest(reviewed.encode()): ("p2", "p1")}}


@pytest.mark.parametrize(
    ("text", "count"),
    [(FILE_PATCHED + "\n" + PATCH_NEW + "\n", 2), (FILE_REVIEWED, 0)],
    ids=["twice", "absent"],
)
def test_new_text_must_occur_exactly_once_when_it_is_undone(tmp_path, text, count):
    message = UNDO.format(0, "ch99-1") + f"new 在倒退到這一筆時的 {CHAPTER} 出現 {count} 次，必須恰好一次"
    with pytest.raises(ValueError, match=exactly(message)):
        chapter_versions(tmp_path, text, [patch_entry()])


def test_overlapping_occurrences_of_new_text_are_counted_separately(tmp_path):
    # 「四四」在「四四四」裡有兩個起點；只數不重疊的出現會數成一次。
    message = UNDO.format(0, "ch99-1") + f"new 在倒退到這一筆時的 {CHAPTER} 出現 2 次，必須恰好一次"
    with pytest.raises(ValueError, match=exactly(message)):
        chapter_versions(tmp_path, INTRO + "## 99.1 權重\n\n四四四\n", [patch_entry(old="三三", new="四四")])


@pytest.mark.parametrize(
    "body",
    [
        "```python\nprint('" + PATCH_NEW + "')\n```\n",
        "```text " + PATCH_NEW + "\nx\n```\n",
        "   ```\n" + PATCH_NEW + "\n   ```\n",
        "```\n" + PATCH_NEW + "\n",
        "~~~python\nprint('" + PATCH_NEW + "')\n~~~\n",
        "> ```\n> " + PATCH_NEW + "\n> ```\n",
        # 收尾要是同一串記號、後面只接空白：較短、較長、另一種字元或帶語言名的那一行都還在區塊裡。
        "````markdown\n```\n" + PATCH_NEW + "\n````\n",
        "```\n````\n" + PATCH_NEW + "\n```\n",
        "~~~\n```\n" + PATCH_NEW + "\n~~~\n",
        "```\n```python\n" + PATCH_NEW + "\n```\n",
    ],
    ids=[
        "inside",
        "fence_line",
        "indented_fence",
        "unclosed",
        "tilde",
        "blockquote",
        "shorter",
        "longer",
        "other_char",
        "info",
    ],
)
def test_new_text_inside_a_fenced_code_block_is_rejected(tmp_path, body):
    message = UNDO.format(0, "ch99-1") + f"new 在 {CHAPTER} 的那一處落在程式圍欄圍起來的區塊裡"
    with pytest.raises(ValueError, match=exactly(message)):
        chapter_versions(tmp_path, INTRO + "## 99.1 權重\n\n" + body, [patch_entry()])


@pytest.mark.parametrize(
    "block",
    ["```python\nprint(4)\n```\n", "~~~\nprint(4)\n~~~\n", "````\n```\nprint(4)\n````\n", "```\nprint(4)\n```  \n"],
    ids=["backticks", "tildes", "longer_fence", "trailing_space"],
)
def test_new_text_after_a_closed_fenced_block_is_undone(tmp_path, block):
    text = INTRO + "## 99.1 權重\n\n" + block + "\n" + PATCH_NEW + "\n"
    versions = chapter_versions(tmp_path, text, [patch_entry()])
    assert versions == {CHAPTER: {digest(text.replace(PATCH_NEW, PATCH_OLD).encode()): ("ch99-1",)}}


def test_page_versions_undo_only_the_patches_found_on_that_page(tmp_path):
    # 同一個章檔：p1 改導言、p2 改 99.1；99.1 這一頁只倒退 p2。
    patches = [patch_entry(id="p1", old="導言。", new="這一章的導言。"), patch_entry(id="p2")]
    snapshot = save(tmp_path, "freeze/99.1.md", PATCHED.encode(), raw=True)
    page = {"page_id": "99.1", "source": CHAPTER, "source_sha256": snapshot["sha256"]}
    frozen = {"snapshot": snapshot["path"]}
    assert review.page_versions(tmp_path, frozen, page, patches) == {digest(REVIEWED.encode()): ("p2",)}
    # source 沒有補丁的頁不讀快照。
    other = {**page, "source": "course/plain.md"}
    assert review.page_versions(tmp_path, {"snapshot": "absent.md"}, other, patches) == {}


# 迷你一輪給整個 audit_manifest 跑：99.1 審閱後被作者補了一句，plain 沿用舊判斷；重放、目前清單與建置換成替身，
# check_technical_reviews.py 用現行這一份。

READER_TASK = "/root/fixture_reader"
TECHNICAL_TASK = "/root/fixture_technical"
AUDITOR_TASK = "/root/fixture_tool_auditor"
PATCH_WAVE = "fixture-wave"
MODEL = "tiny/model.py"
MODEL_BYTES = b"def model():\n    return 4\n"
GATE_STUBS = {
    "docs/review-tools/baseline_replay.py": "def replay(root, supplements=()):\n    return {'failures': []}\n",
    "docs/review-tools/composite_review.py": (
        "import json\nfrom pathlib import Path\n\n\ndef current_inventory(root):\n"
        "    return json.loads((Path(root) / 'inventory.json').read_text(encoding='utf-8'))\n"
    ),
    "scripts/build_course.py": "def build(check=False):\n    return None\n",
    "docs/course-revision-20261016/tools/reader.py": "def units(text):\n    return [text]\n",
}
GATE_PLACEHOLDERS = (
    "docs/review-tools/incremental_revision.py",
    "docs/review-tools/phase7_review.py",
    "docs/review-tools/incremental_reader.py",
    "scripts/reading_time.py",
    "scripts/export_course.py",
    "docs/course-revision-20261016/tools/freeze_sources.py",
    "tests/test_incremental_revision_review.py",
)
NOTE_FIELDS = ("understanding", "materials_and_labels", "expected_change", "confusion_and_quote", "missing_visuals")
CONCEPT_FIELDS = ("concept", "identity", "mechanism", "need", "usage", "example_scope", "source_quote")
TECHNICAL_CHECKS = (
    "factual_accuracy",
    "numeric_verification",
    "figure_consistency",
    "source_verification",
    "limitations",
)


def when(minute):
    return f"2026-10-16T03:{minute:02d}:00+00:00"


def code_source(identifier, path, data):
    return {
        "id": identifier,
        "kind": "repository_code",
        "title": "虛構來源",
        "path": path,
        "sha256": digest(data),
        "version": "審閱時的工作樹",
        "inspection_note": "虛構讀取紀錄",
        "verified": True,
    }


def gate_round(
    root,
    listed=True,
    patches=None,
    reader_on=REVIEWED,
    fix_on=None,
    technical_on=REVIEWED,
    chapter_cited=FILE_REVIEWED,
    model_cited=MODEL_BYTES,
):
    """寫出迷你一輪，回傳 manifest 路徑。

    reader_on、fix_on、technical_on 是讀者目前判斷、修正的複查（None 時就是目前判斷）與技術報告記的那一版 99.1；
    chapter_cited、model_cited 是技術報告兩個 repository_code 來源記的那一版檔案。listed 時帶作者補丁清單，
    patches 指定清單內容（預設是補 99.1 的那一筆）。
    """
    for name, text in GATE_STUBS.items():
        save(root, name, text.encode(), raw=True)
    for name in GATE_PLACEHOLDERS:
        save(root, name, ("fictional " + name + "\n").encode(), raw=True)
    save(
        root, "scripts/check_technical_reviews.py", (ROOT / "scripts/check_technical_reviews.py").read_bytes(), raw=True
    )
    criteria = save(root, ".agents/skills/clear-tutorial/SKILL.md", b"fictional criteria\n", raw=True)
    save(root, CHAPTER, FILE_PATCHED.encode(), raw=True)
    save(root, "course/plain.md", PLAIN.encode(), raw=True)
    save(root, MODEL, MODEL_BYTES, raw=True)
    lesson = {
        "page_id": "99.1",
        "source": CHAPTER,
        "selector": "section:99.1",
        "source_sha256": digest(PATCHED.encode()),
        "figures_sha256": {},
    }
    plain = {
        "page_id": "plain",
        "source": "course/plain.md",
        "selector": "whole-file",
        "source_sha256": digest(PLAIN.encode()),
        "figures_sha256": {},
    }
    save(root, "inventory.json", {"pages": [lesson, plain]})
    snapshots = {
        "99.1": save(root, "freeze/99.1.md", PATCHED.encode(), raw=True)["path"],
        "plain": save(root, "freeze/plain.md", PLAIN.encode(), raw=True)["path"],
    }
    pages = [{**page, "snapshot": snapshots[page["page_id"]]} for page in (lesson, plain)]
    freeze = save(root, "round/freeze.json", {"pages": pages, "figure_snapshots": {}})
    baseline = save(root, "old/baseline-freeze.json", {"pages": pages[1:], "criteria": ["SKILL.md"]})
    inputs = save(root, "old/inputs.json", {"files_sha256": {}})
    save(root, REVIEW_MANIFEST, {"source_freeze": baseline, "current_technical_inputs": inputs})

    seen = save(root, "sessions/s1/source.md", REVIEWED.encode(), raw=True)
    opening = {
        "page_id": "99.1",
        "reviewer_task": READER_TASK,
        "source_sha256": seen["sha256"],
        "figures_sha256": {},
        "started_at": when(5),
    }
    note = {
        "page_id": "99.1",
        "reviewer_task": READER_TASK,
        "unit_index": 0,
        "unit_sha256": seen["sha256"],
        "saved_at": when(6),
        **{field: "虛構觀察" for field in NOTE_FIELDS},
        "concept_checks": [{**{field: "虛構回答" for field in CONCEPT_FIELDS}, "verdict": "已足夠"}],
    }
    session = {
        "opening": save(root, "sessions/s1/opening.json", opening),
        "source": seen,
        "notes": [save(root, "sessions/s1/note-0.json", note)],
        "figures": {},
    }
    issue = {"id": "R-1", "problem": "虛構問題"}
    original_page = {"page_id": "99.1", "source_sha256": seen["sha256"], "figures_sha256": {}, "verdict": "需補說明"}
    original = save(
        root,
        "reviews/reader-original.json",
        {"reviewer_task": READER_TASK, "created_at": when(7), "pages": [{**original_page, "issues": [issue]}]},
    )
    current_page = {
        "page_id": "99.1",
        "source_sha256": digest(reader_on.encode()),
        "figures_sha256": {},
        "verdict": "pass",
        "basis_report": "reviews/reader-current.json",
        "ai_estimate": {"minutes_min": 3, "minutes_max": 5, "reason": "虛構估計"},
    }
    current = save(
        root,
        "reviews/reader-current.json",
        {"reviewer_task": READER_TASK, "created_at": when(8), "pages": [current_page]},
    )
    recheck = current
    if fix_on is not None:
        recheck_page = {
            "page_id": "99.1",
            "source_sha256": digest(fix_on.encode()),
            "figures_sha256": {},
            "verdict": "pass",
        }
        recheck = save(root, "reviews/reader-recheck.json", {"reviewer_task": READER_TASK, "pages": [recheck_page]})
    reader_proof = {
        "mode": "review",
        "wave": PATCH_WAVE,
        "report": current,
        "json_pointer": "/pages/0",
        "original_report": original,
        "original_json_pointer": "/pages/0",
        "basis": {"report": current, "json_pointer": "/pages/0"},
        "sessions": [session],
    }

    technical_report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "99.1",
        "source": CHAPTER + "#99.1",
        "source_sha256": digest(technical_on.encode()),
        "verdict": "pass",
        "reviewer_task": TECHNICAL_TASK,
        "reviewer_context": "fresh",
        "figure_sha256": {},
        "artifacts": [],
        "claims": [],
        "sources": [code_source("chapter", CHAPTER, chapter_cited.encode()), code_source("model", MODEL, model_cited)],
        "applicability": {"substantive_claims": False, "reason": "虛構頁沒有實質主張"},
        "issues": [],
        "checks": {name: {"status": "not_applicable", "details": "虛構核對"} for name in TECHNICAL_CHECKS},
    }
    technical = save(root, "reviews/technical.json", technical_report)
    technical_proof = {
        "mode": "review",
        "wave": PATCH_WAVE,
        "report": technical,
        "json_pointer": "",
        "original_report": technical,
        "original_json_pointer": "",
    }
    tools = [{"path": name, "sha256": digest((root / name).read_bytes())} for name in TOOL_FILES]
    audit = save(
        root, "round/tool-audit.json", {"reviewer_task": AUDITOR_TASK, "verdict": "pass", "checked_files": tools}
    )

    later = {"not_before": when(42), "not_after": when(43)}
    tasks = [
        {
            "task": READER_TASK,
            "role": "reader",
            "wave": PATCH_WAVE,
            "context": "fresh",
            "pages": ["99.1"],
            "dispatch_window": {"not_before": when(1), "not_after": when(2)},
        },
        {
            "task": TECHNICAL_TASK,
            "role": "technical",
            "wave": PATCH_WAVE,
            "context": "fresh",
            "lesson_id": "99.1",
            "dispatch_window": later,
        },
        {"task": AUDITOR_TASK, "role": "technical", "wave": PATCH_WAVE, "context": "fresh", "dispatch_window": later},
    ]
    seals = {
        "reader": save(
            root,
            "round/seal-reader.json",
            {
                "role": "reader",
                "status": "completed",
                "started_at": when(0),
                "completed_at": when(40),
                "report_sha256s": [original["sha256"], current["sha256"]],
            },
        ),
        "technical": save(
            root,
            "round/seal-technical.json",
            {
                "role": "technical",
                "status": "completed",
                "started_at": when(41),
                "completed_at": when(59),
                "report_sha256s": [technical["sha256"], audit["sha256"]],
            },
        ),
    }
    decision = {
        "issue_id": "R-1",
        "report": original,
        "json_pointer": "/pages/0/issues/0",
        "final_necessity": "necessary",
        "action": "fixed",
        **{field: "虛構理由" for field in ("source_support", "learning_benefit", "change_burden", "reason")},
        "affected_pages": ["99.1"],
        "verification": [{"report": recheck, "json_pointer": "/pages/0"}],
    }
    inherited = {"mode": "inherit", "baseline_page_id": "plain", "dependency_note": "這一頁沒有改"}
    implementation = [*GATE_STUBS, *GATE_PLACEHOLDERS, "scripts/check_technical_reviews.py"]
    manifest = {
        "review_policy": review.POLICY,
        "schema_version": 2,
        "baseline_freeze": baseline,
        "source_freeze": freeze,
        "implementation": {name: digest((root / name).read_bytes()) for name in implementation},
        "criteria": {criteria["path"]: criteria["sha256"]},
        "author_tasks": ["/root"],
        "assignments": save(root, "round/assignments.json", {"tasks": tasks}),
        "review_waves": [{"id": PATCH_WAVE, "seals": seals}],
        "pages": [
            {"page_id": "99.1", "reader": reader_proof, "technical": technical_proof},
            {"page_id": "plain", **{role: inherited for role in review.ROLES}},
        ],
        "issue_reports": [original, current, technical, audit] + ([recheck] if fix_on is not None else []),
        "dispositions": save(
            root, "round/dispositions.json", {"at": when(21), "pending_count": 0, "items": [decision]}
        ),
        "dispositions_collection": {"received_at": when(22), "wave": PATCH_WAVE, "role": "reader"},
        "tool_audit": {"wave": PATCH_WAVE, "report": audit},
    }
    if listed:
        manifest["author_patches"] = save(root, PATCH_LIST, patch_list([patch_entry()] if patches is None else patches))
    return save(root, "round/manifest.json", manifest)["path"]


def audit(root, stage="reader", **options):
    return review.audit_manifest(root, root / gate_round(root, **options), stage)


def records(*patched):
    first = {"page_id": "99.1", "source_sha256": digest(PATCHED.encode())}
    if patched:
        first["author_patches"] = list(patched)
    return [first, {"page_id": "plain", "source_sha256": digest(PLAIN.encode())}]


@pytest.mark.parametrize("stage", ["reader", "technical"])
def test_reviews_of_the_version_before_an_author_patch_still_count_and_are_marked(tmp_path, stage):
    result = audit(tmp_path, stage)
    assert result["failures"] == []
    assert result["records"] == records("ch99-1")


@pytest.mark.parametrize("stage", ["reader", "technical"])
def test_without_the_patch_list_the_same_round_fails_as_it_did_before(tmp_path, stage):
    assert audit(tmp_path, stage, listed=False)["failures"] == ["report source is stale: 99.1"]


@pytest.mark.parametrize("stage", ["reader", "technical"])
def test_reviews_of_the_current_version_are_not_marked_with_or_without_the_list(tmp_path, stage):
    current = {"reader_on": PATCHED, "technical_on": PATCHED, "chapter_cited": FILE_PATCHED}
    for listed in (True, False):
        result = audit(tmp_path / str(listed), stage, listed=listed, **current)
        assert result["failures"] == []
        assert result["records"] == records()


def test_fix_recheck_before_the_patch_is_accepted_and_marks_the_page(tmp_path):
    # 目前判斷已對著現行版本重做，只剩修正的複查停在補之前的版本。
    result = audit(tmp_path, reader_on=PATCHED, fix_on=REVIEWED)
    assert result["failures"] == []
    assert result["records"] == records("ch99-1")
    unlisted = audit(tmp_path / "unlisted", reader_on=PATCHED, fix_on=REVIEWED, listed=False)
    assert unlisted["failures"] == ["fix recheck is stale"]


def test_technical_report_before_the_patch_is_accepted_and_marks_the_page(tmp_path):
    result = audit(tmp_path, "technical", reader_on=PATCHED)
    assert result["failures"] == []
    assert result["records"] == records("ch99-1")


def test_chapter_cited_before_the_patch_alone_is_accepted_and_marks_the_page(tmp_path):
    # 讀者與技術報告都對著現行版本，只有 repository_code 來源記的章檔還是補之前的版本。
    options = {"reader_on": PATCHED, "technical_on": PATCHED}
    result = audit(tmp_path, "technical", **options)
    assert result["failures"] == []
    assert result["records"] == records("ch99-1")
    unlisted = audit(tmp_path / "unlisted", "technical", listed=False, **options)
    assert unlisted["failures"] == ["99.1: ['source chapter: 證據版本已改變：course/chapters/99.md']"]


@pytest.mark.parametrize(
    "model_cited", [b"def model():\n    return 3\n", FILE_REVIEWED.encode()], ids=["stale", "chapter_digest"]
)
def test_older_digest_of_a_file_without_patches_is_still_rejected(tmp_path, model_cited):
    # 沒有補丁的檔只認現行位元組，借用補丁檔較早版本的 SHA 也一樣。
    result = audit(tmp_path, "technical", model_cited=model_cited)
    assert result["failures"] == ["99.1: ['source model: 證據版本已改變：tiny/model.py']"]


@pytest.mark.parametrize(
    ("stage", "options", "failure"),
    [
        ("reader", {"reader_on": OTHER}, "report source is stale: 99.1"),
        ("reader", {"reader_on": PATCHED, "fix_on": OTHER}, "fix recheck is stale"),
        ("technical", {"technical_on": OTHER}, "report source is stale: 99.1"),
        (
            "technical",
            {"reader_on": PATCHED, "technical_on": PATCHED, "chapter_cited": INTRO + OTHER},
            "99.1: ['source chapter: 證據版本已改變：course/chapters/99.md']",
        ),
    ],
    ids=["reader_report", "fix_recheck", "technical_report", "cited_chapter"],
)
def test_older_version_the_patches_do_not_lead_back_to_is_still_stale(tmp_path, stage, options, failure):
    assert audit(tmp_path, stage, **options)["failures"] == [failure]


@pytest.mark.parametrize(
    ("patches", "failure"),
    [
        (
            [patch_entry(new="第二項的重要程度變成五倍，其梯度也變成五倍。")],
            UNDO.format(0, "ch99-1") + f"new 在倒退到這一筆時的 {CHAPTER} 出現 0 次，必須恰好一次",
        ),
        ([patch_entry(new="變" * 81)], ENTRY.format(0, "ch99-1") + "new 有 81 個字元，超過 80"),
    ],
    ids=["not_in_current_file", "too_long"],
)
def test_patch_list_that_breaks_a_rule_fails_the_whole_gate(tmp_path, patches, failure):
    result = audit(tmp_path, patches=patches)
    assert result["failures"] == [failure]
    assert result["records"] == []
