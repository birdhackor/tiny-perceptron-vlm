"""獨立核對最小 baseline 設定補件；只讀目前與原始證據。"""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def ref(name):
    path = ROOT / name
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    started = datetime.now(UTC).isoformat()
    batch = "docs/course-revision-20261016/"
    overlay_name = batch + "baseline-overlay.json"
    initial_name = batch + "baseline-overlay-initial-01.json"
    observation_name = batch + "author-baseline-config-observation-01.json"
    snapshot_name = batch + "baseline-overlay/pyproject.toml"
    overlay = json.loads((ROOT / overlay_name).read_bytes())
    initial = json.loads((ROOT / initial_name).read_bytes())
    observation = json.loads((ROOT / observation_name).read_bytes())
    before_paths = [overlay_name, initial_name, observation_name, snapshot_name, "pyproject.toml",
                    batch + "all-gate-01.json", batch + "manifest-all-01.json",
                    batch + "reader-seal-01.json", batch + "technical-seal-01.json",
                    batch + "continuity-seal-01.json", batch + "reviews/tool-audit-callback-03.json",
                    "docs/course-publish-20261008/review-manifest.json",
                    "docs/course-publish-20261008/current-source-freeze.json",
                    "docs/review-tools/baseline_replay.py", "docs/review-tools/composite_review.py",
                    "docs/review-tools/incremental_revision.py", "docs/review-tools/phase7_review.py",
                    batch + "tools/reader.py", batch + "tools/freeze_sources.py",
                    "tests/test_incremental_revision_review.py"]
    before = [ref(name) for name in before_paths]
    assert ref(initial_name) == observation["previous_overlay"]
    assert ref(snapshot_name) == observation["new_snapshot"]
    assert ref(overlay_name)["sha256"] == observation["current_overlay_sha256"]
    assert overlay["baseline_revision"] == initial["baseline_revision"] == observation["original_revision"]
    assert overlay["files"][:-1] == initial["files"]
    remainder = copy.deepcopy(overlay)
    added = remainder["files"].pop()
    assert remainder == initial
    assert added == {"live_path": "pyproject.toml", **ref(snapshot_name)}
    assert len(overlay["files"]) == 8 and len(initial["files"]) == 7
    verified_overlay = []
    for item in overlay["files"]:
        assert ref(item["path"])["sha256"] == item["sha256"]
        verified_overlay.append({"live_path": item["live_path"], **ref(item["path"])})

    git_command = ["git", "show", overlay["baseline_revision"] + ":pyproject.toml"]
    git_raw = subprocess.check_output(git_command, cwd=ROOT)
    assert git_raw == (ROOT / snapshot_name).read_bytes()
    current = tomllib.loads((ROOT / "pyproject.toml").read_text())
    original = tomllib.loads(git_raw.decode())
    new = current["tool"]["ruff"]["extend-exclude"]
    old = original["tool"]["ruff"]["extend-exclude"]
    exclusion = "docs/course-revision-20261016/freeze-*"
    assert exclusion not in old and new.count(exclusion) == 1
    assert [name for name in new if name != exclusion] == old
    normalized = copy.deepcopy(current)
    normalized["tool"]["ruff"]["extend-exclude"] = old
    assert normalized == original

    author_failure = observation["issues"][0]["observed_failure"]
    assert ref(author_failure["path"]) == author_failure
    failure = json.loads((ROOT / author_failure["path"]).read_bytes())
    assert failure["status"] == "failed"
    assert failure["failures"] == ["preserved baseline failed: ['configuration delta does not bind current bytes']"]
    previous_tool_audit = json.loads((ROOT / (batch + "reviews/tool-audit-callback-03.json")).read_bytes())
    prior_files = {item["path"]: item["sha256"] for item in previous_tool_audit["checked_files"]}
    core = ["docs/review-tools/incremental_revision.py", "docs/review-tools/baseline_replay.py",
            "docs/review-tools/phase7_review.py", batch + "tools/reader.py", batch + "tools/freeze_sources.py",
            "tests/test_incremental_revision_review.py"]
    for name in core:
        assert ref(name)["sha256"] == prior_files[name]

    spec = importlib.util.spec_from_file_location("independent_baseline_composite", ROOT / "docs/review-tools/composite_review.py")
    composite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(composite)
    manifest_path = ROOT / "docs/course-publish-20261008/review-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    # 只測設定分支，不在目前工作樹載入需要整棵舊來源的archive aliases。
    class PinnedConfigStore:
        def raw(self, reference):
            assert reference == manifest["configuration_delta"]["baseline_snapshot"]
            raw = (ROOT / reference["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
            return raw

    store = PinnedConfigStore()
    digest = manifest["configuration_delta"]["baseline_snapshot"]["sha256"]
    caught = None
    try:
        composite.check_configuration_delta(store, manifest, digest, (ROOT / "pyproject.toml").read_bytes())
    except Exception as error:
        caught = {"type": type(error).__name__, "message": str(error)}
    assert caught and caught["message"] == "configuration delta does not bind current bytes"
    composite.check_configuration_delta(store, manifest, digest, git_raw)
    replay_name = "docs/technical-reviews/artifacts/revision-20261016/baseline-overlay-config-callback-replay-01.json"
    replay = json.loads((ROOT / replay_name).read_bytes())
    assert replay["status"] == "passed_metadata_checks" and replay["failures"] == []
    assert before == [ref(name) for name in before_paths]
    result = {
        "reviewer_task": "/root/tech_20261016_review_tools", "initial_context": "fresh", "current_context": "informed",
        "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
        "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                        "torch_distribution": importlib.metadata.version("torch")},
        "checked_files": before, "verified_overlay_entries": verified_overlay,
        "git_command": git_command, "git_snapshot_byte_exact": True,
        "only_overlay_delta": added, "current_toml_only_delta": {"added_ruff_exclusion": exclusion},
        "configuration_control": {"current_bytes": {"actual": "rejected", "exception": caught},
                                  "original_snapshot": {"actual": "accepted"}},
        "core_tool_bytes_unchanged": [ref(name) for name in core],
        "preserved_original_all_failure": author_failure,
        "baseline_replay": {**ref(replay_name), "status": replay["status"], "record_count": len(replay["records"]),
                            "failures": replay["failures"], "scope_note": replay.get("scope_note")},
        "checked_and_preserved_files_unchanged_during_probe": True,
        "limits": ["本probe的configuration control只直接重跑原validator的設定分支；未冒稱完整重跑舊七筆overlay的all gate。",
                   "另有本人CLI完整baseline replay，結果passed_metadata_checks，不等於新全輪all gate或發布驗收。"]
    }
    path = ART / "baseline-overlay-config-callback-probes-results-01.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"checks_passed": True, "replay_status": replay["status"], "replay_records": len(replay["records"]),
                      "result": ref(path.relative_to(ROOT).as_posix())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
