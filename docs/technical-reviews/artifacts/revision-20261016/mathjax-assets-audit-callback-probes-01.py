"""MATHJAX-01 知情獨立回查；不改原收據或實作，不重新觀察預覽。"""

import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import stage_mathjax as stage

ART = Path(__file__).resolve().parent
RESULT = ART / "mathjax-assets-audit-callback-probes-results-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def inventory(path):
    if path.is_symlink():
        return {".": {"kind": "symlink", "target": str(path.readlink())}}
    if not path.exists():
        return {}
    result = {".": {"kind": "directory"}}
    for entry in sorted(path.rglob("*")):
        name = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            result[name] = {"kind": "symlink", "target": str(entry.readlink())}
        elif entry.is_file():
            raw = entry.read_bytes()
            result[name] = {"kind": "file", "bytes": len(raw), "sha256": sha(raw)}
        else:
            result[name] = {"kind": "directory"}
    return result


def no_network(*args, **kwargs):
    raise AssertionError("完整快取不得連網")


def main():
    started = datetime.now(UTC).isoformat()
    checked = [ROOT / name for name in (
        "scripts/stage_mathjax.py", "tests/test_stage_mathjax.py", "tests/test_reading_time.py",
        "scripts/export_course.py", "tests/test_course_export.py", "course/web/mathjax-assets.json",
    )]
    original = ROOT / "docs/course-revision-20261016/reviews/mathjax-assets-audit-original.json"
    prior = json.loads(original.read_bytes())
    preserved_paths = [original, ROOT / "docs/course-revision-20261016/reviews/tool-audit-callback-03.json"]
    preserved_paths += [ROOT / item["path"] for item in prior["artifacts"]]
    preserved = [ref(path) for path in preserved_paths]
    for item in prior["artifacts"]:
        assert ref(ROOT / item["path"])["sha256"] == item["sha256"]
    assert preserved[0]["sha256"] == "d4808f1e8918486edc99a0c59b5a3d544120bff0de53b351f9243a98ed9ec1da"
    assert preserved[1]["sha256"] == "90c2dc8bc790b95953c8026e2d713fd68a84bbd9b64ba9a06f604563e7bcd719"
    checked_before = [ref(path) for path in checked]
    results = []

    def record(name, operation):
        try:
            detail = operation()
        except Exception as error:
            results.append({"id": name, "passed": False, "error_type": type(error).__name__, "error": str(error)})
        else:
            results.append({"id": name, "passed": True, "details": detail})

    workspace = ROOT / "work"
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mathjax-callback-", dir=workspace) as temporary:
        temporary = Path(temporary)
        count = 0

        def fixture():
            nonlocal count
            count += 1
            root = temporary / f"fixture-{count}"
            cache = root / "outputs/mathjax-cache/3.2.2"
            value = copy.deepcopy(json.loads((ROOT / "course/web/mathjax-assets.json").read_bytes()))
            value["files"] = []
            for name in ("LICENSE", "tex-chtml.js", "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"):
                raw = ("independent-callback-synthetic:" + name).encode()
                target = cache / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                value["files"].append({"path": name, "upstream_path": name if name == "LICENSE" else "es5/" + name,
                                       "bytes": len(raw), "sha256": sha(raw)})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(stage.urllib.request, "urlopen", no_network):
                prepared = stage.prepare_assets(root, manifest)
            assert prepared[0] == cache
            return root, cache, prepared

        def synthetic_positive():
            root, cache, prepared = fixture()
            destination = root / "destination"
            stage.stage_assets(destination, prepared)
            assert inventory(destination) == inventory(cache)
            return {"normal_prepare_tuple": True, "matching_files": len(prepared[1]["files"])}

        record("complete_cache_and_stage_positive", synthetic_positive)

        def real_positive():
            original_cache = ROOT / "outputs/mathjax-cache/3.2.2"
            before = inventory(original_cache)
            with patch.object(stage.urllib.request, "urlopen", no_network):
                prepared = stage.prepare_assets(ROOT)
                destination = temporary / "real-assets"
                stage.stage_assets(destination, prepared)
            verified = []
            for item in prepared[1]["files"]:
                raw = (destination / item["path"]).read_bytes()
                assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
                assert raw == (original_cache / item["path"]).read_bytes()
                verified.append({"path": item["path"], "bytes": len(raw), "sha256": sha(raw)})
            assert inventory(original_cache) == before
            return {"normal_prepare_tuple": True, "network_disabled": True, "cache_unchanged": True,
                    "verified_file_count": len(verified), "verified_bytes": sum(x["bytes"] for x in verified),
                    "staged_identity_sha256": sha(json.dumps(verified, sort_keys=True).encode())}

        record("real_94_assets_offline_stage_positive", real_positive)

        def symlink(kind):
            root, _, prepared = fixture()
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel"
            sentinel.write_bytes(b"outside unchanged")
            destination = root / "destination"
            if kind == "destination":
                destination.symlink_to(outside, target_is_directory=True)
            else:
                destination.mkdir()
                if kind == "file":
                    (destination / "LICENSE").symlink_to(sentinel)
                else:
                    (destination / "output").symlink_to(outside, target_is_directory=True)
            outside_before, destination_before = inventory(outside), inventory(destination)
            caught = None
            try:
                stage.stage_assets(destination, prepared)
            except ValueError as error:
                caught = str(error)
            assert caught is not None and "symlink" in caught
            outside_after, destination_after = inventory(outside), inventory(destination)
            assert outside_before == outside_after
            assert destination_before == destination_after
            return {"normal_prepare_tuple": True, "actual": "rejected", "exception": caught,
                    "outside_changed": False, "destination_changed": False,
                    "outside_before": outside_before, "outside_after": outside_after,
                    "destination_before": destination_before, "destination_after": destination_after}

        record("destination_file_symlink_must_not_escape", lambda: symlink("file"))
        record("destination_parent_symlink_must_not_escape", lambda: symlink("parent"))
        record("destination_root_symlink_must_not_escape", lambda: symlink("destination"))

        def changed_source(symlink_source=False):
            root, cache, prepared = fixture()
            source = cache / "tex-chtml.js"
            if symlink_source:
                external = root / "same-valid-source"
                external.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(external)
            else:
                source.write_bytes(b"changed after valid prepare" * 1000)
            caught = None
            try:
                stage.stage_assets(root / "destination", prepared)
            except ValueError as error:
                caught = str(error)
            assert caught is not None
            assert ("symlink" if symlink_source else "指紋不符") in caught
            return {"normal_prepare_tuple": True, "actual": "rejected", "exception": caught,
                    "source_mutated_only_after_prepare": True}

        record("source_corruption_after_prepare_rejected", changed_source)
        record("source_symlink_after_prepare_rejected", lambda: changed_source(True))

        def modified_manifest():
            root, _, prepared = fixture()
            prepared[1]["files"][-1]["path"] = "../escape"
            destination = root / "destination"
            before = inventory(root)
            caught = None
            try:
                stage.stage_assets(destination, prepared)
            except ValueError as error:
                caught = str(error)
            assert caught is not None and "安全相對路徑" in caught
            assert inventory(root) == before
            return {"actual": "rejected", "exception": caught, "root_inventory_unchanged": True,
                    "manifest_mutated_only_after_valid_prepare": True}

        record("stage_revalidates_prepared_manifest_before_writing", modified_manifest)

    assert checked_before == [ref(path) for path in checked]
    assert preserved == [ref(path) for path in preserved_paths]
    result = {
        "reviewer_task": "/root/tech_20261016_review_tools",
        "initial_context": "fresh", "current_context": "informed",
        "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
        "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                        "torch_distribution": importlib.metadata.version("torch"),
                        "pytest": importlib.metadata.version("pytest")},
        "checked_files": checked_before, "preserved_original_refs": preserved,
        "probes": results, "probe_count": len(results), "all_probes_passed": all(x["passed"] for x in results),
        "limits": ["只回查MATHJAX-01與相關正向/來源複驗；沒有重新觀察瀏覽器、重讀上游或聲稱全站/全輪驗收。",
                   "測試的是靜態隔離temp樹；不宣稱防止其他process並行更換目錄的TOCTOU競態。"],
    }
    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"probe_count": len(results), "all_probes_passed": result["all_probes_passed"],
                      "failed": [x["id"] for x in results if not x["passed"]], "result": ref(RESULT)}, ensure_ascii=False))
    return 0 if result["all_probes_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
