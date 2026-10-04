"""Recheck T.1's changed asset README and the boundary between both fetchers."""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
old_report_path = "docs/technical-reviews/artifacts/tool-choice-dep-t1-before-natural-assets-review.json"
old_report = json.loads((ROOT / old_report_path).read_text())
old_sources = {row["id"]: row for row in old_report["sources"]}
manifest = json.loads((ROOT / "assets/training/manifest.json").read_text())
expected_ids = [row["id"] for row in manifest["assets"]]
assert len(expected_ids) == 8
assert sha("assets/training/manifest.json") == old_sources["manifest"]["sha256"]
assert sha("scripts/fetch_training_assets.py") == old_sources["fetch"]["sha256"]
assert sha("tiny_perceptron/assets.py") == old_sources["unpack"]["sha256"]
record = {
    "environment": {"python": platform.python_version(), "device": "CPU / standard-library data-control inspection"},
    "scope": "Changed README and two fetcher boundaries only. No data download, model execution, GPU, training or model test-result inspection.",
    "previous_report": {"path": old_report_path, "sha256": sha(old_report_path)},
    "unchanged_original_manifest_sha256": sha("assets/training/manifest.json"),
    "original_ids": expected_ids,
    "original_license_records": {row["id"]: row["license"] for row in manifest["assets"]},
    "readme_sha256": sha("assets/training/README.md"),
    "commands": [],
}
for arguments, expected_exit in (
    (["scripts/fetch_training_assets.py", "--list"], 0),
    (["-I", "scripts/fetch_natural_data.py", "--list"], 0),
    (["scripts/fetch_training_assets.py", "--asset", "natural-vision-v3"], 2),
    (["-I", "scripts/fetch_natural_data.py", "--asset", "all"], 2),
):
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert completed.returncode == expected_exit, completed.stderr
    item = {"command": ".venv/bin/python " + " ".join(arguments), "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    record["commands"].append(item)
    if arguments[0] == "scripts/fetch_training_assets.py" and arguments[-1] == "--list":
        listed = [json.loads(line) for line in completed.stdout.splitlines()]
        assert [row["id"] for row in listed] == expected_ids
        assert all(row["license"] == manifest["assets"][i]["license"] for i, row in enumerate(listed))
    elif "--list" in arguments:
        natural = json.loads(completed.stdout)
        assert natural["files"] == 345 and natural["download_bytes"] == 70648031
        assert natural["unpacked_bytes"] == 94905207
        assert natural["rows"] == {"train": 272, "validation": 52, "test": 66}
        assert natural["audio_rows"] == {"train": 24, "validation": 6, "test": 12}
        assert {row["id"] for row in natural["archives"]} == {"vision", "ocr", "speech"}
        record["natural_list_summary"] = natural

# Execute the actual old CLI selection path while recording, rather than unpacking,
# each selected archive. This establishes what "all" selects without fetching data.
spec = importlib.util.spec_from_file_location("t1_original_fetch_probe", ROOT / "scripts/fetch_training_assets.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
selected, lfs_requests = [], []
def unpack_probe(archive, asset, destination):
    selected.append({"id": asset["id"], "archive": str(Path(archive).relative_to(ROOT)), "output": str(destination)})
    return {"asset": asset["id"], "probe_only": True, "archive_not_unpacked": True}
def lfs_probe(arguments, **kwargs):
    lfs_requests.append(arguments)
    return subprocess.CompletedProcess(arguments, 0)
stdout = io.StringIO()
with patch.object(sys, "argv", ["fetch_training_assets.py", "--asset", "all"]), patch.object(fetch, "unpack_asset", unpack_probe), patch.object(fetch.subprocess, "run", lfs_probe), contextlib.redirect_stdout(stdout):
    fetch.main()
assert [row["id"] for row in selected] == expected_ids
assert all(Path(row["output"]) == ROOT / "data/training" for row in selected)
assert not any("natural-" in row["archive"] for row in selected)
record["old_all_selection_probe"] = {
    "command": "actual fetch_training_assets.main with sys.argv --asset all; unpack_asset and any LFS subprocess instrumented as no-write recorders",
    "selected": selected,
    "lfs_subprocesses_intercepted": lfs_requests,
    "stdout": stdout.getvalue(),
    "scope": "Checks actual argparse/manifest/selection loop; does not assert a real eight-package download or extraction.",
}
natural_manifest = ROOT / "docs/natural-assistant/manifest.json"
record["natural_manifest_sha256"] = hashlib.sha256(natural_manifest.read_bytes()).hexdigest()
record["natural_fetch_sha256"] = sha("scripts/fetch_natural_data.py")
record["result"] = "pass: old eight-package --list/licenses/--asset all remain separate from the three-package stdlib natural-data fetcher"
(ROOT / "docs/technical-reviews/artifacts/tool-choice-dep-t1-natural-assets-update.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": record["result"], "old_all_ids": expected_ids, "natural_archive_groups": [row["id"] for row in record["natural_list_summary"]["archives"]], "readme_sha256": record["readme_sha256"]}, ensure_ascii=False))
