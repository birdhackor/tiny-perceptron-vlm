"""Rebuild W.1's output/download dependencies with the unmodified exporter outside Git."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
ISOLATED = Path("/workspace/work/omni-w1-current-export-v2")
BUNDLE = ROOT / "outputs/omni-notebooks-technical-fix06-current"
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, checked_notebook, output_html


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started_at = datetime.now(UTC).isoformat()
assert not ISOLATED.exists(), "Use a new isolated directory; do not overwrite a prior export."
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
receipt_path = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix06-cumulative.json"
author_receipt = json.loads(receipt_path.read_bytes())
assert author_receipt["total"] == len(index) == 331
records = {r["lesson"]: r for r in author_receipt["records"]}
assert len(records) == 331 and set(records) == {item["id"] for item in index}
assert author_receipt["source_freeze"]["sha256"] == digest(ROOT / author_receipt["source_freeze"]["path"])
assert digest(Path(author_receipt["script"]["path"])) == author_receipt["script"]["sha256"]
run_validation_path = ROOT / "outputs/omni-notebooks-technical-fix06-run/validation-kernel.json"
run_validation = json.loads(run_validation_path.read_bytes())
assert run_validation["total"] == run_validation["passed"] == 1
assert run_validation["failed"] == 0 and run_validation["failures"] == []
verified, stale_shared_downloads = [], []
teaching_cells = displayed_cells = code_sections = 0
for item in index:
    record = records[item["id"]]
    notebook_path = ROOT / item["notebook"]
    target = BUNDLE / Path(item["notebook"]).relative_to("notebooks")
    source = Path(record["source"])
    assert Path(record["bundle_path"]) == target
    assert digest(notebook_path) == record["notebook_sha256"]
    assert digest(target) == record["executed_sha256"]
    assert source.read_bytes() == target.read_bytes()
    actual = json.loads(notebook_path.read_bytes())
    executed = checked_notebook(actual, target)
    code = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    count = sum(bool(output_html(c.get("outputs", []))) for c in code)
    if code:
        assert count > 0, item["id"]
        code_sections += 1
    teaching_cells += len(code)
    displayed_cells += count
    shared_copy = ROOT / "outputs/omni-review-site-current" / item["notebook"]
    if shared_copy.read_bytes() != notebook_path.read_bytes():
        stale_shared_downloads.append({"lesson_id": item["id"], "path": str(shared_copy), "sha256": digest(shared_copy), "current_notebook_sha256": digest(notebook_path)})
    verified.append({"lesson_id": item["id"], "notebook_sha256": digest(notebook_path), "bundle_sha256": digest(target), "method": record["method"]})

# Real copies of only the exporter inputs; no symlink can route output into the
# shared repo. Exporter and helper source files remain byte-for-byte unchanged.
ISOLATED.mkdir()
for folder in ["course", "notebooks"]:
    shutil.copytree(ROOT / folder, ISOLATED / folder)
for relative in ["scripts/export_course.py", "scripts/build_course.py", "scripts/reading_time.py", "scripts/stage_mathjax.py", "zensical.toml", *DOCUMENTS.values()]:
    target = ISOLATED / relative
    if target.exists():
        assert target.read_bytes() == (ROOT / relative).read_bytes()
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)
mathjax_source = ROOT / "outputs/mathjax-cache"
if mathjax_source.is_dir():
    shutil.copytree(mathjax_source, ISOLATED / "outputs/mathjax-cache")
source_inputs = [{"path": str(p.relative_to(ISOLATED)), "sha256": digest(p)} for p in ISOLATED.rglob("*") if p.is_file() and "outputs/" not in str(p.relative_to(ISOLATED))]
commands = [sys.executable, str(ISOLATED / "scripts/export_course.py"), "--output", str(ISOLATED / "outputs/site"), "--executed", str(BUNDLE), "--revision", "W1-current-v5", "--reading-times", str(ISOLATED / "intentionally-absent-reading-times.json")]
completed = subprocess.run(commands, cwd=ISOLATED, text=True, capture_output=True, timeout=120)
result = {
    "started_at": started_at,
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_w_1",
    "command": commands,
    "cwd": str(ISOLATED),
    "exit_code": completed.returncode,
    "stdout": completed.stdout,
    "stderr": completed.stderr,
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "zensical": importlib.metadata.version("zensical"), "torch": importlib.metadata.version("torch"), "device": "cpu"},
    "author_receipt": {"path": str(receipt_path.relative_to(ROOT)), "sha256": digest(receipt_path)},
    "root_kernel_validation": {"path": str(run_validation_path.relative_to(ROOT)), "sha256": digest(run_validation_path), "result": run_validation},
    "bundle_root": str(BUNDLE),
    "bundle_verified": len(verified),
    "code_sections": code_sections,
    "teaching_code_cells": teaching_cells,
    "displayed_cells": displayed_cells,
    "bundle_records": verified,
    "stale_shared_downloads_preserved": stale_shared_downloads,
    "isolated_root": str(ISOLATED),
    "copied_input_fingerprint": hashlib.sha256(json.dumps(source_inputs, sort_keys=True).encode()).hexdigest(),
    "limitations": ["Only output/Notebook/download/footer dependencies are tested; no whole-page or W.2-W.7 technical review.", "Reading times intentionally omitted via the exporter's supported optional missing-file path, without changing any tool; this export does not validate reading-time evidence.", "Shared staged site remains unchanged; the isolated build is inspectable evidence, not deployment or live website update.", "No browser visual or figure inspection; no PNG or whole-site contents are placed in Git."],
}
if completed.returncode == 0:
    site = ISOLATED / "outputs/site"
    downloads = []
    for item in index:
        p = site / item["notebook"]
        assert p.read_bytes() == (ROOT / item["notebook"]).read_bytes(), item["id"]
        shared = ROOT / "outputs/omni-review-site-current" / item["notebook"]
        if item["id"] in {r["lesson_id"] for r in stale_shared_downloads}:
            assert digest(shared) == next(r["sha256"] for r in stale_shared_downloads if r["lesson_id"] == item["id"])
        downloads.append({"lesson_id": item["id"], "sha256": digest(p)})
    info = json.loads((site / "build-info.json").read_bytes())
    assert info["executed_cpu_outputs"] and info["lessons"] == 331
    page911 = (site / "9.11.html").read_text()
    assert "查卡片進度 0.844" in page911 and "更改地址 0.042" in page911 and "以上皆不適合 0.114" in page911
    assert 'href="notebooks/09/9.11.ipynb"' in page911
    assert "/blob/W1-current-v5/notebooks/09/9.11.ipynb" in page911
    result["isolated_downloads_identical"] = len(downloads)
    result["isolated_download_manifest_sha256"] = hashlib.sha256(json.dumps(downloads, sort_keys=True).encode()).hexdigest()
    result["build_info"] = info
    result["isolated_9_11"] = {"html_path": str(site / "9.11.html"), "html_sha256": digest(site / "9.11.html"), "download_path": str(site / "notebooks/09/9.11.ipynb"), "download_sha256": digest(site / "notebooks/09/9.11.ipynb"), "expected_cpu_output_present": True, "download_and_colab_links_present": True}
target = ARTIFACTS / "isolated-export-current-v2.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"receipt": str(target.relative_to(ROOT)), "sha256": digest(target), "exit_code": completed.returncode, "bundle_verified": len(verified), "stale_shared_downloads": stale_shared_downloads, "isolated_downloads_identical": result.get("isolated_downloads_identical")}, ensure_ascii=False, indent=2))
if completed.returncode:
    print(completed.stderr)
    raise SystemExit(completed.returncode)
