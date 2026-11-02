"""Same-owner W.1 global dependency callback for numbered v9 / canonical fix11.

Only small summaries/fingerprints are saved in Git. The site and complete CLI
logs remain in the new external isolated directory.
"""

import hashlib
import html
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
D = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_1"
F = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v9.json"
C = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
RECEIPT = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix11-cumulative.json"
BUNDLE = ROOT / "outputs/omni-notebooks-technical-fix11-current"
ISOLATED = Path("/workspace/work/omni-w1-fix11-export-v1")
sys.path.insert(0, str(ROOT))
from scripts.build_course import BOOTSTRAP, notebook
from scripts.export_course import DOCUMENTS, checked_notebook, output_html, section_footer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


started = datetime.now(UTC).isoformat()
assert not ISOLATED.exists(), "Never overwrite a previous isolated export."
original_path, prior_path = D / "technical-original-v1.json", D / "technical-current-v1.json"
assert sha(original_path) == "18525e63a0d34a51e9acc1eab7938842138b672ba70f33b03981ceb7cebe5175"
assert sha(prior_path) == "073a13f59756d328708b59af1dbf5da8c13f0a7495780b0fa41a5c80eb89d5e0"
prior = json.loads(prior_path.read_bytes())
assert prior["issues"][0]["id"] == "w1-current-dependency-drift"
preserved = [ref(original_path), ref(prior_path), ref(D / "recheck-observation-v1.json")]
for artifact in prior["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
sources = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        actual = sha(ROOT / source["path"])
        assert actual == source["sha256"], source["path"]
        sources.append({"source_id": source["id"], "path": source["path"], "sha256": actual, "identical_to_previous_inspected_source": True})
official = json.loads((A / "official-source-inspection.json").read_bytes())
for source in official:
    assert sha(Path(source["raw_path"])) == source["raw_sha256"]

freeze, canonical = json.loads(F.read_bytes()), json.loads(C.read_bytes())
assert freeze["canonical_source_freeze"] == ref(C)
page = next(p for p in freeze["pages"] if p["page_id"] == "W.1")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"] == prior["source_sha256"]
assert page["figures_sha256"] == prior["figure_sha256"] == {}
old_f = json.loads((ROOT / prior["source_freeze"]["path"]).read_bytes())
old_w1 = next(p for p in old_f["pages"] if p["page_id"] == "W.1")
assert raw == (ROOT / old_w1["snapshot"]).read_bytes()
parent = next(p for p in canonical["pages"] if p["page_id"] == "first-steps")
parent_bytes = (ROOT / parent["snapshot"]).read_bytes()
assert hashlib.sha256(parent_bytes).hexdigest() == parent["source_sha256"]
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", parent_bytes, re.M | re.S)[0] == raw
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ROOT / "course/first-steps.md").read_bytes(), re.M | re.S)[0] == raw

author = json.loads(RECEIPT.read_bytes())
assert author["source_freeze"] == ref(C)
assert sha(Path(author["script"]["path"])) == author["script"]["sha256"]
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
rows = {r["lesson"]: r for r in author["records"]}
notebook_pages = {p["page_id"]: p for p in canonical["pages"] if p.get("notebook")}
assert len(index) == len(rows) == len(notebook_pages) == author["total"] == 331
assert set(rows) == set(notebook_pages) == {r["id"] for r in index}
old_export = json.loads((A / "isolated-export-current-v2.json").read_bytes())
old_rows = {r["lesson_id"]: r for r in old_export["bundle_records"]}
verified, changed, stale_shared, special = [], [], [], []
teaching_count = display_count = code_sections = 0
expected_output_counts = {}
for item in index:
    row = rows[item["id"]]
    path = ROOT / item["notebook"]
    bundle = BUNDLE / Path(item["notebook"]).relative_to("notebooks")
    assert Path(row["bundle_path"]) == bundle
    assert sha(path) == row["notebook_sha256"] == notebook_pages[item["id"]]["notebook_sha256"]
    assert sha(bundle) == row["executed_sha256"]
    assert Path(row["source"]).read_bytes() == bundle.read_bytes()
    current = json.loads(path.read_bytes())
    executed = checked_notebook(current, bundle)
    code = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    displayed = sum(bool(output_html(c.get("outputs", []))) for c in code)
    if code:
        assert displayed > 0, item["id"]
        code_sections += 1
    teaching_count += len(code)
    display_count += displayed
    expected_output_counts[item["id"]] = displayed
    verified.append({"lesson_id": item["id"], "notebook": item["notebook"], "notebook_sha256": sha(path), "bundle_sha256": sha(bundle), "upstream_sha256": sha(Path(row["source"])), "displayed_teaching_cells": displayed})
    if old_rows[item["id"]]["notebook_sha256"] != sha(path):
        changed.append({"lesson_id": item["id"], "prior_notebook_sha256": old_rows[item["id"]]["notebook_sha256"], "current_notebook_sha256": sha(path)})
    shared = ROOT / "outputs/omni-review-site-current" / item["notebook"]
    if shared.read_bytes() != path.read_bytes():
        stale_shared.append({"lesson_id": item["id"], **ref(shared), "current_notebook_sha256": sha(path)})
    if item["id"] in {"9.11", "9.18", "12.17"}:
        texts = [{"cell": i, "execution_count": c.get("execution_count"), "source_sha256": hashlib.sha256("".join(c["source"]).encode()).hexdigest(), "stdout": "".join("".join(o["text"]) if isinstance(o["text"], list) else o["text"] for o in c.get("outputs", []) if o["output_type"] == "stream")} for i, c in enumerate(executed["cells"]) if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
        special.append({"lesson_id": item["id"], "bundle": ref(bundle), "current_notebook": ref(path), "actual_text_outputs": texts})

# Check the preserved true-kernel lineage, not just the newest author's summary.
lineages = []
for fix, lesson in [("06", "9.11"), ("09", "9.18"), ("11", "12.17")]:
    rp = ROOT / f"docs/course-revision-20261029-omni/evidence/notebooks-technical-fix{fix}-cumulative.json"
    r = json.loads(rp.read_bytes())
    record = next(x for x in r["records"] if x["lesson"] == lesson)
    vp = ROOT / f"outputs/omni-notebooks-technical-fix{fix}-run/validation-kernel.json"
    validation = json.loads(vp.read_bytes())
    assert validation["total"] == validation["passed"] == 1 and validation["failed"] == 0 and not validation["failures"]
    assert record["method"] == "new_independent_kernel"
    assert sha(Path(record["source"])) == record["executed_sha256"]
    assert Path(record["source"]).read_bytes() == Path(rows[lesson]["bundle_path"]).read_bytes()
    lineages.append({"lesson_id": lesson, "author_receipt": ref(rp), "new_kernel_source": ref(Path(record["source"])), "validation": ref(vp), "result": {"total": 1, "passed": 1, "failed": 0}, "latest_bundle_retains_the_same_raw_bytes": True})

environment = {"python": platform.python_version(), "platform": platform.platform(), **{p: importlib.metadata.version(p) for p in ["torch", "jupyterlab", "ipykernel", "nbclient", "zensical"]}, "device": "cpu"}
import torch
assert torch.version.cuda is None
check_env = subprocess.run([sys.executable, str(ROOT / "scripts/check_env.py")], cwd=ROOT, text=True, capture_output=True, timeout=30)
assert check_env.returncode == 0 and "compute         OK（cpu）" in check_env.stdout and "未安裝" not in check_env.stdout
nb11 = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_bytes())
assert "".join(next(c for c in nb11["cells"] if c.get("metadata", {}).get("course_setup"))["source"]) == BOOTSTRAP.rstrip("\n")
chapter = (ROOT / "course/chapters/01.md").read_text()
section = re.search(r"^## 1\.1 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
inline = re.search(r"```python\n(.*?)```", section, re.S)[1]
exercise = []
for text in ["貓看狗，狗看貓。", "鳥看狗，狗看鳥。"]:
    namespace = {}
    import contextlib, io
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(inline.replace("貓看狗，狗看貓。", text), "actual-1.1-inline", "exec"), namespace)
    assert namespace["restored"] == text
    exercise.append({"input": text, "stdout": captured.getvalue()})
footer_cases = []
for label, value, outputs in [("python_outputs", nb11, True), ("python_no_outputs", nb11, False), ("terminal", notebook("X.1", "terminal", "```bash\npython scripts/check_env.py\n```", ROOT / "course/first-steps.md"), True), ("prose", notebook("X.1", "prose", "text", ROOT / "course/first-steps.md"), True)]:
    rendered = section_footer({"notebook": "notebooks/01/1.1.ipynb"}, value, "W1-fix11-test", outputs)
    assert ("CPU 實際輸出" in rendered) == (label == "python_outputs")
    if label in {"terminal", "prose"}:
        assert "本節沒有要在 Notebook 執行的 Python 程式" in rendered
    footer_cases.append({"case": label, "sha256": hashlib.sha256(rendered.encode()).hexdigest(), "passed": True})

ISOLATED.mkdir()
for folder in ["course", "notebooks"]:
    shutil.copytree(ROOT / folder, ISOLATED / folder)
for relative in ["scripts/export_course.py", "scripts/build_course.py", "scripts/reading_time.py", "scripts/stage_mathjax.py", "zensical.toml", *DOCUMENTS.values()]:
    target = ISOLATED / relative
    if target.exists():
        assert target.read_bytes() == (ROOT / relative).read_bytes()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
if (ROOT / "outputs/mathjax-cache").is_dir():
    shutil.copytree(ROOT / "outputs/mathjax-cache", ISOLATED / "outputs/mathjax-cache")
command = [sys.executable, str(ISOLATED / "scripts/export_course.py"), "--output", str(ISOLATED / "outputs/site"), "--executed", str(BUNDLE), "--revision", "W1-fix11-test", "--reading-times", str(ISOLATED / "intentionally-absent-reading-times.json")]
exported = subprocess.run(command, cwd=ISOLATED, text=True, capture_output=True, timeout=120)
log = ISOLATED / "export-cli.txt"
log.write_text("STDOUT\n" + exported.stdout + "\nSTDERR\n" + exported.stderr)
env_log = ISOLATED / "check-env-cli.txt"
env_log.write_text(check_env.stdout + "\nSTDERR\n" + check_env.stderr)
assert exported.returncode == 0, str(log)
downloads = []
for item in index:
    source = ROOT / item["notebook"]
    copied = ISOLATED / "outputs/site" / item["notebook"]
    assert copied.read_bytes() == source.read_bytes()
    assert (ISOLATED / item["notebook"]).read_bytes() == source.read_bytes()
    body = (ISOLATED / "outputs/site" / (item["id"] + ".html")).read_text()
    assert f'href="{item["notebook"]}"' in body
    assert f'/blob/W1-fix11-test/{item["notebook"]}' in body
    if expected_output_counts[item["id"]]:
        assert "實際執行結果 · CPU" in body
    downloads.append({"lesson_id": item["id"], "download_sha256": sha(copied)})
for sample in special:
    body = (ISOLATED / "outputs/site" / (sample["lesson_id"] + ".html")).read_text()
    for output in sample["actual_text_outputs"]:
        for line in output["stdout"].splitlines():
            assert html.escape(line) in body, (sample["lesson_id"], line)
    sample["exported_html"] = ref(ISOLATED / "outputs/site" / (sample["lesson_id"] + ".html"))
for row in stale_shared:
    assert sha(ROOT / row["path"]) == row["sha256"], "Never change the shared stage."
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ISOLATED / "course/first-steps.md").read_bytes(), re.M | re.S)[0] == raw
info = json.loads((ISOLATED / "outputs/site/build-info.json").read_bytes())
assert info["lessons"] == 331 and info["executed_cpu_outputs"]
claim_status = [{"claim_id": c["id"], "status": "verified", "basis": "W.1 exact bytes and original official/repo evidence preserved; current global software dependencies independently checked against fix11. Original platform/quality limitations retained."} for c in prior["claims"]]
assert len(claim_status) == 33
result = {
    "started_at": started, "created_at": datetime.now(UTC).isoformat(), "reviewer_task": prior["reviewer_task"], "lesson_id": "W.1", "verdict": "pass",
    "original_and_prior_reports_preserved": preserved, "original_current_issue_id_preserved": prior["issues"][0]["id"],
    "source_freeze": ref(F), "canonical_source_freeze": ref(C), "w1_snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]}, "w1_exact_bytes_same_as_current_v1": True, "figure_sha256": {},
    "direct_dependencies": sources, "official_original_versions_preserved": len(official), "environment": environment,
    "author_bundle_receipt": ref(RECEIPT), "bundle_root": str(BUNDLE), "all331_manifest_source_target_and_hash_checked": True,
    "counts": {"canonical_notebooks": 331, "matched_executed_notebooks": len(verified), "teaching_code_sections": code_sections, "teaching_code_cells": teaching_count, "cells_with_displayed_output": display_count, "isolated_matching_downloads": len(downloads)},
    "all331_verified_rows_fingerprint": fingerprint(verified), "all331_download_fingerprint": fingerprint(downloads), "changed_notebooks_since_current_v1": changed,
    "kernel_lineage": lineages, "three_changed_lineage_outputs_inspected": special, "exercise_reexecuted": exercise, "check_env": {"exit_code": 0, "compute": "OK(cpu)", "complete_cli_log": ref(env_log)}, "footer_branch_checks": footer_cases,
    "isolated_export": {"root": str(ISOLATED), "command": command, "exit_code": exported.returncode, "full_cli_log": ref(log), "build_info": info, "input_exporter_sha256": sha(ISOLATED / "scripts/export_course.py"), "input_bootstrap_generator_sha256": sha(ISOLATED / "scripts/build_course.py")},
    "shared_stage_not_modified_stale_downloads": stale_shared, "claims_rechecked": claim_status,
    "issue_continuation": {"id": "w1-current-dependency-drift", "status": "resolved", "note": "Retain the complete original current-v1 issue and failure observations. Current proof is the genuine fix11 bundle and new isolated export; shared obsolete downloads are neither used as proof nor claimed updated."},
    "limitations": ["Only W.1 and its global Notebook/output/download/footer dependencies are reviewed; no fresh technical review of W.2-W.7 or other lesson subject matter.", "331 raw copies were individually checked; this reviewer did not freshly execute331 kernels. The genuine9.11/9.18/12.17 kernel lineage is inspected and preserved.", "No new figure visual inspection, Windows/macOS/Colab run, training-quality/GPU claim, or deployment. No new whole-first-steps reader proof or overall seal is claimed.", "Reading times are intentionally outside this dependency export through the existing optional CLI path. The test revision is a link-template label, not a remotely tested Colab ref.", "Complete CLI and full site remain outside Git; only small output/configuration/fingerprint receipts are retained here."]
}
target = A / "fix11-current-recheck-v1.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"receipt": str(target.relative_to(ROOT)), "sha256": sha(target), "created_at": result["created_at"], "verdict": result["verdict"], "counts": result["counts"], "changed_notebooks_since_current_v1": changed, "shared_stage_stale_ids": [r["lesson_id"] for r in stale_shared]}, ensure_ascii=False, indent=2))
