"""Audit global source/notebook/output presence, without executing the whole course.

This does not decide whether prose is understandable or sufficient. That judgment
is recorded separately after the reviewer reads selected primary text and notes.
"""

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
import html
import json
from pathlib import Path
import platform
import re
import sys


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
CATALOGUE = ROOT / "docs/course-revision-20261029-omni/reviews/reader-evidence-catalogue-technical-fix07-v1.json"
EXECUTED = ROOT / "outputs/omni-notebooks-technical-fix06-current"
SITE = ROOT / "outputs/omni-review-site-technical-fix07"
freeze = json.loads(FREEZE.read_text())
catalogue = json.loads(CATALOGUE.read_text())
lessons = [p for p in freeze["pages"] if p["kind"] == "lesson"]
errors = []
totals = Counter()
no_python = []
outputless_teaching = []
external_import_pages = []


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pointer(data, value):
    for part in value.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def joined(value):
    return "".join(value) if isinstance(value, list) else value


for page in lessons:
    pid = page["page_id"]
    snapshot = ROOT / page["snapshot"]
    if digest(snapshot) != page["source_sha256"]:
        errors.append([pid, "frozen source hash mismatch"])
    raw = snapshot.read_text()
    blocks = [c.strip() for c in re.findall(r"```python\n(.*?)```", raw, re.S)]
    nbpath = ROOT / page["notebook"]
    if digest(nbpath) != page["notebook_sha256"]:
        errors.append([pid, "notebook hash mismatch"])
    notebook = json.loads(nbpath.read_text())
    teaching = [c for c in notebook["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and not c.get("metadata", {}).get("course_figure")]
    if [joined(c["source"]).strip() for c in teaching] != blocks:
        errors.append([pid, "teaching code differs from complete source fences"])
    if not blocks:
        no_python.append(pid)
    if re.search(r"(?:from|import) tiny_perceptron", "\n".join(blocks)):
        external_import_pages.append(pid)
    totals["lessons"] += 1
    totals["python_fences_in_source"] += len(blocks)
    totals["teaching_notebook_cells"] += len(teaching)
    expath = EXECUTED / Path(page["notebook"]).relative_to("notebooks")
    executed = json.loads(expath.read_text())
    source_cells = [(c["cell_type"], joined(c["source"])) for c in notebook["cells"]]
    executed_cells = [(c["cell_type"], joined(c["source"])) for c in executed["cells"]]
    if source_cells != executed_cells:
        errors.append([pid, "recorded executed notebook is not exact current notebook"])
    sitepath = SITE / (pid + ".html")
    rendered = html.unescape(sitepath.read_text())
    totals["existing_site_lesson_pages"] += 1
    has_output = False
    for i, cell in enumerate(executed["cells"]):
        if cell["cell_type"] != "code":
            continue
        totals["recorded_executed_code_cells"] += 1
        if cell.get("execution_count") is None:
            errors.append([pid, i, "recorded code cell not executed"])
        outputs = cell.get("outputs", [])
        if any(o["output_type"] == "error" for o in outputs):
            errors.append([pid, i, "recorded code cell error"])
        is_teaching = not cell.get("metadata", {}).get("course_setup") and not cell.get("metadata", {}).get("course_figure")
        if is_teaching and not outputs:
            outputless_teaching.append([pid, i])
        for output in outputs:
            text = None
            if output["output_type"] == "stream":
                text = joined(output["text"])
            elif "text/plain" in output.get("data", {}):
                text = joined(output["data"]["text/plain"])
            if is_teaching and text:
                has_output = True
                totals["teaching_text_outputs_checked_against_html"] += 1
                if text not in rendered:
                    errors.append([pid, i, "recorded text output absent from rendered HTML"])
    if has_output:
        totals["lesson_pages_with_teaching_text_output"] += 1
        if "實際執行結果 · CPU" not in rendered:
            errors.append([pid, "CPU output label absent"])

catalogue_status = Counter()
catalogue_identity_mismatches = []
for row in catalogue["rows"]:
    catalogue_status[row["status"]] += 1
    if row["status"] == "preserved_inherit_not_new_read":
        continue
    for ref in [row["current_report"], row["basis"]["report"]]:
        if digest(ROOT / ref["path"]) != ref["sha256"]:
            errors.append([row["page_id"], "reader reference file hash mismatch", ref["path"]])
    current = pointer(json.loads((ROOT / row["current_report"]["path"]).read_text()), row["current_pointer"])
    if row["status"] == "current_owner_source_identity" and current.get("source_sha256") != row["current_source_sha256"]:
        catalogue_identity_mismatches.append(row["page_id"])

old_steps_path = ROOT / "docs/course-review-20261019/freeze-02/sources/first-steps.md"
new_steps_page = next(p for p in freeze["pages"] if p["page_id"] == "first-steps")
new_steps_path = ROOT / new_steps_page["snapshot"]
fence_pattern = r"```[^\n]*\n.*?```"
old_blocks = re.findall(fence_pattern, old_steps_path.read_text(), re.S)
new_blocks = re.findall(fence_pattern, new_steps_path.read_text(), re.S)
assert old_blocks == new_blocks
assert not catalogue_identity_mismatches
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/audit_course_presence_callback_v1.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; static source/output inspection; no lesson or training execution"},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "catalogue": {"path": str(CATALOGUE.relative_to(ROOT)), "sha256": digest(CATALOGUE)},
    "recorded_executed_root": str(EXECUTED.relative_to(ROOT)),
    "candidate_site_root": str(SITE.relative_to(ROOT)),
    "counts": dict(totals),
    "catalogue_status_counts_recomputed": dict(catalogue_status),
    "catalogue_current_report_identity_mismatches": catalogue_identity_mismatches,
    "source_pages_without_python_fences": no_python,
    "teaching_cells_without_stored_output": outputless_teaching,
    "pages_with_project_imports_not_a_missing_code_judgment": external_import_pages,
    "first_steps_exception": {
        "old_snapshot": str(old_steps_path.relative_to(ROOT)),
        "old_sha256": digest(old_steps_path),
        "current_snapshot": str(new_steps_path.relative_to(ROOT)),
        "current_sha256": digest(new_steps_path),
        "all_fenced_code_bytes_unchanged": True,
        "fenced_code_block_count": len(new_blocks),
        "new_complete_reader_status": "pending; this program does not establish prose sufficiency",
    },
    "errors": errors,
    "scope": "Exhaustive presence/identity check for all 331 lesson source Python fences, current Notebook teaching cells, existing executed copies, and candidate HTML outputs. Existing execution records are inspected, not rerun or attributed to this reviewer. Counts and hashes do not establish semantic sufficiency; selected original reader notes and actual prose are read separately. No whole-book reread, publication proof, or current first-steps reader pass is claimed.",
}
out = Path(__file__).with_name("course-presence-callback-v1.json")
with out.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
