"""Author operations after all six initial reader reports; never a review verdict."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

BASE = ROOT / "docs/course-revision-20261007-phase7"
DRAFT = ROOT / "outputs/phase7-proposed-fixes"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_reports():
    freeze = BASE / "reviews/freeze-01"
    manifest = json.loads((freeze / "manifest.json").read_text())
    groups = {item["group"]: item for item in manifest["groups"]}
    rows = [json.loads(line) for line in (freeze / "collections.jsonl").read_text().splitlines()]
    found = {}
    for row in rows:
        if row["stage"] != "reader":
            continue
        path = ROOT / row["path"]
        assert digest(path) == row["sha256"], "Collected report bytes changed"
        report = json.loads(path.read_text())
        assert [item["page_id"] for item in report["pages"]] == groups[row["group"]]["primary_page_ids"]
        assert report["reviewer_task"] == row["reviewer_task"]
        assert report["verdict"] in {"pass", "revise"}
        found[row["group"]] = {"path": row["path"], "sha256": row["sha256"]}
    assert set(found) == set(groups), "Wait for all six genuine full initial reports"
    return found


def apply():
    reports = initial_reports()
    drafts = json.loads((DRAFT / "drafts.json").read_text())
    prepared, sections = [], []
    for name, binding in drafts["files"].items():
        source, proposal = ROOT / name, DRAFT / name
        assert digest(source) == binding["source_sha256"], name
        assert digest(proposal) == binding["draft_sha256"], name
        old, new = source.read_text(), proposal.read_text()
        assert re.findall(r"```[^\n]*\n.*?\n```", old, re.S) == re.findall(r"```[^\n]*\n.*?\n```", new, re.S), name
        before, after = lesson_slices(old), lesson_slices(new)
        assert before.keys() == after.keys(), name
        sections.extend(pid for pid in before if before[pid] != after[pid])
        prepared.append((name, proposal, source, binding))
    figure = DRAFT / "course/figures/p7-12-frequency-match.svg"
    figure_receipt = json.loads((BASE / "authoring/frequency-figure-check.json").read_text())
    assert digest(figure) == figure_receipt["source_sha256"]
    figure_target = ROOT / "course/figures/p7-12-frequency-match.svg"
    assert not figure_target.exists(), "Repair operations already applied or destination exists"
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    selected = [item for item in index if item["id"] in sections]
    archive = ROOT / "outputs/phase7-before-reader-repairs"
    assert not archive.exists(), "Preserve previous run; do not overwrite it"
    archive.mkdir()
    for item in selected:
        executed = ROOT / "outputs/notebooks" / Path(item["notebook"]).relative_to("notebooks")
        if executed.exists():
            target = archive / Path(item["notebook"]).relative_to("notebooks")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(executed, target)
    for _, proposal, source, _ in prepared:
        shutil.copyfile(proposal, source)
    shutil.copyfile(figure, figure_target)
    record = {
        "kind": "author applies readability repairs, not independent rechecks",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "initial_full_reports": reports,
        "code_blocks_unchanged": True,
        "files": drafts["files"],
        "changed_numbered_sections": sections,
        "kernel_lessons": [item["id"] for item in selected],
        "new_figure": {"path": str(figure_target.relative_to(ROOT)), "sha256": digest(figure_target)},
        "prior_executed_archive": str(archive.relative_to(ROOT)),
    }
    (BASE / "authoring/reader-repairs-applied.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"applied": len(prepared), "kernel_lessons": record["kernel_lessons"]}, ensure_ascii=False), flush=True)


def kernels():
    from scripts.check_notebooks import run_kernel
    from scripts.export_course import checked_notebook
    import torch

    applied = json.loads((BASE / "authoring/reader-repairs-applied.json").read_text())
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    selected = [item for item in index if item["id"] in applied["kernel_lessons"]]
    output, rows, failures = ROOT / "outputs/notebooks", [], []
    for item in selected:
        notebook = ROOT / item["notebook"]
        started = time.perf_counter()
        try:
            executed_path = Path(run_kernel(notebook, output, 120))
            executed = json.loads(executed_path.read_text())
            checked_notebook(json.loads(notebook.read_text()), executed_path)
            row = {"lesson": item["id"], "status": "executed_without_errors_and_source_matched", "source_notebook_sha256": digest(notebook), "executed_path": str(executed_path.relative_to(ROOT)), "executed_sha256": digest(executed_path), "python_version_from_notebook": executed["metadata"]["language_info"]["version"], "torch_version_in_kernel_venv": torch.__version__}
        except Exception as error:
            row = {"lesson": item["id"], "status": "failed", "error": repr(error)}
            failures.append(item["id"])
        row["seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        record = {"kind": "author fresh CPU kernels, not a student or technical verdict", "recorded_at": datetime.now(timezone.utc).isoformat(), "command": ".venv/bin/python docs/course-revision-20261007-phase7/authoring/reader-repair-operations.py kernels", "device": "CPU", "records": rows, "planned": len(selected), "completed": len(rows), "failures": failures}
        (BASE / "authoring/reader-repair-kernel-checks.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "kernels"))
    args = parser.parse_args()
    {"apply": apply, "kernels": kernels}[args.action]()
