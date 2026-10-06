"""Freeze the reference-stage correction and retain actual unchanged-code outputs."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_course import DOCUMENTS, home_introduction
from scripts.reading_time import build_inventory, lesson_slices

root = Path.cwd()
base = root / "docs/course-revision-20261005/continuity"
target = base / "revised-04"
assert not target.exists()
prior_inventory_path = base / "revised-03/inventory.json"
prior_inventory = json.loads(prior_inventory_path.read_text())
prior_pages = {p["page_id"]: p for p in prior_inventory["pages"]}
index = json.loads((root / "course/lesson-index.json").read_text())
inventory = build_inventory(root, index, DOCUMENTS, home_introduction(index, True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source_changes = []
figure_changes = []
for page in inventory["pages"]:
    old = prior_pages[page["page_id"]]
    if page["source_sha256"] != old["source_sha256"]:
        source_changes.append(page["page_id"])
        assert page["page_id"] in {"13.12", "13.14"}
        snapshot = target / "source-pages" / (page["page_id"] + ".md")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(lesson_slices((root / page["source"]).read_text())[page["page_id"]])
        page["snapshot"] = str(snapshot.relative_to(root))
    else:
        page["snapshot"] = old["snapshot"]
    assert sha(root / page["snapshot"]) == page["source_sha256"]
    if page["figures_sha256"] != old["figures_sha256"]:
        figure_changes.append(page["page_id"])
        assert page["page_id"] == "13.14"
        assert set(page["figures_sha256"]) == set(old["figures_sha256"])
        assert set(page["figures_sha256"]) == {"course/figures/rewrite-13-model-roles.svg"}
    for path, digest in page["figures_sha256"].items():
        assert sha(root / path) == digest
assert source_changes == ["13.12", "13.14"]
assert figure_changes == ["13.14"]
(target / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
receipt = {
    "recorded_at": datetime.now(UTC).isoformat(),
    "inventory_path": str((target / "inventory.json").relative_to(root)),
    "inventory_sha256": sha(target / "inventory.json"),
    "prior_inventory_path": str(prior_inventory_path.relative_to(root)),
    "prior_inventory_sha256": sha(prior_inventory_path),
    "source_count": 321,
    "changed_source_pages": source_changes,
    "changed_figure_pages": figure_changes,
    "changed_versions": [
        {
            "page_id": page["page_id"],
            "previous_source_sha256": prior_pages[page["page_id"]]["source_sha256"],
            "current_source_sha256": page["source_sha256"],
            "previous_figures_sha256": prior_pages[page["page_id"]]["figures_sha256"],
            "current_figures_sha256": page["figures_sha256"],
        }
        for page in inventory["pages"] if page["page_id"] in source_changes
    ],
    "snapshot_policy": "Two changed pages receive new snapshots;319 identical-version V3 snapshots remain byte-verified. Only13.14 static diagram reference-stage wording changes.",
    "scope": "Root source/figure freeze only; not reader or scientific acceptance.",
}
(target / "freeze-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

output_base = root / "outputs/course-revision-20261005"
prior = output_base / "cpu-kernels-after-continuity-v3"
destination = output_base / "cpu-kernels-after-continuity-v4"
assert not destination.exists()
prior_receipt_path = prior / "assembly-receipt.json"
prior_receipt = json.loads(prior_receipt_path.read_text())
prior_rows = {r["lesson"]: r for r in prior_receipt["pages"]}
rows = []
for notebook in sorted((root / "notebooks").rglob("*.ipynb")):
    rel = notebook.relative_to(root / "notebooks")
    current = json.loads(notebook.read_text())
    old_path = prior / rel
    assert sha(old_path) == prior_rows[notebook.stem]["assembled_output_sha256"]
    old = json.loads(old_path.read_text())
    assert len(current["cells"]) == len(old["cells"])
    executed = copy.deepcopy(current)
    for cell, previous in zip(executed["cells"], old["cells"], strict=True):
        assert cell["cell_type"] == previous["cell_type"]
        if cell["cell_type"] == "code":
            assert cell["source"] == previous["source"], notebook.stem
            cell["outputs"] = copy.deepcopy(previous.get("outputs", []))
            cell["execution_count"] = previous["execution_count"]
            if "".join(cell["source"]).strip():
                assert cell["execution_count"] is not None
                assert not any(o["output_type"] == "error" for o in cell["outputs"])
    output = destination / rel
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(executed, ensure_ascii=False, indent=1) + "\n")
    rows.append({
        "lesson": notebook.stem,
        "notebook_sha256": sha(notebook),
        "actual_execution_origin": str(old_path.relative_to(root)),
        "actual_execution_origin_sha256": sha(old_path),
        "assembled_output": str(output.relative_to(root)),
        "assembled_output_sha256": sha(output),
        "mode": "reused_actual_identical_code_outputs_with_current_markdown",
        "static_diagram_wording_changed": notebook.stem == "13.14",
    })
assert len(rows) == 283
assembly = {
    "recorded_at": datetime.now(UTC).isoformat(), "total": 283,
    "new_independent_kernels": 0, "reused_actual_prior_outputs": 283,
    "code_changed": [], "static_diagram_wording_changed": ["13.14"],
    "prior_assembly_receipt": str(prior_receipt_path.relative_to(root)),
    "prior_assembly_receipt_sha256": sha(prior_receipt_path), "pages": rows,
    "scope": "No new CPU/GPU run. All283 code-cell sources match actual prior executions. Current13.12/13.14 prose and13.14 static diagram wording are separately reviewed; no claim that its SVG bytes stayed identical. All prior aggregates remain preserved.",
}
(destination / "assembly-receipt.json").write_text(json.dumps(assembly, ensure_ascii=False, indent=2) + "\n")
permanent = root / "docs/course-revision-20261005/verification/cpu-continuity-v4"
permanent.mkdir(parents=True, exist_ok=True)
(permanent / "assembly-receipt.json").write_bytes((destination / "assembly-receipt.json").read_bytes())
(permanent / "freeze_and_stitch_v4.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt, ensure_ascii=False))
print("Actual283 prior identical-code outputs reused;0 new kernels/GPU;13.14 diagram wording explicitly changed.")
