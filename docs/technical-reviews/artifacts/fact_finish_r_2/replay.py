"""R.2 navigation, retained kernel receipts, and independent short CPU examples.

Run from the repository root. This checks existing full-kernel evidence and
executes the named route examples in fresh Python namespaces, not 248 kernels.
"""

import contextlib
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import torch

    from scripts.check_technical_reviews import sections
    from scripts.export_course import checked_notebook

    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}
    inventory = json.loads((out / "route-inventory.json").read_text())
    body = (out / "R.2-read-snapshot.md").read_text()
    assert dict(sections(root / "course/README.md"))["R.2"] == body
    assert hashlib.sha256(body.encode()).hexdigest() == inventory["source_sha256"]
    assert inventory["first_section"] == "R.1"
    links = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        path, _, fragment = target.partition("#")
        source = (root / "course" / path).resolve()
        section = dict(sections(source))[fragment]
        links.append({"target": target, "heading": section.splitlines()[0]})

    cpu_examples = []
    for record in inventory["route_sections"]:
        source = root / record["path"]
        snapshot = root / record["snapshot"]
        text = snapshot.read_text()
        assert dict(sections(source))[record["lesson_id"]] == text
        assert digest(snapshot) == record["section_sha256"]
        namespace = {"__name__": "__main__"}
        torch.set_num_threads(1)
        torch.manual_seed(42)
        stdout = io.StringIO()
        snippets = re.findall(r"```python\n(.*?)```", text, flags=re.S)
        with contextlib.redirect_stdout(stdout):
            for index, snippet in enumerate(snippets):
                exec(compile(snippet, f"{source}#{record['lesson_id']}:block{index}", "exec"), namespace)
        cpu_examples.append(
            {
                "lesson": record["lesson_id"],
                "snapshot_sha256": digest(snapshot),
                "code_blocks": len(snippets),
                "stdout": stdout.getvalue(),
                "status": "passed",
                "is_fresh_kernel": False,
            }
        )

    receipt_path = root / "docs/validation-artifacts/integration-current-notebooks.json"
    current = json.loads(receipt_path.read_text())
    assert current["status"] == "passed" and len(current["records"]) == 248
    shutil.copyfile(receipt_path, out / "current-notebooks-receipt-original.json")
    baseline_path = root / "docs/validation-artifacts/integration-notebooks.json"
    baseline = json.loads(baseline_path.read_text())
    assert baseline["mode"] == "kernel" and baseline["total"] == baseline["passed"] == 248
    assert baseline["failed"] == 0
    shutil.copyfile(baseline_path, out / "full-kernel-receipt-original.json")
    source_receipts = []
    for path, sha in current["source_run_receipts"].items():
        assert digest(root / path) == sha
        receipt = json.loads((root / path).read_text())
        shutil.copyfile(root / path, out / ("retained-" + Path(path).name))
        source_receipts.append({"original_path": path, "sha256": sha, "original_record": receipt})
    route_ids = {r["lesson_id"] for r in inventory["route_sections"]}
    executed_routes = []
    checked_cells = 0
    provenance = []
    for record in current["records"]:
        source = root / record["source"]
        executed = root / record["executed"]
        assert digest(source) == record["source_sha256"]
        assert digest(executed) == record["executed_sha256"]
        original = json.loads(source.read_text())
        run = checked_notebook(original, executed)
        code_cells = sum(c["cell_type"] == "code" for c in run["cells"])
        assert code_cells == record["code_cells"]
        checked_cells += code_cells
        evidence_path = record["execution_evidence"]
        if evidence_path == baseline_path.relative_to(root).as_posix():
            output_key = source.relative_to(root / "notebooks").as_posix()
            assert baseline["source_notebooks"][record["source"]] == record["source_sha256"]
            assert baseline["executed_notebooks"][output_key] == record["executed_sha256"]
        else:
            original_receipt = json.loads((root / evidence_path).read_text())
            if "records" in original_receipt:
                original_receipt = next(r for r in original_receipt["records"] if r["lesson"] == source.stem)
            original_result = original_receipt["report"]
            assert original_result["mode"] == "kernel"
            assert original_result["total"] == original_result["passed"] == 1 and original_result["failed"] == 0
            assert original_receipt["source_sha256"][record["source"]] == record["source_sha256"]
            assert original_receipt["source_sha256"][record["executed"]] == record["executed_sha256"]
        provenance.append({"source": record["source"], "verified_kernel_receipt": evidence_path})
        if source.stem in route_ids:
            executed_routes.append({"original_identity": record, "executed_notebook": run})
    assert len(executed_routes) == len(route_ids - {"W.3"})
    (out / "route-executed-notebooks-original.json").write_text(
        json.dumps(executed_routes, ensure_ascii=False, indent=2) + "\n"
    )

    build = subprocess.run(
        [str(root / ".venv/bin/python"), "scripts/build_course.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    public_list = subprocess.run(
        [str(root / ".venv/bin/python"), "scripts/fetch_capstone.py", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    published = [json.loads(line) for line in public_list.stdout.splitlines()]
    assert any(item["stage"] == "joint" and item["format_version"] == "capstone-v1" for item in published)
    result = {
        "reviewer": "/root/fact_finish_r_2",
        "checked_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "links": links,
        "links_count": len(links),
        "cpu_example_count": len(cpu_examples),
        "cpu_examples": cpu_examples,
        "retained_kernel_audit": {
            "original_receipt_sha256": digest(receipt_path),
            "baseline_mode": baseline["mode"],
            "baseline_command": baseline["command"],
            "source_receipts": source_receipts,
            "all_current_notebooks_checked": len(current["records"]),
            "all_current_code_cells_checked": checked_cells,
            "per_notebook_original_kernel_provenance": provenance,
            "durable_route_execution_copies": len(executed_routes),
            "fresh_kernels_started_by_this_reviewer": 0,
            "scope": "Personally checked original receipts and all current cell source/count/error records; reused prior separately started CPU kernels; no claim of rerunning 248 kernels, Colab, GPU, or training.",
        },
        "build_check": {"returncode": build.returncode, "stdout": build.stdout, "stderr": build.stderr},
        "capstone_list": {"returncode": public_list.returncode, "stdout": public_list.stdout},
        "status": "passed",
    }
    (out / "execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "links": len(links),
                "short_cpu_examples": len(cpu_examples),
                "audited_existing_kernel_notebooks": len(current["records"]),
                "audited_code_cells": checked_cells,
                "retained_route_copies": len(executed_routes),
                "build": build.stdout.strip(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
