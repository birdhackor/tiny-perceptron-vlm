"""Bounded read-only evidence checks for the course guide; no training/downloads."""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SITE = ROOT / "outputs/tokenizer-20261027/site"
EXECUTED = Path("/workspace/work/tokenizer-dynamic-20261027/notebooks")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def section(path, identifier):
    text = path.read_text()
    match = re.search(r"^## " + re.escape(identifier) + r" .+$", text, re.M)
    assert match, identifier
    end = re.search(r"^## ", text[match.end():], re.M)
    return text[match.start():match.end() + end.start() if end else len(text)]


def run_blocks(text, first_only=False):
    codes = re.findall(r"```python\n(.*?)```", text, re.S)
    if first_only:
        codes = codes[:1]
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for code in codes:
            exec(compile(code, "reviewed-canonical-example", "exec"), namespace)
    return output.getvalue(), namespace


def main():
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    freeze_path = ROOT / "docs/course-review-20261027-tokenizer/freeze-01.json"
    freeze = json.loads(freeze_path.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "course")
    guide = ROOT / page["source"]
    assert guide.read_bytes() == (ROOT / page["snapshot"]).read_bytes()
    assert sha(guide) == page["source_sha256"]
    assert page["figures_sha256"] == {}
    text = guide.read_text()
    links = []
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if "://" in target:
            continue
        path_text, _, anchor = target.partition("#")
        path = (guide.parent / path_text).resolve() if path_text else guide
        assert path.is_file(), target
        if anchor:
            assert re.search(r"^## " + re.escape(anchor) + r" ", path.read_text(), re.M), target
        links.append({"label": label, "target": target, "path": path.relative_to(ROOT).as_posix(), "anchor": anchor, "sha256": sha(path)})
    builder = load_module("course_builder_review", ROOT / "scripts/build_course.py")
    builder.build(check=True)
    exporter = load_module("course_export_review", ROOT / "scripts/export_course.py")
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    notebook_rows = []
    for item in index:
        original_path = ROOT / item["notebook"]
        executed_path = EXECUTED / Path(item["notebook"]).relative_to("notebooks")
        original = json.loads(original_path.read_bytes())
        executed = exporter.checked_notebook(original, executed_path)
        site_path = SITE / (item["id"] + ".html")
        html = site_path.read_text()
        visible, code_cells = 0, 0
        for cell in executed["cells"]:
            if cell["cell_type"] != "code" or cell.get("metadata", {}).get("course_setup") or cell.get("metadata", {}).get("course_figure"):
                continue
            code_cells += 1
            output = exporter.output_html(cell.get("outputs", []))
            if output:
                # Markdown may insert inter-tag whitespace inside details.
                assert re.sub(r">\s+<", "><", output) in re.sub(r">\s+<", "><", html), item["id"]
                visible += 1
        assert (SITE / item["notebook"]).read_bytes() == original_path.read_bytes()
        assert item["notebook"] in html
        assert "colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/main/" + item["notebook"] in html
        notebook_rows.append({"id": item["id"], "source_sha256": sha(original_path), "executed_sha256": sha(executed_path), "site_sha256": sha(site_path), "code_cells": code_cells, "visible_output_cells": visible})
    entry_output, ns = run_blocks(section(ROOT / "course/chapters/01.md", "1.1"))
    assert ns["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
    assert ns["restored"] == "貓看狗，狗看貓。"
    assert ns["ids"][0] == ns["ids"][6]
    recipe_output, ns = run_blocks(section(ROOT / "course/first-steps.md", "W.5"), first_only=True)
    assert ns["result"].tolist() == [[40.0, 1.0], [80.0, 6.0]]
    from tiny_perceptron.lesson_products.inference import rule_depth
    policy = json.loads((ROOT / "docs/lesson-products/results/b-auto-policy.json").read_bytes())
    depths = [rule_depth(policy, "boxes", n) for n in range(1, 7)]
    assert depths == [3, 3, 3, 3, 5, 5]
    assert rule_depth(policy, "sum") == 1
    raw_receipts = []
    for stage in ("pretrain", "sft", "vision", "ocr", "audio", "joint", "native", "weighted"):
        path = ROOT / f"docs/selftrained/results/training-raw/moe-{stage}/raw/train-receipt.json"
        receipt = json.loads(path.read_bytes())
        raw_receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), **{k:receipt.get(k) for k in ("stage", "steps", "seed", "origin", "train_records", "validation_records", "test_used_for_selection")}})
    b_c = {}
    for product in ("b", "c"):
        path = ROOT / f"docs/lesson-products/results/initial-{product}-execution.json"
        raw = json.loads(path.read_bytes())
        b_c[product] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "run_id": raw["run_id"], "code_revision": raw["code_revision"], "gpu_name": raw["gpu_name"], "training": {k:raw["training"].get(k) for k in ("random_initialization", "resume_source", "last_step", "selected_step", "seed", "parameters", "initial_weights_sha256", "selected_weights_sha256", "weights_changed", "data_manifest_sha256", "scope")}}
    result = {
        "command": "uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/course/audit-course.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU", "platform": platform.platform()},
        "source_sha256": sha(guide), "freeze_sha256": sha(freeze_path), "guide_snapshot_equal": True,
        "guide_links": links,
        "notebook_audit": {"total": len(notebook_rows), "current_sources_and_executions_equal": True, "rendered_outputs_match_execution": True, "site_downloads_equal_repo_notebooks": True, "rows": notebook_rows},
        "entry_output": entry_output, "recipe_output": recipe_output, "auto_depths_events_1_to_6": depths,
        "moe_training_raw_receipts": raw_receipts, "bc_training_execution": b_c,
        "scope": "No model training, weight/data download, publication or Colab run. Entire notebook set was checked for source/execution/output identity; only guide-relevant 1.1, W.5 first example and rule_depth were independently executed. Historical training records are inspected evidence, not a new training reproduction."
    }
    (HERE / "audit-course.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"guide_links": len(links), "notebooks": len(notebook_rows), "code_cells": sum(r["code_cells"] for r in notebook_rows), "visible_output_cells": sum(r["visible_output_cells"] for r in notebook_rows), "entry_output": entry_output, "recipe_output": recipe_output, "auto_depths": depths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
