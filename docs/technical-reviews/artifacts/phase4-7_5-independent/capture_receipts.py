"""Preserve commands, observed process completion, runtime and render provenance."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
run = json.loads((HERE / "bounded-controls.stdout.json").read_text())
original = json.loads((HERE / "original/execution.json").read_text())
command = ["timeout", "25", "inkscape", str(HERE / "context-figure/alignment.svg"), "--export-type=png", "--export-width=1600", f"--export-filename={HERE / 'context-figure/alignment.png'}"]
render = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
(HERE / "context-figure/render.stdout.txt").write_bytes(render.stdout)
(HERE / "context-figure/render.stderr.txt").write_bytes(render.stderr)
renderer_version = subprocess.run(["inkscape", "--version"], capture_output=True, text=True, check=True).stdout.strip()
assert render.returncode == 0 and (HERE / "context-figure/alignment.png").is_file()
receipts = {
    "original_extraction_and_execution": {
        "command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.5 --execute --output /tmp/phase4-7_5-original --timeout 45",
        "cwd": str(ROOT), "observed_outer_exit_code": 0, "worker_execution_receipt": original,
        "permanent_copy": "original/", "temporary_workspace_required": False,
    },
    "controls": {
        "command": "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_5-independent/bounded_controls.py > docs/technical-reviews/artifacts/phase4-7_5-independent/bounded-controls.stdout.json 2> docs/technical-reviews/artifacts/phase4-7_5-independent/bounded-controls.stderr.txt",
        "cwd": str(ROOT), "observed_exit_code": 0, "environment": run["environment"],
        "code_sha256": hashlib.sha256((HERE / "bounded_controls.py").read_bytes()).hexdigest(),
        "stdout_sha256": hashlib.sha256((HERE / "bounded-controls.stdout.json").read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256((HERE / "bounded-controls.stderr.txt").read_bytes()).hexdigest(),
    },
    "sources": {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-7_5-independent/acquire_sources.py > docs/technical-reviews/artifacts/phase4-7_5-independent/source-acquisition.stdout.json",
        "observed_exit_code": 0, "authority_provenance": "sources/download-provenance.json",
        "note": "An initial urllib request without User-Agent returned HTTP403; the saved acquisition script uses an explicit User-Agent and all requested authority sources were successfully obtained. No TLS bypass.",
    },
    "paper_text": {
        "command": "pdftotext -layout docs/technical-reviews/artifacts/phase4-7_5-independent/sources/attention-paper-v7.pdf docs/technical-reviews/artifacts/phase4-7_5-independent/sources/attention-paper-v7.txt",
        "observed_exit_code": 0,
    },
    "contextual_figure": {
        "argv": command, "exit_code": render.returncode, "renderer_version": renderer_version,
        "viewed_by_reviewer": True, "view_method": "functions.exec tools.view_image",
        "original_source": "course/figures/rewrite-07-04-answer-alignment.svg",
        "svg_sha256": hashlib.sha256((HERE / "context-figure/alignment.svg").read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256((HERE / "context-figure/alignment.png").read_bytes()).hexdigest(),
        "scope": "7.4 contextual alignment figure only; 7.5 has no figure reference",
    },
}
(HERE / "command-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"render_exit_code": render.returncode, "original_exit_code": original["exit_code"], "controls_observed_exit_code": 0, "runtime": run["environment"]}, ensure_ascii=False, indent=2))
