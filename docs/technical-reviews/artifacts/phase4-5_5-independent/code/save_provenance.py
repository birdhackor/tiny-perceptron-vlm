"""Persist byte-level inputs, actual commands and the scope of this independent review."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read_ranges = []
for source, section in [("course/chapters/05.md", "5.3"), ("course/chapters/05.md", "5.4"), ("course/chapters/01.md", "1.13")]:
    raw, whole, start = helper.original_section(ROOT / source, section)
    path = BASE / "original" / ("context-" + section + ".md")
    path.write_bytes(raw)
    read_ranges.append({"source": source + "#" + section, "first_line": start,
                        "snapshot": str(path.relative_to(BASE)), "sha256": hashlib.sha256(raw).hexdigest()})
raw = (ROOT / "course/chapters/05.md").read_bytes().splitlines(keepends=True)
incidental = b"".join(raw[59:149])
(BASE / "original/context-ch05-lines60-149.md").write_bytes(incidental)
read_ranges.append({"source": "course/chapters/05.md", "lines": "60-149", "purpose": "Actual extra read while checking preceding Adam/momentum context; contains latter part of 5.2 as well as 5.3/5.4", "sha256": hashlib.sha256(incidental).hexdigest()})
provenance = {"reviewer_task": "/root/phase4_factual_coordinator/factual_5_5", "context": "fresh",
              "selected_source": "course/chapters/05.md#5.5", "selected_section_raw_sha256": hashlib.sha256((BASE / "original/section.md").read_bytes()).hexdigest(),
              "actual_related_read_ranges": read_ranges,
              "first_section_of_chapter": False, "chapter_introduction_read": False,
              "legacy_technical_or_reader_report_content_read": False,
              "locator_index": "outputs/course-revision-20261005/original-paper-locators.json read for PDF paths/hashes/printed IDs only; no relevant AdamW PDF there, so obtained fresh from arXiv",
              "external_primary_sources": ["results/source-acquisition.json", "results/matching-source-acquisition.json", "results/validation-source-acquisition.json"],
              "inputs_not_used": ["checkpoints", "datasets", "model weights", "legacy review conclusions", "prior training result JSON"],
              "figures": "5.5 has zero SVG references and no embedded figure; no textbook figure to render. Original paper page 3 was actually rendered and viewed to distinguish its colored L2 and decoupled terms.",
              "original_empirical_json": "5.5 references no empirical JSON or measured model metrics; numeric verification concerns the scalar optimizer example only.",
              "environment_skill": "cloud-environment-onboarding:setup read; existing CPU .venv validated; no setup configuration changed"}
(BASE / "results/input-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
commands = '''All shell commands were executed with bash login:false from /workspace/tiny-perceptron-vlm.

Original fence extraction and guarded execution (exit 0, 45-second bound):
.venv/bin/python -I docs/review-tools/section_facts.py course/chapters/05.md#5.5 --output /tmp/phase4-5_5-facts-20261005 --execute --timeout 45
The extracted raw bytes, bootstrap, execution.json, environment.json, stdout.txt and stderr.txt were copied without rewriting into original/ in this permanent artifact directory. No workspace symlinks or weights copied.

Primary HTTPS acquisition (exit 0 each; default verified TLS; per-request 30-second timeout):
.venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/acquire_sources.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/inspect_installed.py
.venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/acquire_matching_pytorch.py
.venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/acquire_validation_contract.py

Paper text and actual image inspection (exit 0 each):
pdftotext -layout docs/technical-reviews/artifacts/phase4-5_5-independent/sources/adamw-1711.05101v3.pdf docs/technical-reviews/artifacts/phase4-5_5-independent/sources/adamw-1711.05101v3.txt
pdftoppm -f 3 -l 3 -r 110 -png -singlefile docs/technical-reviews/artifacts/phase4-5_5-independent/sources/adamw-1711.05101v3.pdf docs/technical-reviews/artifacts/phase4-5_5-independent/sources/adamw-algorithm2-page3
functions.view_image(path='/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-5_5-independent/sources/adamw-algorithm2-page3.png')

Bounded exact fence and independent checks (exit 0):
timeout 45s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/bounded_checks.py > docs/technical-reviews/artifacts/phase4-5_5-independent/results/bounded-stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_5-independent/results/bounded-stderr.txt

Provenance and report serialization:
First save_provenance invocation exited 1 because its own repo-root parent index was 4 instead of 5; corrected this reviewer-script path and reran. No textbook execution failed.
.venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/save_provenance.py
.venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/build_report.py

Final checker (performed after report creation):
.venv/bin/python -I scripts/check_technical_reviews.py --lesson 5.5
'''
(BASE / "results/commands.txt").write_text(commands)
print(json.dumps({"saved": "results/input-provenance.json", "related_snapshots": len(read_ranges)}, indent=2))
