"""Inspect method origin for administrative STOP classification, never score models."""
import ast
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REVISION = "1e71b8abffb34eccd297747d39827135a627107a"


def digest(data):
    return hashlib.sha256(data).hexdigest()


methods = []
for relative, function in (
    ("scripts/score_natural_v4_validation.py", "score_blind"),
    ("scripts/modal_natural.py", "selection_gate"),
    ("scripts/modal_natural.py", "execute_stage"),
):
    command = ["git", "show", REVISION + ":" + relative]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    pinned = result.stdout.decode("utf-8")
    current = (ROOT / relative).read_text()
    node = next(n for n in ast.walk(ast.parse(pinned))
                if isinstance(n, ast.FunctionDef) and n.name == function)
    current_node = next(n for n in ast.walk(ast.parse(current))
                        if isinstance(n, ast.FunctionDef) and n.name == function)
    segment = ast.get_source_segment(pinned, node)
    same = segment == ast.get_source_segment(current, current_node)
    assert same
    methods.append({"path": relative, "function": function,
                    "pinned_revision": REVISION, "command": command,
                    "pinned_source_sha256": digest(result.stdout),
                    "pinned_function_sha256": digest(segment.encode()),
                    "pinned_lines": [node.lineno, node.end_lineno],
                    "same_function_bytes_current": same})

manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
row_id = "vision-v4:docci/test_01972/scene"
row = next(r for r in manifest["rows"] if r["id"] == row_id)
labels_path = ROOT / "docs/natural-assistant/v4/data/vision-labels/validation-01.jsonl"
labels_bytes = labels_path.read_bytes()
raw_line, original = next((n, json.loads(line))
                          for n, line in enumerate(labels_bytes.splitlines(), 1)
                          if json.loads(line)["id"] == row_id)
assert original["source"] == row["source"]
source = original["source"]
proof = {
    "kind": "administrative_primary_method_and_annotation_origin_inspection",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(), "device": "metadata only, no model or tensor execution",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-20_8-coordinator-scope-classification/inspect_primary_generation.py",
    "scientific_pass_evidence": False,
    "pinned_methods": methods,
    "selection_method_locator": "score_blind:425-440 derives chosen from original grades/gates;454-463 emits conditional selection decision. selection_gate:376-384 relays original selection;execute_stage:806-817 records pretest_selection.",
    "annotation_origin": {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(manifest_bytes), "row_id": row_id,
        "raw_label_path": labels_path.relative_to(ROOT).as_posix(),
        "raw_label_sha256": digest(labels_bytes), "raw_line": raw_line,
        "annotation_source_identical_manifest": True,
        "generation_method": "scripts/build_natural_v4_assets.py::assemble:195-199 loads original vision-label rows;build:438-458 serializes them into manifest.",
        "evaluation_subject": "original photograph gold QA targets; not model outputs or textbook review",
        "declared_authored_before_model_inference": source["authored_before_v4_model_inference"],
        "declared_peer_reviewed_before_model_inference": source["peer_review"]["reviewed_before_v4_model_inference"],
        "declared_model_predictions_consulted": source["peer_review"]["model_predictions_consulted"],
        "scope_limit": "This origin classification does not independently certify quality or chronology; the original factual owner must verify targets, dates, criteria and counts itself.",
    },
}
print(json.dumps(proof, ensure_ascii=False, indent=2))
