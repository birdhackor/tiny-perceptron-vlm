"""Verify a specific official-train caption; no images, test rows or inference."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
manifest_path = ROOT / "docs/natural-assistant/manifest.json"
source_path = ROOT / "outputs/natural-extension/data/vision/sources/docci-descriptions.jsonl"
helper_path = ROOT / "scripts/prepare_natural_vision.py"
digest = lambda data: hashlib.sha256(data).hexdigest()
manifest_bytes = manifest_path.read_bytes()
assert digest(manifest_bytes) == "7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91"
rows = [r for r in json.loads(manifest_bytes)["rows"] if r["split"] == "train" and r["task"] == "scene"]
selected = [r for r in rows if r["source"]["original_id"] == "train_03125"]
assert len(selected) == 1
row = selected[0]
# Hash the official whole source as bytes; JSON parsing selects only the requested train line.
source_sha = digest(source_path.read_bytes())
assert source_sha == row["source"]["description_source_sha256"]
matches = []
with source_path.open("rb") as handle:
    for line_number, raw in enumerate(handle, 1):
        if b'"train_03125"' in raw:
            matches.append((line_number, raw))
assert len(matches) == 1
line_number, raw_line = matches[0]
original = json.loads(raw_line)
assert original["example_id"] == "train_03125" and original["split"] == "train"
helper_code = helper_path.read_text()
helper_node = next(n for n in ast.parse(helper_code).body if isinstance(n, ast.FunctionDef) and n.name == "first_caption_sentence")
helper_source = ast.get_source_segment(helper_code, helper_node)
computed_target = re.split(r"(?<=[.!?])\s+", original["description"].strip(), maxsplit=1)[0]
assert computed_target == row["answer"]
assert '"STOP." On the bottom of the white boarder is a rust color.' in row["answer"]
fragment_path = OUT / "train_03125-original-line.jsonl"
fragment_path.write_bytes(raw_line)
quote_boundary_rows = [r["id"] for r in rows if re.search(r'[.!?]["\u201d\u2019\x27]\s+', r["answer"])]
receipt = {
    "scope": "Specific official-train caption boundary verification; byte hashing/scanning only for other source lines; no test JSON row parsing, test labels/images/outputs, GPU or inference",
    "manifest_sha256": digest(manifest_bytes),
    "manifest_unchanged": digest(manifest_path.read_bytes()) == digest(manifest_bytes),
    "row_id": row["id"], "source_official_split": original["split"],
    "frozen_manifest_row": row,
    "original_train_description": original,
    "source": {"path": str(source_path.relative_to(ROOT)), "sha256": source_sha,
        "line_number": line_number, "line_bytes": len(raw_line), "line_sha256": digest(raw_line),
        "official_url": "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines?generation=1714384012999810",
        "license": "CC BY 4.0"},
    "exact_byte_line_excerpt": {"path": str(fragment_path.relative_to(ROOT)), "sha256": digest(fragment_path.read_bytes()),
        "byte_identical_to_source_line": fragment_path.read_bytes() == raw_line},
    "helper": {"path": str(helper_path.relative_to(ROOT)), "sha256": digest(helper_path.read_bytes()),
        "source_excerpt": helper_source, "line_number": helper_node.lineno},
    "computed_target_equals_frozen_target": computed_target == row["answer"],
    "explanation": "The period in STOP. is followed by a closing double quote, not whitespace, so the existing regex skips that sentence boundary and cuts after the following rust-color sentence.",
    "corrected_wording": "開頭片段，多數一句；句尾引號可能帶下一句",
    "train_caption_rows": len(rows),
    "targets_with_sentence_punctuation_closing_quote_whitespace": quote_boundary_rows,
    "quote_screen_limit": "This is a textual closing-quote boundary screen, not a general linguistic sentence-count algorithm.",
}
(OUT / "caption-boundary-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ["row_id", "manifest_sha256", "manifest_unchanged",
    "computed_target_equals_frozen_target", "corrected_wording", "targets_with_sentence_punctuation_closing_quote_whitespace"]}, ensure_ascii=False, indent=2))
