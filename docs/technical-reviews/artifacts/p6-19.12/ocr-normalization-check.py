"""Fixed-string check of the original OCR scorer, with static target inspection."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tarfile
import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("original_ocr_scorer", ROOT / "scripts/selftrained/evaluate.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
chapter = (ROOT / "course/chapters/19.md").read_bytes()
heading = re.search(rb"^## 19\.12 .+$", chapter, re.M)
next_heading = re.search(rb"^## ", chapter[heading.end():], re.M)
end = heading.end() + next_heading.start() if next_heading else len(chapter)
section = chapter[heading.start():end]
assert sha(section) == "94c49ae8bfc6b0d4d4ba33431208cff5f3be7ec9f14e2d8bd6f80116b0652883"

record = {"task": "ocr", "messages": [{"role": "user", "content": "讀出指定ROI。"}, {"role": "assistant", "content": "大小"}], "supervision": {"ocr_text": "大小"}}
trace = {"tool_call": None, "status": "no_tool"}
cases = [
    ("exact glyph sequence", "大小", "大小", True, 0),
    ("internal ASCII space", "大 小", "大小", True, 0),
    ("internal newline and tab", "大\n\t小", "大小", True, 0),
    ("Unicode ideographic space", "大\u3000小", "大小", True, 0),
    ("edge Chinese/ASCII punctuation and quotes", "！?「'\"大小\"'」?！", "大小", True, 0),
    ("unpaired edge quote and question mark", "「大小?", "大小", True, 0),
    ("internal exclamation mark", "大！小", "大！小", False, 1),
    ("ASCII full stop is retained", "大小.", "大小.", False, 1),
    ("Chinese comma is retained", "大小，", "大小，", False, 1),
    ("curly quotation marks are retained", "“大小”", "“大小”", False, 2),
    ("different corner quotes are retained", "『大小』", "『大小』", False, 2),
    ("target glyph order changed", "小大", "小大", False, 2),
    ("target glyph missing", "大", "大", False, 1),
]
observed = []
for label, text, expected_normalized, expected_success, expected_errors in cases:
    score = scorer.score_reply(record, text, trace)
    normalized = scorer.normalize(text)
    assert normalized == expected_normalized
    assert score["semantic"] == score["exact"] == expected_success
    assert score["ocr_errors"] == expected_errors and score["ocr_characters"] == 2
    observed.append({"label": label, "output": text, "normalized_output": normalized, "exact": score["exact"], "semantic": score["semantic"], "edit_errors": score["ocr_errors"], "gold_characters": score["ocr_characters"], "CER": score["ocr_errors"] / score["ocr_characters"]})

manifest_path = ROOT / "docs/selftrained/v2-manifest.json"
manifest = json.loads(manifest_path.read_bytes())
archive_path = ROOT / manifest["package"]["path"]
archive_sha = sha(archive_path.read_bytes())
assert archive_sha == manifest["package"]["sha256"]
targets = {}
with tarfile.open(archive_path) as tar:
    for split in ["train", "validation", "test"]:
        filename = f"ocr-{split}.jsonl"
        raw = tar.extractfile(filename).read()
        binding = next(item for item in manifest["records"] if item["path"] == filename)
        assert sha(raw) == binding["sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        glyphs = {character for row in rows for character in row["supervision"]["ocr_text"]}
        equal = all(row["messages"][-1]["content"] == row["supervision"]["ocr_text"] for row in rows)
        invariant = all(scorer.normalize(row["supervision"]["ocr_text"]) == row["supervision"]["ocr_text"] for row in rows)
        assert equal and invariant and len(glyphs) == 12
        targets[split] = {"file": filename, "raw_bytes_sha256": sha(raw), "question_rows": len(rows), "characters": "".join(sorted(glyphs)), "all_assistant_targets_equal_ocr_gold": equal, "all_gold_unchanged_by_normalize": invariant, "gold_lengths": sorted({len(row["supervision"]["ocr_text"]) for row in rows}), "gold_character_denominator": sum(len(row["supervision"]["ocr_text"]) for row in rows)}
out = {
    "reviewer_task": "/root/p6_fact_19_12",
    "source_sha256": sha(section),
    "scorer_path": "scripts/selftrained/evaluate.py",
    "scorer_sha256": sha((ROOT / "scripts/selftrained/evaluate.py").read_bytes()),
    "method_locators": ["normalize L197–198", "score_reply OCR L274–278", "summarize final_cer L529–531"],
    "normalization_rule": "Remove all Python Unicode-regex whitespace anywhere; then strip the individual characters 。！!？?\"'「」 repeatedly from both edges. Internal punctuation is retained; other punctuation/quote characters are retained.",
    "scoring_rule": "OCR semantic success is normalized output == unnormalized ocr_text gold. Initial exact is normalize(output)==normalize(final assistant target). Because every actual OCR target equals its normalization and ocr_text gold, these two success checks coincide on this fixed data. Final CER is sum(edit_distance(normalized output,gold))/sum(len(gold)), not raw-output-byte or raw-punctuation fidelity.",
    "cases": observed,
    "target_scope": {"manifest_sha256": sha(manifest_path.read_bytes()), "archive_sha256": archive_sha, "splits": targets},
    "section_contains_CER_claim": "CER" in section.decode(),
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "neural_generation": False,
    "heldout_model_evaluation": False,
    "training": False,
    "paid_calls": False,
}
(ART / "ocr-normalization-check-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"scorer_sha256": out["scorer_sha256"], "source_sha256": out["source_sha256"], "fixed_cases_passed": len(observed), "normalization_rule": out["normalization_rule"], "targets": targets, "section_contains_CER_claim": out["section_contains_CER_claim"]}, ensure_ascii=False, indent=2))
