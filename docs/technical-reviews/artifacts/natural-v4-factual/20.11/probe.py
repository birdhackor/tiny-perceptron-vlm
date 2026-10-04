"""Bounded, independent lesson and selected OCR-record checks; no inference."""
from pathlib import Path
from collections import Counter
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import time
import unicodedata

import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
start = time.perf_counter()
environment = {"python": platform.python_version(), "python_executable": sys.executable,
               "torch": torch.__version__, "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
               "unicode_database": unicodedata.unidata_version}
print("environment", json.dumps(environment))
section = (OUT / "20.11.source.md").read_text()
code = re.search(r"```python\n(.*?)```", section, re.S)[1]
print("exact lesson example")
exec(compile(code, "20.11-raw-snippet", "exec"), {})
reference, swapped, exercise = "牛奶\n麵包", "麵包\n牛奶", "牛奶麵包"
print("reference character counts", repr(dict(Counter(reference))))
print("swapped character counts", repr(dict(Counter(swapped))))
print("exercise Counter equality", Counter(reference) == Counter(exercise))
print("exercise exact equality", reference == exercise)
print("exercise lines", exercise.splitlines())
print("exercise nonwhitespace counts", Counter("".join(reference.split())) == Counter(exercise))
print("exercise whitespace-stripped equality", "".join(reference.split()) == "".join(exercise.split()))
print("swapped whitespace-stripped equality", "".join(reference.split()) == "".join(swapped.split()))
print("Counter first-key order retained but not original sequence", list(Counter("牛奶牛")) == list(Counter("牛牛奶")), Counter("牛奶牛") == Counter("牛牛奶"))

from tiny_perceptron.natural_concepts import text_error_report
print("11.16 necessary numeric prerequisite", text_error_report("今天去臺北", "今天去台北"))

# Two pages retain identical region pixels but different absolute page coordinates.
patch = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.uint8)
page_a, page_b = torch.zeros(64, 32, dtype=torch.uint8), torch.zeros(64, 32, dtype=torch.uint8)
page_a[8:11, 6:9], page_b[48:51, 6:9] = patch, patch
print("crop pixel equality / page equality", torch.equal(page_a[8:11, 6:9], page_b[48:51, 6:9]), torch.equal(page_a, page_b))
print("crop parent xyxy", [6, 8, 9, 11], [6, 48, 9, 51])

# Compile only the actual pure normalization function; no CLI/Git/inference entrypoint.
scorer_path = ROOT / "scripts/score_natural_v4_test.py"
tree = ast.parse(scorer_path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalized")
namespace = {"unicodedata": unicodedata}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(scorer_path), "exec"), namespace)
actual_normalized = namespace["normalized"]
def independent_normalized(text, strip_whitespace):
    text = unicodedata.normalize("NFKC", text).strip()
    return "".join(c for c in text if not c.isspace()) if strip_whitespace else text

manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
generations_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/generations-base.json"
scores_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json"
manifest = json.loads(manifest_path.read_text())
records = {(r["task"], r["id"]): r for r in json.loads(generations_path.read_text())}
scores = json.loads(scores_path.read_text())
audit = {"environment": environment, "inputs": {}, "test_groups": {}, "cases": [], "scope": "Direct fixed-record recomputation conditional on committed annotations; not model inference, training, independent human gold validation, or arbitrary page-layout evaluation."}
for p in [scorer_path, ROOT / "tiny_perceptron/natural_concepts.py", manifest_path, generations_path, scores_path]:
    audit["inputs"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
for task, group, strip in [("ocr", "single_ocr", True), ("ocr_order", "ordered_ocr", False)]:
    rows = [r for r in manifest["rows"] if r["split"] == "test" and r["task"] == task]
    count = 0
    for row in rows:
        record = records[task, row["id"]]
        left, right = independent_normalized(row["answer"], strip), independent_normalized(record["prediction"], strip)
        assert left == actual_normalized(row["answer"], strip) and right == actual_normalized(record["prediction"], strip)
        token_ids, eos_ids = record["generated_token_ids"], record["eos_token_ids"]
        complete = (record["ended_with_eos"] is True and record["stop_reason"] == "eos" and record["truncated"] is False
                    and record["completion_unknown"] is False and record["reached_max_new_tokens"] is False
                    and len(token_ids) == record["generated_tokens"] and token_ids[-1] in eos_ids
                    and 0 < len(token_ids) <= 384 and all(type(t) is int and t >= 0 for t in token_ids + eos_ids))
        passed = complete and left == right
        count += int(passed)
        audit["cases"].append({"id": row["id"], "group": group, "split": row["split"], "view": row.get("view"),
                              "image": row["image"], "question": row["question"], "reading_order": row.get("reading_order"),
                              "references": row["references"], "reference": row["answer"], "prediction": record["prediction"],
                              "normalized_reference": left, "normalized_prediction": right, "decoder_complete": complete, "passed": passed})
    audit["test_groups"][group] = {"passed": count, "denominator": len(rows), "strip_whitespace": strip,
                                  "saved_passed": scores["correct_counts"][group], "saved_denominator": scores["denominators"][group]}
    assert count == scores["correct_counts"][group] and len(rows) == scores["denominators"][group]
audit["bindings_match"] = {"manifest": audit["inputs"][str(manifest_path.relative_to(ROOT))] == scores["artifact_binding"]["manifest_sha256"],
                           "generations": audit["inputs"][str(generations_path.relative_to(ROOT))] == scores["artifact_binding"]["generation_file"]["sha256"],
                           "scorer": audit["inputs"][str(scorer_path.relative_to(ROOT))] == scores["artifact_binding"]["scoring_sources"]["scripts/score_natural_v4_test.py"]["sha256"]}
print("OCR record groups", json.dumps(audit["test_groups"], ensure_ascii=False))
print("record bindings", json.dumps(audit["bindings_match"]))
print("newline / whitespace scoring", actual_normalized(reference, False) == actual_normalized(exercise, False), actual_normalized(reference, True) == actual_normalized(exercise, True))
print("normalization form / script / punctuation", repr(actual_normalized("Ａ\u3000臺！\n", True)), actual_normalized("臺", True) == actual_normalized("台", True))
audit["elapsed_seconds"] = time.perf_counter() - start
(OUT / "record-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print("elapsed_seconds", audit["elapsed_seconds"])
