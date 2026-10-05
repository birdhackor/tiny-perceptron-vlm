"""Bounded CPU checks of the actual text_error_report API; no model evaluation."""
from collections import Counter
from pathlib import Path
import inspect
import json
import platform
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.natural_concepts import edit_distance, text_error_report

torch.set_num_threads(1)
assert not torch.cuda.is_available()
cases = [
    ("original", "今天去臺北", "今天去台北", False, 1, 5, 0.2),
    ("identical", "今天去臺北", "今天去臺北", True, 0, 5, 0.0),
    ("missing_final_only", "今天去臺北", "今天去臺", False, 1, 5, 0.2),
    ("variant_and_missing_final", "今天去臺北", "今天去台", False, 2, 5, 0.4),
    ("extra_final_only", "今天去臺北", "今天去臺北北", False, 1, 5, 0.2),
    ("role_swap_unequal_lengths", "今天去臺", "今天去臺北", False, 1, 4, 0.25),
    ("all_missing", "今天去臺北", "", False, 5, 5, 1.0),
    ("empty_both", "", "", True, 0, 0, None),
    ("blank_reference_with_text", "", "今日休館", False, 4, 0, None),
    ("codepoints_not_graphemes", "é", "e\u0301", False, 2, 1, 2.0),
]
results = []
for name, reference, prediction, exact, edits, n, cer in cases:
    report = text_error_report(reference, prediction)
    assert (report["exact"], report["edits"], report["reference_characters"], report["cer"]) == (exact, edits, n, cer)
    assert report["reference"] == reference and report["prediction"] == prediction
    assert edit_distance(prediction, reference) == edits
    results.append({"case": name, "report": report, "reference_utf8_bytes": len(reference.encode("utf-8")), "prediction_codepoints": len(prediction)})

raw_reference, raw_prediction = cases[0][1:3]
mapping = str.maketrans({"台": "臺"})
normalized = text_error_report(raw_reference.translate(mapping), raw_prediction.translate(mapping))
assert normalized["exact"] and normalized["edits"] == 0
assert text_error_report(raw_reference, raw_prediction)["edits"] == 1
assert not text_error_report(unicodedata.normalize("NFC", raw_reference), unicodedata.normalize("NFC", raw_prediction))["exact"]
results.append({"case": "explicit_variant_mapping", "raw": text_error_report(raw_reference, raw_prediction), "normalized": normalized, "NFC_alone_keeps_variant_difference": True})

reference = "今日休館\n明日開放"
prediction = "明日開放\n今日休館"
for name, r, p in [("line_order_with_newline", reference, prediction), ("line_order_ignore_newline", reference.replace("\n", ""), prediction.replace("\n", ""))]:
    report = text_error_report(r, p)
    assert Counter(r) == Counter(p)
    assert not report["exact"] and report["edits"] > 0
    results.append({"case": name, "same_character_multiset": True, "report": report})

environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "device": "cpu", "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(), "unicode_database": unicodedata.unidata_version}
print(json.dumps({"environment": environment, "signature": str(inspect.signature(text_error_report)), "case_count": len(results), "results": results}, ensure_ascii=False, indent=2, allow_nan=False))
