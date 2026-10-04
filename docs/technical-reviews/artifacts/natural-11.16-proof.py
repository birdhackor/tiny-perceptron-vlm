"""Bounded, CPU-only independent factual checks for section 11.16.

The short-string oracle explores actual edit strings using breadth-first search;
it does not reproduce the repository's dynamic-programming recurrence.
"""

import ast
from collections import Counter, deque
from datetime import UTC, datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.natural_concepts import edit_distance, text_error_report

torch.set_num_threads(1)
assert torch.version.cuda is None
assert not torch.cuda.is_available()
assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"


def brute_edits(reference, prediction):
    """Shortest actual edit path, from prediction to reference, over {a,b}."""
    alphabet = "ab"
    # A deletion-then-insertion path costs <= this bound. Every string visited
    # on a better path has length <= starting length + this bound.
    upper = len(reference) + len(prediction)
    max_length = len(prediction) + upper
    queue = deque([(prediction, [])])
    seen = {prediction}
    while queue:
        current, path = queue.popleft()
        if current == reference:
            return len(path), path
        if len(path) >= upper:
            continue
        candidates = []
        for i, old in enumerate(current):
            candidates.append((current[:i] + current[i + 1 :], f"delete {old!r} at {i}"))
            for new in alphabet:
                if old != new:
                    candidates.append((current[:i] + new + current[i + 1 :], f"replace {old!r} with {new!r} at {i}"))
        if len(current) < max_length:
            for i in range(len(current) + 1):
                for new in alphabet:
                    candidates.append((current[:i] + new + current[i:], f"insert {new!r} at {i}"))
        for following, operation in candidates:
            if following not in seen:
                seen.add(following)
                queue.append((following, path + [operation]))
    raise AssertionError("Deletion/insertion path should always exist")


strings = ["".join(chars) for size in range(4) for chars in product("ab", repeat=size)]
comparisons = []
for reference in strings:
    for prediction in strings:
        oracle, path = brute_edits(reference, prediction)
        observed = edit_distance(reference, prediction)
        assert observed == oracle, (reference, prediction, oracle, observed, path)
        assert observed == edit_distance(prediction, reference)
        comparisons.append([reference, prediction, oracle])

golden = [
    ("original 台/臺", "今天去臺北", "今天去台北", False, 1, 5, 0.2, "Replace prediction's 台 with 臺: one edit, reference has five code points."),
    ("omitted 北", "今天去臺北", "今天去臺", False, 1, 5, 0.2, "Prediction to reference inserts 北; the reference denominator stays five."),
    ("extra 北", "今天去臺北", "今天去臺北北", False, 1, 5, 0.2, "Prediction to reference deletes one 北; denominator is not six."),
    ("leading omission", "abc", "bc", False, 1, 3, 1 / 3, "Insert a at the start; positional zip would incorrectly report two mismatches."),
    ("internal omission", "ababa", "abba", False, 1, 5, 0.2, "Insert a at index two, rather than count shifted position mismatches."),
    ("empty pair", "", "", True, 0, 0, None, "No edit is needed; empty-reference CER is deliberately undefined (None)."),
    ("empty reference with hallucinated text", "", "你好", False, 2, 0, None, "Delete both predicted characters. This does not turn zero-denominator CER into a number."),
    ("empty prediction", "字", "", False, 1, 1, 1.0, "Insert 字; nonempty-reference denominator is one."),
    ("CER may exceed one", "字", "字字字", False, 2, 1, 2.0, "Delete two extra 字 characters; this ratio is not clipped to one."),
    ("no transposition primitive", "ab", "ba", False, 2, 2, 1.0, "Two replacements, or delete/insert: a swap is not a unit-cost operation."),
    ("non-BMP codepoint", "𠀀", "𠀀", True, 0, 1, 0.0, "U+20000 occupies one Python str position but four UTF-8 bytes."),
    ("no automatic Unicode normalization", "é", "e\u0301", False, 2, 1, 2.0, "Replace e by é, delete U+0301. Python code-point comparison does not silently perform NFC."),
    ("grapheme is not a codepoint", "👩\u200d💻", "👩", False, 2, 3, 2 / 3, "Insert U+200D and U+1F4BB; the reference is three code points."),
]
golden_results = []
for label, reference, prediction, exact, edits, characters, cer, rationale in golden:
    expected = {"exact": exact, "edits": edits, "reference_characters": characters, "cer": cer}
    actual = text_error_report(reference, prediction)
    assert {key: actual[key] for key in expected} == expected, (label, expected, actual)
    golden_results.append({"case": label, "expected": expected, "observed": actual, "hand_reason": rationale,
                           "reference_utf8_bytes": len(reference.encode("utf-8"))})

# Execute the exact official TorchMetrics v1.8.2 helper function, isolated from
# its unrelated dependencies. The saved source is original downloaded bytes.
official_path = Path(__file__).with_name("natural-11.16-torchmetrics-helper.py")
tree = ast.parse(official_path.read_bytes())
node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_edit_distance")
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(official_path), "exec"), namespace)
official = namespace["_edit_distance"]
for item in golden_results:
    actual = item["observed"]
    assert official(list(actual["prediction"]), list(actual["reference"])) == actual["edits"]

reference, prediction = "今天去臺北", "今天去台北"
normalized = [s.replace("台", "臺") for s in (reference, prediction)]
normalization = {"declared_policy": "On both strings, replace every 台 by 臺; no other normalization.",
                 "raw": text_error_report(reference, prediction), "normalized": text_error_report(*normalized)}
assert normalization["normalized"]["exact"] is True
assert normalization["normalized"]["edits"] == 0
assert normalization["normalized"]["cer"] == 0

reference, prediction = "今日休館\n明日開放", "明日開放\n今日休館"
stripped = [s.replace("\n", "") for s in (reference, prediction)]
order = {"raw": text_error_report(reference, prediction), "newlines_removed": text_error_report(*stripped),
         "same_character_counts_before": Counter(reference) == Counter(prediction),
         "same_character_counts_after": Counter(stripped[0]) == Counter(stripped[1]),
         "reference_line_order": reference.splitlines(), "prediction_line_order": prediction.splitlines(),
         "six_replacement_path_zero_based_indices": [i for i, (r, p) in enumerate(zip(*stripped)) if r != p]}
assert order["same_character_counts_before"] and order["same_character_counts_after"]
assert order["newlines_removed"]["reference_characters"] == 8
assert order["newlines_removed"]["exact"] is False
assert order["raw"]["edits"] == order["newlines_removed"]["edits"] == 6
assert order["newlines_removed"]["cer"] == 6 / 8
assert official(list(prediction), list(reference)) == 6
assert official(list(stripped[1]), list(stripped[0])) == 6
assert order["six_replacement_path_zero_based_indices"] == [0, 2, 3, 4, 6, 7]

results = {
    "checked_at": datetime.now(UTC).isoformat(),
    "environment": {"python": platform.python_version(), "python_executable": sys.executable,
                    "torch": torch.__version__, "device": "cpu", "cuda_build": str(torch.version.cuda),
                    "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"], "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"]},
    "repository_code_sha256": hashlib.sha256((ROOT / "tiny_perceptron/natural_concepts.py").read_bytes()).hexdigest(),
    "official_helper_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
    "independent_short_edit_graph": {"alphabet": "ab", "lengths": "0..3", "ordered_pairs": len(comparisons),
                                     "all_equal": True, "all_symmetric": True, "distance_sum": sum(p[2] for p in comparisons),
                                     "oracle": "Breadth-first search through actual insertion/deletion/replacement edits, not DP."},
    "golden_cases": golden_results,
    "normalization": normalization,
    "reading_order": order,
    "result": "All assertions passed.",
    "scope": "Deterministic string metric checks only. No images, pretrained model, natural OCR quality, GPU, or corpus evaluation was tested.",
}
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
