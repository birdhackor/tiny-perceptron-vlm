"""Bounded independent checks of lesson 11.16; no inference or training."""
from pathlib import Path
from collections import Counter, deque
from contextlib import redirect_stdout
from itertools import product
import hashlib
import io
import json
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.natural_concepts import edit_distance, text_error_report


def full_matrix(reference, prediction):
    table = [[0] * (len(reference) + 1) for _ in range(len(prediction) + 1)]
    table[0] = list(range(len(reference) + 1))
    for j in range(1, len(prediction) + 1):
        table[j][0] = j
        for i in range(1, len(reference) + 1):
            table[j][i] = min(table[j - 1][i] + 1, table[j][i - 1] + 1,
                              table[j - 1][i - 1] + int(prediction[j - 1] != reference[i - 1]))
    return table[-1][-1]


def bfs_distance(reference, prediction):
    # Independent shortest-path search over actual edit operations, not DP.
    alphabet = set(reference + prediction)
    queue, seen = deque([(prediction, 0)]), {prediction}
    while queue:
        text, distance = queue.popleft()
        if text == reference:
            return distance
        successors = [text[:i] + text[i + 1:] for i in range(len(text))]
        successors += [text[:i] + c + text[i + 1:] for i in range(len(text)) for c in alphabet]
        if len(text) < max(len(reference), len(prediction)):
            successors += [text[:i] + c + text[i:] for i in range(len(text) + 1) for c in alphabet]
        for candidate in successors:
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, distance + 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text, strip_all=False):
    result = unicodedata.normalize("NFKC", text).strip()
    return "".join(result.split()) if strip_all else result


body = (ROOT / "course/chapters/11.md").read_bytes().decode()
section = body[re.search(r"^## 11\.16\b", body, re.M).start():]
code = re.search(r"```python\n(.*?)```", section, re.S)[1]
stream = io.StringIO()
with redirect_stdout(stream):
    exec(compile(code, "course/chapters/11.md#11.16", "exec"), {})
assert stream.getvalue() == "完全相同 False\n最少編輯次數 1\n參考字數與CER 5 0.2\n"

pairs = [("今天去臺北", "今天去台北"), ("今天去臺北", "今天去臺"),
         ("今天去臺北", "今天去台"), ("今天去臺北", "今天去臺北北"),
         ("今天去臺北", "今天去臺北"), ("", ""), ("", "臺北"), ("臺北", ""),
         ("1010", "010"), ("1010", "10100")]
reports = []
for reference, prediction in pairs:
    report = text_error_report(reference, prediction)
    assert report["edits"] == full_matrix(reference, prediction)
    assert report["reference_characters"] == len(reference)
    reports.append(report)
assert reports[1]["edits"] == 1 and reports[1]["cer"] == 0.2
assert reports[2]["edits"] == 2 and reports[2]["cer"] == 0.4
strings = ["".join(chars) for size in range(4) for chars in product("台臺", repeat=size)]
for reference, prediction in product(strings, repeat=2):
    assert edit_distance(reference, prediction) == bfs_distance(reference, prediction)

reference, prediction = "今日休館明日開放", "明日開放今日休館"
ordering = {"reference": reference, "prediction": prediction,
            "same_multiset": Counter(reference) == Counter(prediction),
            "raw": text_error_report(reference, prediction),
            "ignore_linebreaks": normalize(reference, True) == normalize(prediction, True)}
assert ordering["same_multiset"] and not ordering["raw"]["exact"]
assert ordering["raw"]["edits"] == full_matrix(reference, prediction) > 0
assert not ordering["ignore_linebreaks"]
custom = text_error_report("今天去臺北".replace("台", "臺"), "今天去台北".replace("台", "臺"))
assert custom["exact"] and custom["edits"] == 0
assert unicodedata.normalize("NFKC", "台臺Ａ") == "台臺A"

manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
generation_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/generations-base.json"
scores_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json"
protocol_path = ROOT / "docs/natural-assistant/v4/validation-protocol-lower-lr.json"
manifest = json.loads(manifest_path.read_text())
generations = json.loads(generation_path.read_text())
scores = json.loads(scores_path.read_text())
protocol = json.loads(protocol_path.read_text())
rows = [r for r in manifest["rows"] if r["split"] == "test" and r["task"] in {"text_presence", "ocr", "ocr_order"}]
counts, correct, cases, photos = Counter(), Counter(), [], {}
groups = {"text_presence": "text_presence", "ocr": "single_ocr", "ocr_order": "ordered_ocr"}
for row in rows:
    group = groups[row["task"]]
    counts[group] += 1
    record, = [r for r in generations if r["id"] == row["id"] and r["task"] == row["task"]]
    assert record["reference_answer"] == row["answer"] and record["image"] == row["image"]
    assert row["synthetic"] is False and row["view"] == "whole"
    image_path = ROOT / "outputs/natural-v4/data" / row["image"]
    image_sha = digest(image_path)
    assert image_sha == row["image_sha256"]
    photos[row["image"]] = image_sha
    strip_all = group == "single_ocr"
    target, output = normalize(row["answer"], strip_all), normalize(record["prediction"], strip_all)
    tokens = record["generated_token_ids"]
    complete = (record["ended_with_eos"] is True and record["stop_reason"] == "eos"
                and not record["truncated"] and not record["completion_unknown"]
                and bool(tokens) and tokens[-1] in record["eos_token_ids"]
                and len(tokens) <= protocol["generation"]["max_new_tokens"])
    passed = complete and target == output
    correct[group] += int(passed)
    cases.append({"id": row["id"], "group": group, "reference": row["answer"],
                  "prediction": record["prediction"], "reference_characters": len(target),
                  "edits": full_matrix(target, output), "passed": passed,
                  "decoder_complete": complete, "reading_order": row["reading_order"]})
    if group != "text_presence":
        assert full_matrix(target, output) == record["score"]["errors"]
for group in counts:
    assert counts[group] == scores["denominators"][group]
    assert correct[group] == scores["correct_counts"][group]
assert counts == {"text_presence": 18, "single_ocr": 10, "ordered_ocr": 3}
assert correct == {"text_presence": 18, "single_ocr": 8, "ordered_ocr": 1}
assert Counter(r["answer"] for r in rows if r["task"] == "text_presence") == {"有": 10, "沒有": 8}
assert digest(manifest_path) == scores["artifact_binding"]["manifest_sha256"]
assert digest(protocol_path) == scores["artifact_binding"]["protocol_sha256"]
assert digest(generation_path) == scores["artifact_binding"]["generation_file"]["sha256"]

print(json.dumps({"environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu",
                                  "cuda_available": torch.cuda.is_available(), "unicode": unicodedata.unidata_version},
                  "source_sha256": hashlib.sha256(section.encode()).hexdigest(),
                  "section_stdout": stream.getvalue(), "examples": reports,
                  "shortest_path_pairs_checked": len(strings) ** 2, "ordering_exercise": ordering,
                  "custom_variant_normalized": custom,
                  "unicode_units": {"Chinese_codepoints": len("今天去臺北"),
                                    "Chinese_utf8_bytes": len("今天去臺北".encode()),
                                    "combining_sequence_codepoints": len("e\u0301"),
                                    "composed_codepoints": len("é"), "nfkc_probe": normalize("台臺Ａ")},
                  "final_record_scope": {"counts": counts, "correct": correct, "distinct_photos": len(photos),
                                         "checked_photo_hashes": photos, "cases": cases,
                                         "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                                                           [manifest_path, protocol_path, generation_path, scores_path]},
                                         "limit": "Recomputed fixed existing predictions only; no GPU/model inference replication; no independent full-dataset gold-label audit."}},
                 ensure_ascii=False, indent=2))
