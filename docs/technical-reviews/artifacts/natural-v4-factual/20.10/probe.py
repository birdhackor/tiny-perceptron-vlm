"""Bounded independent verification for section 20.10; no models or training."""
from collections import Counter, deque
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from itertools import product
from pathlib import Path
import json
import platform
import re
import unicodedata
import torch

from tiny_perceptron.natural_concepts import edit_distance, text_error_report
from tiny_perceptron.natural_assistant import normalized, score_output

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent


def shortest_edits(start, end):
    """Breadth-first edit graph, independently of the production DP recurrence."""
    alphabet = set(start + end)
    limit = max(len(start), len(end))
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        if current == end:
            return distance
        neighbours = {current[:i] + current[i + 1:] for i in range(len(current))}
        neighbours.update(current[:i] + letter + current[i + 1:]
                          for i in range(len(current)) for letter in alphabet)
        if len(current) < limit:
            neighbours.update(current[:i] + letter + current[i:]
                              for i in range(len(current) + 1) for letter in alphabet)
        for value in neighbours - seen:
            seen.add(value)
            queue.append((value, distance + 1))
    raise AssertionError((start, end))


section = (OUT / "20.10.raw.md").read_text()
code = re.search(r"```python\n(.*?)```", section, re.S)[1]
capture = StringIO()
with redirect_stdout(capture):
    exec(compile(code, "course/chapters/20.md#20.10", "exec"), {})
assert capture.getvalue() == "完整相同 False\n最少編輯次數 1\n參考字數 5\nCER 0.2\n"

examples = []
for ref, pred, edits, cer in [
    ("今天去臺北", "今天去台北", 1, 0.2),
    ("今天去臺北", "今天去臺", 1, 0.2),
    ("今天去臺北", "今天去台", 2, 0.4),
    ("今天去臺北", "今天去臺北北", 1, 0.2),
    ("", "", 0, None), ("", "幻覺", 2, None),
    ("甲", "甲乙丙丁", 3, 3.0), ("é", "e\u0301", 2, 2.0),
]:
    report = text_error_report(ref, pred)
    assert (report["edits"], report["cer"]) == (edits, cer)
    assert report["reference_characters"] == len(ref)
    examples.append(report)
assert len("今天去臺北") == 5 and len("今天去臺北".encode("utf-8")) == 15

strings = ["".join(chars) for length in range(5) for chars in product("甲乙", repeat=length)]
for ref, pred in product(strings, repeat=2):
    assert edit_distance(ref, pred) == shortest_edits(ref, pred)

nfkc = {text: unicodedata.normalize("NFKC", text)
        for text in ["Ａ", "臺", "台", "絕味鴨脖", "绝味鸭脖", "Ａ，Ｂ", " A\u3000B\n"]}
assert nfkc["Ａ"] == "A" and nfkc["臺"] == "臺" and nfkc["台"] == "台"
assert nfkc["絕味鴨脖"] != nfkc["绝味鸭脖"]
policy = []
for ref, pred, kind, strip, passed in [
    ("Ａ B", "AB", "ocr", True, True),
    ("臺北", "台北", "ocr", True, False),
    ("绝味鸭脖", "絕味鴨脖", "ocr", True, False),
    ("AB，", "AB,", "ocr", True, True),
    ("AB,", "AB", "ocr", True, False),
    ("AB", "ab", "ocr", True, False),
    ("甲乙\n丙丁", "甲乙丙丁", "ocr_order", False, False),
    ("今日休館明日開放", "明日開放今日休館", "ocr_order", False, False),
    ("", "幻覺", "ocr", True, False),
]:
    row = {"answer": ref, "references": {"kind": kind, "text": ref, "strip_whitespace": strip}}
    result = score_output(row, pred)
    assert result["passed"] is passed
    if not ref:
        assert result["cer"] is None and result["empty_reference_false_positive"] is True
    policy.append({"reference": ref, "prediction": pred, "kind": kind, "strip_whitespace": strip, "score": result})

manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
run = ROOT / "outputs/natural-v4/modal-runs/evaluate-37221188153/natural-natural-v4-evaluate-37221188153-1/review"
generation_path = run / "generations-base.json"
result_path = run / "result.json"
score_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json"
manifest = json.loads(manifest_path.read_text())
raw = json.loads(generation_path.read_text())
run_info = json.loads(result_path.read_text())
saved_scores = json.loads(score_path.read_text())
cases = [row for row in manifest["rows"] if row["split"] == "test" and row["task"] in {"ocr", "ocr_order", "text_presence"}]
records = {(record["id"], record["task"]): record for record in raw}
recomputed = []
counts, successes = Counter(), Counter()
family_mismatches = []
for row in cases:
    record = records[row["id"], row["task"]]
    assert record["reference_answer"] == row["answer"]
    assert record["image"] == row["image"] and record["user"] == row["user"]
    if record["family"] != row["family"]:
        family_mismatches.append({"id": row["id"], "raw_family": record["family"], "manifest_family": row["family"]})
    tokens, eos = record["generated_token_ids"], record["eos_token_ids"]
    complete = bool(tokens and eos and len(tokens) <= 384 and tokens[-1] in eos
                    and record["generated_tokens"] == len(tokens)
                    and record["ended_with_eos"] is True and record["stop_reason"] == "eos"
                    and record["truncated"] is False and record["completion_unknown"] is False)
    assert complete
    strip = row["task"] == "ocr"
    target, prediction = normalized(row["answer"], strip), normalized(record["prediction"], strip)
    passed = target == prediction and complete
    counts[row["task"]] += 1
    successes[row["task"]] += passed
    recomputed.append({"id": row["id"], "task": row["task"], "reference": row["answer"],
                       "prediction": record["prediction"], "complete": complete,
                       "passed": passed, "reference_codepoints": len(target),
                       "edits": edit_distance(target, prediction)})
assert dict(counts) == {"text_presence": 18, "ocr": 10, "ocr_order": 3}
assert dict(successes) == {"text_presence": 18, "ocr": 8, "ocr_order": 1}
for task, group in [("text_presence", "text_presence"), ("ocr", "single_ocr"), ("ocr_order", "ordered_ocr")]:
    assert saved_scores["correct_counts"][group] == successes[task]
    assert saved_scores["denominators"][group] == counts[task]
positive = sum(row["answer"] == "有" for row in cases if row["task"] == "text_presence")
assert positive == 10
assert all(not row["image"].endswith(".h5") and row["image"].startswith(("ocr/commons/", "vision/images/")) for row in cases)
assert sha256(manifest_path.read_bytes()).hexdigest() == run_info["manifest_sha256"]
assert sha256(generation_path.read_bytes()).hexdigest() == saved_scores["artifact_binding"]["generation_file"]["sha256"]

output = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "unicode_database": unicodedata.unidata_version, "device": "cpu", "cuda_available": torch.cuda.is_available()},
    "lesson_stdout": capture.getvalue(), "examples": examples,
    "unit_boundary": {"reference_codepoints": 5, "reference_utf8_bytes": 15, "combining_sequence_codepoints": len("e\u0301")},
    "independent_edit_graph": {"alphabet": "甲乙", "maximum_length": 4, "strings": len(strings), "ordered_pairs": len(strings) ** 2, "mismatches": 0},
    "nfkc": nfkc, "policy_probes": policy,
    "record_recomputation": {"counts": dict(counts), "passed": dict(successes), "presence_positive": positive,
                             "presence_negative": 18 - positive, "records": recomputed,
                             "family_metadata_differences": family_mismatches,
                             "source_bytes": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                                              for p in [manifest_path, generation_path, result_path, score_path]],
                             "original_gpu": {key: run_info[key] for key in ["device", "dtype", "gpu_name", "versions", "seed", "model", "model_revision", "split", "max_tokens", "selected_only"]},
                             "limits": "CPU recomputation of fixed targets/predictions/EOS, not my GPU inference or semantic re-grading. Presence negatives mean no clear Chinese text, not necessarily a blank image. Targets, prompts, image paths and family metadata match; these conditional counts do not establish split independence. No synthetic final-test tasks or arbitrary full-page evaluation."},
}
(OUT / "probe-result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"environment": output["environment"], "lesson_stdout": capture.getvalue(),
                  "independent_edit_graph": output["independent_edit_graph"], "nfkc": nfkc,
                  "fixed_gpu_record_counts": dict(counts), "fixed_gpu_record_passed": dict(successes),
                  "family_metadata_differences": family_mismatches}, ensure_ascii=False, indent=2))
