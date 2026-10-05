"""Independent finite copying checks, not a model evaluation or a course helper."""
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = (ROOT / "original-section/section.md").read_bytes()
body = raw.decode("utf-8")
rows = []
for line in body.splitlines():
    match = re.fullmatch(r"\| ([1-4]) \| (中文|英文) \| (.+) \|", line)
    if match:
        rows.append({"row": int(match[1]), "language": match[2], "text": match[3]})
assert len(rows) == 4 and [row["row"] for row in rows] == [1, 2, 3, 4]
fence_raw = re.search(rb"```text\n(.*?)```", raw, re.S)[1]
(ROOT / "original-text-fence.txt").write_bytes(fence_raw)
zh = [row["text"] for row in rows if row["language"] == "中文"]
english = [row["text"] for row in rows if row["language"] == "英文"]
all_text = [row["text"] for row in rows]
target = "\n".join(zh)
# The final newline is the fence's line delimiter, retained in the raw snapshot.
assert fence_raw == (target + "\n").encode("utf-8")
phrase_pattern = re.compile("|".join(map(re.escape, all_text)))


def inspect(answer, wanted):
    matched = phrase_pattern.findall(answer)
    leftover = phrase_pattern.sub("", answer).strip()
    wanted_in_answer = [text for text in matched if text in wanted]
    return {
        "contains_今日休館": "今日休館" in answer,
        "required_source_text_complete_ordered": wanted_in_answer == wanted,
        "range_complete_and_no_unrequested_text": matched == wanted and not leftover,
        "one_source_item_per_line": answer.splitlines() == matched,
        "exact_target": answer == "\n".join(wanted),
        "matched_source_phrases": matched,
        "non_source_extra_text": leftover,
    }


cases = {
    "original_target": target,
    "all_four_source_lines": "\n".join(all_text),
    "prefixed_explanation": "以下是中文版本\n" + target,
    "only_first_chinese_line": zh[0],
    "both_chinese_on_one_line": " ".join(zh),
    "reversed_chinese_order": "\n".join(reversed(zh)),
}
results = {name: {"answer": value, **inspect(value, zh)} for name, value in cases.items()}
assert results["original_target"]["exact_target"]
for name in ["all_four_source_lines", "prefixed_explanation", "only_first_chinese_line"]:
    assert results[name]["contains_今日休館"] and not results[name]["exact_target"]
assert results["all_four_source_lines"]["required_source_text_complete_ordered"]
assert not results["all_four_source_lines"]["range_complete_and_no_unrequested_text"]
assert not results["only_first_chinese_line"]["required_source_text_complete_ordered"]
assert results["both_chinese_on_one_line"]["required_source_text_complete_ordered"]
assert results["both_chinese_on_one_line"]["range_complete_and_no_unrequested_text"]
assert not results["both_chinese_on_one_line"]["one_source_item_per_line"]
assert not results["reversed_chinese_order"]["exact_target"]
first_request = [all_text[0]]
english_request = english
tomorrow_request = [row["text"] for row in rows if row["row"] in [3, 4]]
variants = {
    "only_first_line": {"target": first_request[0], "original_two_line_answer_valid": inspect(target, first_request)["exact_target"]},
    "only_english": {"target": "\n".join(english_request), "rows": [row["row"] for row in rows if row["language"] == "英文"]},
    "only_tomorrow_schedule_literal_rows": {"target": "\n".join(tomorrow_request), "rows": [3, 4]},
}
assert not variants["only_first_line"]["original_two_line_answer_valid"]
assert variants["only_english"]["rows"] == [2, 4]
output = {
    "scope": "Literal table-derived hand-written examples; no learned model, weights, inference, optimizer, OCR, EOS/stop-reason or empirical rate is evaluated.",
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "raw_fence_sha256": hashlib.sha256(fence_raw).hexdigest(),
    "raw_fence_bytes": len(fence_raw),
    "material_rows": rows,
    "required_chinese_rows": [1, 3],
    "required_items": len(zh),
    "original_cases": results,
    "changed_request_cases": variants,
    "assertions": "all passed",
    "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "device": "CPU; strings only", "cwd": str(Path.cwd())},
}
(ROOT / "copy-contract-results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
