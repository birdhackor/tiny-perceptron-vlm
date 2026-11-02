"""Same-reviewer W.2 callback: compare exact snapshots and run the current examples."""

from datetime import UTC, datetime
from pathlib import Path
import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
OLD = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3/sections/W.2.md"
NEW = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/W.2.md"
ORIGINAL = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_2/technical-original-v1.json"
OLD_SHA = "e2fd0349c4fe6400594861ef81bab3533323cb0f34dd63855bc7ebe33fae00bd"
NEW_SHA = "6312a579f5eb8b278b01c954381808d3aa8b91e6cfcf0389c4a863ca4298f4ae"
ORIGINAL_SHA = "1c6e95a1e85e039c0b86eb084620197c4761906509fd6541cf16a50dce181fe0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(OLD) == OLD_SHA and sha(NEW) == NEW_SHA and sha(ORIGINAL) == ORIGINAL_SHA
report = json.loads(ORIGINAL.read_text())
assert len(report["claims"]) == 74
original_evidence = []
for artifact in report["artifacts"]:
    path = ROOT / artifact["path"]
    assert sha(path) == artifact["sha256"]
    original_evidence.append({"path": artifact["path"], "sha256": sha(path)})

old, new = OLD.read_text(), NEW.read_text()
old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
assert len(old_lines) == len(new_lines)
changed_lines = [i + 1 for i, (before, after) in enumerate(zip(old_lines, new_lines)) if before != after]
assert changed_lines == [71]
old_phrase = "把錯誤訊息存在 `error`，改設"
new_phrase = "把這次的錯誤物件暫時叫作 `error`，由 `print` 顯示它的訊息，再設"
assert old.replace(old_phrase, new_phrase) == new

fence = chr(96) * 3
old_blocks = re.findall(fence + r"python\n(.*?)" + fence, old, re.S)
new_blocks = re.findall(fence + r"python\n(.*?)" + fence, new, re.S)
assert len(new_blocks) == 2 and old_blocks == new_blocks
assert [s for s in old_lines if s.startswith("|")] == [s for s in new_lines if s.startswith("|")]
assert old_lines[22] == new_lines[22] and old_lines[24] == new_lines[24]

working = (ROOT / "course/first-steps.md").read_text()
start = working.index("## W.2 ")
end = working.find("\n## ", start + 1)
assert working[start:] == new if end < 0 else working[start:end + 1] == new
canonical = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
section_freeze = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"
canonical_data = json.loads(canonical.read_text())
section_data = json.loads(section_freeze.read_text())
assert section_data["canonical_source_freeze"]["path"] == str(canonical.relative_to(ROOT))
assert section_data["canonical_source_freeze"]["sha256"] == sha(canonical)
for row in section_data["pages"]:
    if row["page_id"] in ("W.1", "W.2"):
        assert sha(ROOT / row["snapshot"]) == row["source_sha256"]
        assert row["figures_sha256"] == {}
for row in canonical_data["pages"]:
    if row["page_id"] == "first-steps":
        canonical_raw = (ROOT / row["snapshot"]).read_bytes()
        assert hashlib.sha256(canonical_raw).hexdigest() == row["source_sha256"]
        assert canonical_raw == (ROOT / "course/first-steps.md").read_bytes()

previous = json.loads((ART / "execution.json").read_text())
records = {row["name"]: row for row in previous["records"]}
observations = []


def run(name, source, expected):
    out = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(out):
        exec(compile(source, name, "exec"), namespace)
    assert out.getvalue() == expected
    observations.append({"name": name, "input_source": source, "expected_stdout": expected,
                         "observed_stdout": out.getvalue(), "passed": True})
    return namespace


run("current_block_1", new_blocks[0], records["frozen_block_1"]["expected_stdout"])
run("current_block_2", new_blocks[1], records["frozen_block_2"]["expected_stdout"])
exercise = new_blocks[0].replace('animals = ["貓", "狗"]', 'animals = ["鳥", "貓", "狗"]').replace(
    "assert len(animals) == 2", "assert len(animals) == 3")
assert exercise == records["exercise_three_animals"]["input_source"]
run("current_exercise_animals", exercise, records["exercise_three_animals"]["expected_stdout"])
caption = exercise.replace('"caption": "貓在睡覺"', '"caption": "貓在吃飯"')
assert caption == records["exercise_caption_only"]["input_source"]
run("current_exercise_caption", caption, records["exercise_caption_only"]["expected_stdout"])
environment = {"scores": [5, 9, 7], "animal": "貓"}
for row in previous["records"]:
    if "expression" in row:
        assert row["expression"] in new
        expected = ast.literal_eval(row["expected_repr"])
        value = eval(row["expression"], environment)
        assert value == expected and type(value) is type(expected)
        observations.append({"name": "current_" + row["name"], "expression": row["expression"],
                             "expected_repr": row["expected_repr"], "observed_repr": repr(value), "passed": True})
assert sum("expression" in row for row in observations) == 22
assert "`def first(items):`" in new and "`return items[0]`" in new and "`first(animals)`" in new
run("current_inline_first", records["inline_first_function"]["input_source"], "貓\n")

out = io.StringIO()
exception = {}
with contextlib.redirect_stdout(out):
    try:
        {"貓": 0}["狗"]
    except KeyError as error:
        assert isinstance(error, KeyError) and not isinstance(error, str)
        print("找不到", error)
        exception = {"bound_type": type(error).__name__, "is_str": isinstance(error, str), "str_value": str(error)}
assert "error" not in globals()
assert out.getvalue() == "找不到 '狗'\n"
exception.update(observed_print=out.getvalue(), binding_cleared_after_handler=True)
assert ord("狗") == 29399 and ord("貓") == 35987
assert (3 << 4) | 5 == 53 and (53 >> 4) == 3 and (53 & 15) == 5
assert isinstance(torch.nn.Module, type) and issubclass(torch.nn.Linear, torch.nn.Module)

source_receipt = json.loads((ART / "source-inspection.json").read_text())
official_integrity = []
for row in source_receipt["official_originals"]:
    assert sha(Path(row["local_cache"])) == row["sha256"]
    official_integrity.append({"upstream_path": row["upstream_path"], "sha256": row["sha256"]})
module_path = Path(inspect.getsourcefile(torch.nn.Module))
assert sha(module_path) == source_receipt["installed_torch_Module_source"]["sha256"]
for source in report["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]
old_w1 = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3/sections/W.1.md"
new_w1 = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/W.1.md"
assert old_w1.read_bytes() == new_w1.read_bytes()
print(json.dumps({"created_at": datetime.now(UTC).isoformat(), "reviewer_task": report["reviewer_task"],
                  "old_source_sha256": OLD_SHA, "current_source_sha256": NEW_SHA,
                  "canonical_freeze_sha256": sha(canonical), "section_freeze_sha256": sha(section_freeze),
                  "changed_source_lines": changed_lines, "old_source_quote": old_lines[70].strip(),
                  "current_source_quote": new_lines[70].strip(), "code_blocks_byte_equal": True,
                  "table_rows_byte_equal": True, "inline_function_and_exercise_byte_equal": True,
                  "W1_byte_equal": True, "linked_repository_sources_byte_equal": True,
                  "original_report_and_evidence_preserved": original_evidence,
                  "official_cached_originals_unchanged": official_integrity,
                  "exception_binding_diagnostic": exception, "observations": observations,
                  "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu",
                                  "platform": platform.platform(), "optimize": str(sys.flags.optimize)},
                  "all_passed": True,
                  "limits": "No browser keypress, Colab, GPU, training, or full linked chapter/figure review."},
                 ensure_ascii=False, indent=2))
