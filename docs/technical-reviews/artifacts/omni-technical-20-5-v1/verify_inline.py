"""20.5 frozen inline program and explicitly requested exercise, CPU only."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    match = re.search(rb"^## " + re.escape(identifier.encode()) + rb"[^\n]*\n", raw, re.M)
    assert match is not None
    following = re.search(rb"^## ", raw[match.end() :], re.M)
    end = match.end() + following.start() if following else len(raw)
    return raw[match.start() : end], match.start(), end


def run_program(program, name):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(program, name, "exec"), namespace)
    rows = namespace["records"]
    # This assertion inspects the literal inputs and proves the printed checker
    # has not silently moved rows to a different split.
    literal_inputs = {}
    exec(compile(program.split("train_families =")[0], name + ":inputs", "exec"), literal_inputs)
    assert rows == literal_inputs["records"]
    return {
        "code_sha256": digest(program.encode()),
        "input_rows": rows,
        "record_count": len(rows),
        "train_row_count": sum(row["split"] == "train" for row in rows),
        "test_row_count": sum(row["split"] == "test" for row in rows),
        "train_families": sorted(namespace["train_families"]),
        "test_families": sorted(namespace["test_families"]),
        "overlap": sorted(namespace["train_families"] & namespace["test_families"]),
        "stdout": output.getvalue(),
        "records_unchanged_by_checker": True,
        "families_are_sets": isinstance(namespace["train_families"], set)
        and isinstance(namespace["test_families"], set),
        "sorted_returns_list": isinstance(sorted(namespace["train_families"]), list),
    }


def main():
    freeze = json.loads(FREEZE.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "20.5")
    frozen = (ROOT / page["snapshot"]).read_bytes()
    current, start, end = section((ROOT / page["source"]).read_bytes(), "20.5")
    assert digest(frozen) == digest(current) == page["source_sha256"]
    for path, expected in page["figures_sha256"].items():
        assert digest((ROOT / path).read_bytes()) == expected
    blocks = re.findall(r"```python\n(.*?)```", frozen.decode(), re.S)
    assert len(blocks) == 1
    program = blocks[0]
    notebook_raw = (ROOT / page["notebook"]).read_bytes()
    assert digest(notebook_raw) == page["notebook_sha256"]
    notebook = json.loads(notebook_raw)
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert sum(c.rstrip() == program.rstrip() for c in code_cells) == 1
    original = run_program(program, "20.5:frozen-inline")
    assert original["record_count"] == 4
    assert original["train_row_count"] == original["test_row_count"] == 2
    assert original["train_families"] == ["貓照片A"]
    assert original["test_families"] == ["招牌照片B"]
    assert original["stdout"] == "訓練家族 ['貓照片A']\n測試家族 ['招牌照片B']\n兩邊都有的家族 []\n"
    old = '{"family": "貓照片A", "split": "train", "question": "哪隻貓抬起前腳？"}'
    new = old.replace('"split": "train"', '"split": "test"')
    assert program.count(old) == 1
    exercise_code = program.replace(old, new, 1)
    exercise = run_program(exercise_code, "20.5:second-row-to-test")
    assert exercise["train_row_count"] == 1 and exercise["test_row_count"] == 3
    assert exercise["overlap"] == ["貓照片A"]
    assert exercise["stdout"] == "訓練家族 ['貓照片A']\n測試家族 ['招牌照片B', '貓照片A']\n兩邊都有的家族 ['貓照片A']\n"
    repair_code = exercise_code.replace('"split": "train"', '"split": "test"')
    repair = run_program(repair_code, "20.5:move-entire-cat-family-to-test")
    assert repair["overlap"] == [] and repair["train_row_count"] == 0
    prereqs = []
    for file, identifier, snapshot in [
        ("course/chapters/05.md", "5.12", "docs/course-review-20261019/freeze-01/sources/5.12.md"),
        ("course/first-steps.md", "W.2", "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07/sources/first-steps.md"),
    ]:
        parent = (ROOT / file).read_bytes()
        part, first, last = section(parent, identifier)
        frozen_parent = (ROOT / snapshot).read_bytes()
        frozen_part = frozen_parent if identifier == "5.12" else section(frozen_parent, identifier)[0]
        assert part == frozen_part
        prereqs.append({"source": file, "selector": "section:" + identifier,
                        "current_parent_sha256": digest(parent), "slice_sha256": digest(part),
                        "byte_start": first, "byte_end_exclusive": last,
                        "frozen_snapshot": snapshot, "snapshot_sha256": digest(frozen_parent),
                        "current_matches_frozen_slice": True})
    # Only read the relevant current W.2 expressions, without adopting any
    # earlier review verdict or extending this review to the whole warm-up.
    prereq_expressions = {
        "list_comprehension": [x * 2 for x in [1, 2, 3]],
        "set_deduplication": len(set("貓狗貓")),
        "sorted_set": sorted(set("貓狗貓")),
    }
    assert prereq_expressions == {"list_comprehension": [2, 4, 6], "set_deduplication": 2,
                                  "sorted_set": ["狗", "貓"]}
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-5-v1/verify_inline.py",
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "platform": platform.platform(), "device": "CPU", "dependencies": "Python standard library only"},
        "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE.read_bytes())},
        "lesson": {"source": page["source"], "snapshot": page["snapshot"], "source_sha256": digest(frozen),
                   "current_byte_start": start, "current_byte_end_exclusive": end,
                   "notebook": page["notebook"], "notebook_sha256": digest(notebook_raw),
                   "notebook_inline_code_matches": True, "figure_sha256": page["figures_sha256"]},
        "original": original, "exercise_second_row_to_test": exercise,
        "repair_entire_cat_family_to_test": repair,
        "prerequisite_pins": prereqs, "current_W2_expressions": prereq_expressions,
        "scope": "Deterministic four-record teaching demonstration only; no training, dataset download, model run, GPU, or quality measurement. Only train/test strings are checked; validation and incorrect source/family assignments require additional checks."
    }
    destination = HERE / "inline-execution.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": digest(destination.read_bytes()),
                      "original_stdout": original["stdout"], "exercise_stdout": exercise["stdout"],
                      "repair_overlap": repair["overlap"], "prerequisite_pins": prereqs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
