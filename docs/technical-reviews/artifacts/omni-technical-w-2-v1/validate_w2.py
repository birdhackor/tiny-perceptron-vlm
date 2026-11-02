"""Bounded CPU verification of the exact frozen W.2 examples, by its reviewer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3/sections/W.2.md"
EXPECTED_SHA = "e2fd0349c4fe6400594861ef81bab3533323cb0f34dd63855bc7ebe33fae00bd"
raw = SNAPSHOT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
text = raw.decode()
fence = chr(96) * 3
blocks = re.findall(fence + r"python\n(.*?)" + fence, text, re.S)
assert len(blocks) == 2
records: list[dict] = []


def run_block(name: str, source: str, expected: str) -> dict:
    output = io.StringIO()
    namespace: dict = {}
    with contextlib.redirect_stdout(output):
        exec(compile(source, name, "exec"), namespace)
    observed = output.getvalue()
    assert observed == expected, (name, observed, expected)
    records.append({"name": name, "input_source": source, "expected_stdout": expected,
                    "observed_stdout": observed, "passed": True})
    return namespace


base = run_block("frozen_block_1", blocks[0], "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n")
exercise = blocks[0].replace('animals = ["貓", "狗"]', 'animals = ["鳥", "貓", "狗"]').replace(
    "assert len(animals) == 2", "assert len(animals) == 3")
assert exercise != blocks[0]
run_block("exercise_three_animals", exercise, "鳥 3\n貓在睡覺\n看到 鳥\n看到 貓\n看到 狗\n")
changed = exercise.replace('"caption": "貓在睡覺"', '"caption": "貓在吃飯"')
run_block("exercise_caption_only", changed, "鳥 3\n貓在吃飯\n看到 鳥\n看到 貓\n看到 狗\n")
run_block("frozen_block_2", blocks[1], "16\n找不到 '狗'\n-1\n53 3 5\n")

# Table expressions are transcribed exactly; expected values are checked by value and type.
examples = [
    ("slice_drop_last", "[5, 2, 7][:-1]", [5, 2]),
    ("slice_from_one", "[5, 2, 7][1:]", [2, 7]),
    ("range", "list(range(3))", [0, 1, 2]),
    ("set_count", 'len(set("貓狗貓"))', 2),
    ("sorted", "sorted([3, 1, 2])", [1, 2, 3]),
    ("sorted_reverse", "sorted([3, 1, 2], reverse=True)", [3, 2, 1]),
    ("sorted_unicode", 'sorted(set("貓狗貓"))', ["狗", "貓"]),
    ("max_len", 'max(["貓", "小花貓"], key=len)', "小花貓"),
    ("max_index", "max(range(3), key=lambda i: scores[i])", 1),
    ("zip", 'list(zip(["貓", "狗"], [3, 1]))', [("貓", 3), ("狗", 1)]),
    ("dict_zip", 'dict(zip(["貓", "狗"], [0, 1]))', {"貓": 0, "狗": 1}),
    ("enumerate", 'list(enumerate(["貓", "狗"]))', [(0, "貓"), (1, "狗")]),
    ("reversed", 'list(reversed(["貓", "狗", "鳥"]))', ["鳥", "狗", "貓"]),
    ("join_empty", '"".join(["貓", "看", "狗"])', "貓看狗"),
    ("join_separator", '"、".join(["貓", "狗"])', "貓、狗"),
    ("list_comprehension", "[x * 2 for x in [1, 2, 3]]", [2, 4, 6]),
    ("generator_sum", "sum(x * 2 for x in [1, 2, 3])", 12),
    ("next_match", "next(x for x in [3, 8, 12] if x > 5)", 8),
    ("next_default", "next((x for x in [3] if x > 5), None)", None),
    ("f_string", 'f"看到{animal}"', "看到貓"),
    ("floor_positive", "7 // 2", 3),
    ("floor_negative", "-7 // 2", -4),
]
env = {"scores": [5, 9, 7], "animal": "貓"}
for name, expression, expected in examples:
    assert expression in text, expression
    observed = eval(expression, env)
    assert observed == expected and type(observed) is type(expected), name
    records.append({"name": name, "expression": expression, "expected_repr": repr(expected),
                    "observed_repr": repr(observed), "type": type(observed).__name__, "passed": True})

run_block("inline_first_function", 'animals = ["貓", "狗"]\ndef first(items):\n    return items[0]\nprint(first(animals))\n', "貓\n")
values = [3, 1, 2]
assert sorted(values) == [1, 2, 3] and values == [3, 1, 2]
records.append({"name": "sorted_does_not_mutate", "observed_original": values, "passed": True})
assert ord("狗") == 29399 and ord("貓") == 35987
records.append({"name": "unicode_code_points", "狗": ord("狗"), "貓": ord("貓"), "passed": True})
try:
    exec("assert len([1]) == 2")
except AssertionError:
    records.append({"name": "assert_failure_normal_mode", "exception": "AssertionError", "passed": True})
else:
    raise AssertionError("false assertion did not raise")

# Scope diagnostics, not replacements for the frozen examples.
assert isinstance(("貓", 3), tuple) and isinstance(("貓"), str)
records.append({"name": "tuple_scope", "pair_type": type(("貓", 3)).__name__,
                "parenthesized_single_type": type(("貓")).__name__, "passed": True})
assert {"caption": "a", "caption": "b"} == {"caption": "b"}
records.append({"name": "dictionary_key_uniqueness_scope", "duplicate_literal_result": {"caption": "a", "caption": "b"}, "passed": True})
try:
    {"貓": 0}["狗"]
except KeyError as error:
    assert isinstance(error, KeyError) and not isinstance(error, str)
    records.append({"name": "except_binding_type", "type": type(error).__name__,
                    "message": str(error), "is_str": isinstance(error, str), "passed": True})

import torch

torch.set_num_threads(1)
records.append({"name": "import_torch", "version": torch.__version__, "module_file": torch.__file__,
                "nn_Module_is_class": isinstance(torch.nn.Module, type),
                "nn_Linear_is_Module_subclass": issubclass(torch.nn.Linear, torch.nn.Module), "passed": True})

print(json.dumps({"created_at": datetime.now(UTC).isoformat(), "snapshot": str(SNAPSHOT.relative_to(ROOT)),
                  "snapshot_sha256": EXPECTED_SHA, "environment": {"python": sys.version, "implementation": platform.python_implementation(),
                  "platform": platform.platform(), "torch": torch.__version__, "device": "cpu", "optimize": str(sys.flags.optimize)},
                  "records": records, "all_passed": True}, ensure_ascii=False, indent=2))
