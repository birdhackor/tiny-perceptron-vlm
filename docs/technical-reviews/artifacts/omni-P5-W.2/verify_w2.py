"""Reviewer-owned exact-frozen W.2 checks; no model construction or forward."""

import ast
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v5/sections/W.2.md"
raw = FROZEN.read_bytes()
body = raw.decode("utf-8")
assert hashlib.sha256(raw).hexdigest() == "6312a579f5eb8b278b01c954381808d3aa8b91e6cfcf0389c4a863ca4298f4ae"
current = (ROOT / "course/first-steps.md").read_text()
start = current.index("## W.2 ")
end = current.index("## W.3 ", start)
assert current[start:end].encode() == raw
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 2
checks = []


def execute(label, code, expected):
    stream = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, str(FROZEN) + ":" + label, "exec"), namespace)
    observed = stream.getvalue()
    assert observed == expected, (label, observed, expected)
    checks.append({"id": label, "input_code": code, "expected_stdout": expected,
                   "observed_stdout": observed, "matches": True})
    return namespace


base = execute("original_block_1", blocks[0], "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n")
execute("original_block_2", blocks[1], "16\n找不到 '狗'\n-1\n53 3 5\n")
practice = blocks[0].replace('["貓", "狗"]', '["鳥", "貓", "狗"]')
practice = practice.replace("assert len(animals) == 2", "assert len(animals) == 3")
practice_ns = execute("exercise_three_animals", practice,
                      "鳥 3\n貓在睡覺\n看到 鳥\n看到 貓\n看到 狗\n")
changed = practice.replace('"caption": "貓在睡覺"', '"caption": "貓在窗邊"')
changed_ns = execute("exercise_caption_only", changed,
                     "鳥 3\n貓在窗邊\n看到 鳥\n看到 貓\n看到 狗\n")
assert practice_ns["animals"] == changed_ns["animals"]
assert practice_ns["animals"] is not practice_ns["record"]
checks.append({"id": "caption_independence", "observed": repr(changed_ns["animals"]),
               "expected": "['鳥', '貓', '狗']", "matches": True})

context = {"scores": [5, 9, 7], "animal": "貓"}
table_rows = []
for line_number, line in enumerate(body.splitlines(), 1):
    if line.startswith("| "):
        for expression, expected_text in re.findall(r"`([^`]+)` → `([^`]+)`", line):
            expected = ast.literal_eval(expected_text)
            observed = eval(compile(expression, str(FROZEN) + ":table", "eval"), context)
            assert observed == expected and type(observed) is type(expected)
            table_rows.append({"line": line_number, "expression": expression,
                               "expected": expected_text, "observed": repr(observed),
                               "type": type(observed).__name__, "matches": True})
assert len(table_rows) == 21, len(table_rows)

extra_expressions = [
    ("animals[0]", "貓"), ("animals[1]", "狗"),
    ('record["caption"]', "貓在睡覺"), ("len(animals)", 2),
    ("ord('狗')", 29399), ("ord('貓')", 35987), ("-7 // 2", -4),
    ("3 << 4", 48), ("(3 << 4) | 5", 53),
    ("53 >> 4", 3), ("53 & 15", 5),
    ("format(3, 'b')", "11"), ("format(48, 'b')", "110000"),
    ("format(5, 'b')", "101"), ("format(53, 'b')", "110101"),
    ("format(15, 'b')", "1111"),
]
for expression, expected in extra_expressions:
    observed = eval(expression, base)
    assert observed == expected and type(observed) is type(expected)
    checks.append({"id": expression, "expected": repr(expected),
                   "observed": repr(observed), "matches": True})

exec("def first(items):\n    return items[0]\n", base)
assert base["first"](base["animals"]) == "貓"
checks.append({"id": "def_return_call", "code": "def first(items):\n    return items[0]\nfirst(animals)",
               "expected": "'貓'", "observed": repr(base["first"](base["animals"])), "matches": True})
silent = io.StringIO()
with contextlib.redirect_stdout(silent):
    exec("assert len(animals) == 2", base)
assert silent.getvalue() == ""
try:
    exec("assert len(animals) == 3", base)
except AssertionError as error:
    failure = type(error).__name__
else:
    raise RuntimeError("An expected false assertion did not fail")
checks.append({"id": "assert_true_and_false", "true_stdout": silent.getvalue(),
               "false_exception": failure, "debug": __debug__, "matches": True})

numbers = [3, 1, 2]
assert sorted(numbers) == [1, 2, 3] and numbers == [3, 1, 2]
checks.append({"id": "sorted_preserves_input", "before": [3, 1, 2],
               "sorted": sorted(numbers), "after": numbers, "matches": True})
visits = []
generator = (visits.append(x) or x * 2 for x in [1, 2, 3])
assert visits == []
first_value = next(generator)
assert first_value == 2 and visits == [1]
checks.append({"id": "generator_laziness", "before_next": [],
               "first_value": first_value, "visits_after_next": visits.copy(), "matches": True})
duplicate = {"貓": 1, "貓": 2}
assert len(duplicate) == 1 and duplicate["貓"] == 2
checks.append({"id": "dictionary_unique_keys_scope", "code": '{"貓": 1, "貓": 2}',
               "observed": repr(duplicate), "note": "Keys are unique in the resulting dictionary; duplicate literals overwrite rather than raise.", "matches": True})

import torch

module_text = Path(inspect.getfile(torch.nn.Module)).read_text()
linear_text = Path(inspect.getfile(torch.nn.Linear)).read_text()
assert "class Module:" in module_text and "class Linear(Module):" in linear_text
assert isinstance(torch.nn.Linear, type) and issubclass(torch.nn.Linear, torch.nn.Module)
checks.append({"id": "import_torch_and_module_classes", "torch_version": torch.__version__,
               "module_file": inspect.getfile(torch.nn.Module),
               "module_file_sha256": hashlib.sha256(module_text.encode()).hexdigest(),
               "linear_file": inspect.getfile(torch.nn.Linear),
               "linear_file_sha256": hashlib.sha256(linear_text.encode()).hexdigest(),
               "observed": "import torch binds module; nn.Module and nn.Linear are Python classes; Linear subclasses Module.",
               "matches": True, "model_instantiated": False, "neural_forward_executed": False})

result = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_w_2",
    "source": str(FROZEN.relative_to(ROOT)), "source_bytes": len(raw),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "environment": {"python": sys.version, "implementation": platform.python_implementation(),
                    "platform": platform.platform(), "optimization_level": str(sys.flags.optimize),
                    "torch": torch.__version__, "device": "CPU; Python builtins and import/introspection only"},
    "exact_fenced_blocks": len(blocks), "table_expression_count": len(table_rows),
    "table_expressions": table_rows, "checks": checks,
    "scope": "Exact frozen W.2 builtins, stated prose snippets, practice variants, Unicode and binary arithmetic. No model construction, neural inference, training, download, workflow, or publication.",
    "all_matches": True,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
