"""核對凍結教材的相鄰對、合併路徑、平手次序與還原性。"""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import re
import sys


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/6.2.md"
raw = SOURCE.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {}
original_stdout = io.StringIO()
with redirect_stdout(original_stdout):
    exec(compile(code, str(SOURCE), "exec"), namespace)
merge = namespace["merge"]


def count_pairs(rows):
    counts = Counter()
    for row in rows:
        counts.update(zip(row, row[1:]))
    return counts


def trace_merge(row, pair):
    function = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef))
    while_node = next(node for node in ast.walk(function) if isinstance(node, ast.While))
    checks = []

    def trace(frame, event, arg):
        if frame.f_code is merge.__code__ and event == "line" and frame.f_lineno == while_node.lineno:
            checks.append({"i": frame.f_locals["i"], "out": frame.f_locals["out"][:]})
        return trace

    sys.settrace(trace)
    try:
        result = merge(row, pair)
    finally:
        sys.settrace(None)
    return {"input": row, "pair": pair, "while_checks": checks, "result": result}


rows = [list("abab"), list("abac")]
counts = count_pairs(rows)
selected = counts.most_common(1)[0][0]
base_result = [merge(row, selected) for row in rows]
assert counts == Counter({("a", "b"): 3, ("b", "a"): 2, ("a", "c"): 1})
assert base_result == [["ab", "ab"], ["ab", "a", "c"]]
assert [len(row) for row in base_result] == [2, 3]
assert count_pairs([list("ab"), list("ac")]) == Counter({("a", "b"): 1, ("a", "c"): 1})

exercise_rows = rows + [list("acacacac")]
exercise_counts = count_pairs(exercise_rows)
exercise_selected = exercise_counts.most_common(1)[0][0]
exercise_result = [merge(row, exercise_selected) for row in exercise_rows]
assert exercise_counts == Counter({("a", "c"): 5, ("a", "b"): 3, ("c", "a"): 3, ("b", "a"): 2})
assert exercise_selected == ("a", "c")
assert exercise_result == [list("abab"), ["a", "b", "ac"], ["ac"] * 4]

forward_tie = count_pairs([list("ab"), list("ac")]).most_common()
reverse_tie = count_pairs([list("ac"), list("ab")]).most_common()
assert forward_tie[0][0] == ("a", "b")
assert reverse_tie[0][0] == ("a", "c")
missing = Counter()
before = missing[("x", "y")]
missing[("x", "y")] += 1
assert before == 0 and missing[("x", "y")] == 1
assert list("abab") == ["a", "b", "a", "b"]
assert rows[0][1:] == ["b", "a", "b"]
assert list(zip(rows[0], rows[0][1:])) == [("a", "b"), ("b", "a"), ("a", "b")]
assert rows[0][:2] != ("a", "b")
assert tuple(rows[0][:2]) == ("a", "b")
assert "".join(selected) == "ab"

traces = [trace_merge(row, selected) for row in rows]
assert [step["i"] for step in traces[0]["while_checks"]] == [0, 2, 4]
assert [step["i"] for step in traces[1]["while_checks"]] == [0, 2, 3, 4]

cases = 0
for length in range(7):
    for letters in itertools.product("abc", repeat=length):
        word = "".join(letters)
        for pair in itertools.product("abc", repeat=2):
            result = merge(list(word), pair)
            assert "".join(result) == word
            assert len(result) == len(word) - word.count("".join(pair))
            cases += 1
assert merge(list("aaaaa"), ("a", "a")) == ["aa", "aa", "a"]

multistep_rows = [row[:] for row in rows]
merges = []
while counts := count_pairs(multistep_rows):
    pair = counts.most_common(1)[0][0]
    multistep_rows = [merge(row, pair) for row in multistep_rows]
    merges.append({"pair": pair, "result": multistep_rows})
    assert ["".join(row) for row in multistep_rows] == ["abab", "abac"]
assert len(merges) == 4

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-2/verify_bpe.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; standard library only"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "fence_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "original_stdout": original_stdout.getvalue(),
    "base": {"counts": [[list(pair), n] for pair, n in count_pairs(rows).items()], "most_common": count_pairs(rows).most_common(1), "selected": selected, "result": base_result, "lengths": [len(row) for row in base_result]},
    "exercise": {"counts": [[list(pair), n] for pair, n in exercise_counts.items()], "selected": exercise_selected, "result": exercise_result},
    "counter_ties": {"forward": forward_tie, "reverse": reverse_tie},
    "missing_count": {"before": before, "after_increment": missing[("x", "y")]},
    "sequence_operations": {"list": list("abab"), "slice": rows[0][1:], "zip": list(zip(rows[0], rows[0][1:])), "list_equals_tuple": rows[0][:2] == selected, "converted_tuple_equals_pair": tuple(rows[0][:2]) == selected, "join": "".join(selected)},
    "merge_traces": traces,
    "exhaustive": {"alphabet": "abc", "lengths": [0, 6], "pair_count_per_word": 9, "cases": cases, "properties": ["串接後等於原文", "輸出項數等於輸入項數減去非重疊命中次數"], "status": "pass"},
    "overlap": {"input": "aaaaa", "pair": ["a", "a"], "result": merge(list("aaaaa"), ("a", "a"))},
    "multistep": {"merges": merges, "termination": "每份文件各剩一個符號，沒有相鄰對", "roundtrip": ["".join(row) for row in multistep_rows]},
    "byte_alphabet_size": 2**8,
    "result": "所有斷言通過",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
