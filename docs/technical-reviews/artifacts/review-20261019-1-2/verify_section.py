"""核對文件切分、Python 語意與教材算例；不執行訓練。"""

import ast
import hashlib
import inspect
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.data import split_documents, toy_documents


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boundaries(n):
    raw_a, raw_b = max(1, int(n * 0.8)), max(2, int(n * 0.9))
    b = min(raw_b, n - 1)
    a = min(raw_a, b - 1)
    return {"n": n, "raw": [raw_a, raw_b], "final": [a, b], "sizes": [a, b-a, n-b]}


frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/1.2.md"
body = frozen.read_text()
fences = re.findall(r"```python\n(.*?)```", body, flags=re.S)
repository_function = inspect.getsource(split_documents)
excerpt_tree = ast.parse(fences[1])
assert ast.dump(excerpt_tree) == ast.dump(ast.parse(repository_function))

docs = ["貓看狗。", "狗看貓。", "鳥看魚。", "魚看鳥。", "貓看狗。"]
unique = list(dict.fromkeys(docs))
parts = split_documents(docs, seed=42)
assert unique == ["貓看狗。", "狗看貓。", "鳥看魚。", "魚看鳥。"]
assert parts == {"train": ["鳥看魚。", "狗看貓。"], "validation": ["魚看鳥。"], "test": ["貓看狗。"]}
assert list(parts.items()) == [(name, parts[name]) for name in ("train", "validation", "test")]
assert not (set(parts["train"]) & set(parts["validation"]))
assert set(["貓", "貓", "狗"]) == {"貓", "狗"}
assert ({"貓", "狗"} & {"狗", "鳥"}) == {"狗"}
assert not set() and not not {"狗"}
sorted_characters = sorted(set("貓看狗，狗看貓。"))
assert sorted_characters == ["。", "狗", "看", "貓", "，"]

shuffled = unique.copy()
shuffle_return = random.Random(42).shuffle(shuffled)
assert shuffle_return is None
assert shuffled == ["鳥看魚。", "狗看貓。", "魚看鳥。", "貓看狗。"]
assert parts == split_documents(docs)
assert split_documents(docs + ["貓看狗。"]) == parts
novel_parts = split_documents(docs + ["鳥看貓。"])
assert [len(x) for x in novel_parts.values()] == [3, 1, 1]

namespace = {"random": random, "split_documents": split_documents}
exec(compile(fences[1], "frozen-function", "exec"), namespace)
excerpt_function = namespace["split_documents"]
assert excerpt_function is not split_documents
assert excerpt_function(docs) == parts
assert excerpt_function.__defaults__ == (42,)
assert excerpt_function.__doc__ == ast.get_docstring(excerpt_tree.body[0])
assert len(repository_function.splitlines()) == 11
assert int(3.2) == 3 and int(3.6) == 3 and int(-3.2) == -3
assert max(1, 0) == 1 and min(2, 3) == 2 and len(unique) == 4
first, second = max(1, 3), max(2, 3)
assert (first, second) == (3, 3)
assert shuffled[:2] + shuffled[2:3] + shuffled[3:] == shuffled

errors = []
for n in range(3):
    try:
        split_documents([str(i) for i in range(n)] * 2)
    except ValueError as error:
        errors.append({"unique_documents": n, "type": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("少於三份不同文件卻沒有拒絕")
for n in range(3, 301):
    source = [f"文件{i}" for i in range(n)]
    candidate = split_documents(source + source[:2])
    groups = list(candidate.values())
    assert all(groups)
    assert sum(map(len, groups)) == n
    assert set.union(*map(set, groups)) == set(source)
    assert all(not set(groups[i]) & set(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert candidate == excerpt_function(source)

toy = toy_documents()
assert len(toy) == len(set(toy)) == 12
toy_parts = split_documents(toy)
assert [len(x) for x in toy_parts.values()] == [9, 1, 2]
experiment_tree = ast.parse((ROOT / "scripts/course_experiments/text.py").read_text())
experiment = next(node for node in experiment_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_simple_models")
experiment_calls = [ast.unparse(node) for node in ast.walk(experiment) if isinstance(node, ast.Call)]
assert "split_documents(toy_documents(), ctx.seed)" in experiment_calls
assert all(f"('mlp{context}', {context})" in ast.unparse(experiment) for context in (1, 3, 5))

text = "貓看狗。狗看貓。"
windows = [text[i:i+4] for i in range(len(text) - 3)]
assert windows[:2] == ["貓看狗。", "看狗。狗"]
assert windows[0][1:] == windows[1][:-1] == "看狗。"
assert set(windows[:1]).isdisjoint(windows[1:2])
near_duplicates = split_documents(["貓看狗。", "貓看狗。 ", "鳥看魚。"])
assert sum(map(len, near_duplicates.values())) == 3

hash_orders = {}
for hash_seed in ("1", "2", "3", "4"):
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-c", "import random; docs=list(set(['貓看狗。','狗看貓。','鳥看魚。','魚看鳥。'])); random.Random(42).shuffle(docs); print(docs)"]
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    hash_orders[hash_seed] = result.stdout.strip()
assert len(set(hash_orders.values())) > 1

results = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-2/verify_section.py",
    "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [frozen, ROOT / "tiny_perceptron/data.py", ROOT / "scripts/course_experiments/text.py", Path(__file__)]},
    "result": "全部斷言通過；未載入模型或執行訓練。",
    "repository_excerpt_ast_equal": True,
    "function_line_count": len(repository_function.splitlines()),
    "unique_documents": unique,
    "parts": parts,
    "items": list(parts.items()),
    "sorted_characters": sorted_characters,
    "shuffled": shuffled,
    "default_seed": excerpt_function.__defaults__,
    "redefinition_preserves_results": excerpt_function(docs) == parts,
    "repeated_document_preserves_results": split_documents(docs + ["貓看狗。"]) == parts,
    "novel_document_parts": novel_parts,
    "boundaries": [boundaries(n) for n in (3, 4, 5, 12, 100)],
    "rejected_inputs": errors,
    "partition_checks": {"smallest_n": 3, "largest_n": 300, "checked_sizes": 298, "all_nonempty": True, "all_pairwise_disjoint": True, "all_complete": True},
    "toy_documents": toy,
    "toy_parts": toy_parts,
    "experiment_split_call": "split_documents(toy_documents(), ctx.seed)",
    "experiment_contexts": [1, 3, 5],
    "overlapping_windows": windows[:2],
    "overlap": windows[0][1:],
    "exact_document_check_does_not_detect_overlap": True,
    "near_duplicate_documents_retained": near_duplicates,
    "set_shuffle_by_hash_seed": hash_orders,
    "language_semantics": {"dictionary_first_position_retained": True, "items_pair_unpacking": True, "set_deduplication_and_intersection": True, "empty_set_false": True, "shuffle_mutates_and_returns_none": True, "int_truncates_toward_zero": True, "tuple_assignment": True, "max_min_len": True, "default_and_docstring": True, "slice_partition": True},
}
print(json.dumps(results, ensure_ascii=False, indent=2))
