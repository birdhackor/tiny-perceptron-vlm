"""Bounded CPU review: exact fence variants and existing source labels only."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
raw = (BASE / "original-fence/fence-1.py").read_bytes()

def run(code):
    output = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "original-9.1-fence", "exec"), namespace)
    return output.getvalue(), namespace

original_stdout, original = run(raw)
assert original_stdout == "True True True\n較安全回答可直接當安全正例 False\n"
assert original["pair"][f"回答{original['pair']['相對較安全']}安全"] is False
changed_raw = raw.replace('"回答0安全": False'.encode(), '"回答0安全": True'.encode())
assert changed_raw != raw
(BASE / "exercise-variant.py").write_bytes(changed_raw)
exercise_stdout, exercise = run(changed_raw)
assert exercise_stdout == "True True True\n較安全回答可直接當安全正例 True\n"
assert exercise["record"] == original["record"]
assert "有用" not in exercise["pair"] and "誠實" not in exercise["pair"]
independent_record = {"有用": False, "誠實": False, "安全": False}
independent_record["安全"] = True
assert independent_record == {"有用": False, "誠實": False, "安全": True}
lookup_cases = []
for safe0, safe1, relative in [(False, False, 0), (False, False, 1), (True, False, 0), (True, False, 1)]:
    pair = {"回答0安全": safe0, "回答1安全": safe1, "相對較安全": relative}
    key = f"回答{pair['相對較安全']}安全"
    actual = pair[key]
    assert actual is (safe0 if relative == 0 else safe1)
    lookup_cases.append({"safety0": safe0, "safety1": safe1, "relative": relative, "lookup_key": key, "result": actual})
tree = ast.parse(raw)
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
assert {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} == {"print"}

archive = ROOT / "assets/training/pku-safe-rlhf-v1.tar.gz"
with tarfile.open(archive, "r:gz") as handle:
    rows_raw = handle.extractfile("behavior-initial/pku-safe-rlhf/train-first-100.jsonl").read()
rows = [json.loads(line) for line in rows_raw.splitlines()]
assert hashlib.sha256(rows_raw).hexdigest() == "c4a88d08ef7456669766f1a3b908f82d918be5625a24794d9843d7f6df1d552b"
labels = {
    "rows": len(rows),
    "both_safe": sum(row["is_response_0_safe"] and row["is_response_1_safe"] for row in rows),
    "both_unsafe": sum(not row["is_response_0_safe"] and not row["is_response_1_safe"] for row in rows),
    "mixed": sum(row["is_response_0_safe"] != row["is_response_1_safe"] for row in rows),
    "safer_is_unsafe": sum(not row[f"is_response_{row['safer_response_id']}_safe"] for row in rows),
}
assert labels == {"rows": 100, "both_safe": 44, "both_unsafe": 40, "mixed": 16, "safer_is_unsafe": 40}
assert all(row["safer_response_id"] in (0, 1) and row["better_response_id"] in (0, 1) for row in rows)
result = {
    "environment": {"python": platform.python_version(), "executable": sys.executable, "device": "cpu; no tensor operation", "cwd": str(Path.cwd())},
    "fence_sha256": hashlib.sha256(raw).hexdigest(),
    "original_stdout": original_stdout,
    "exercise_stdout": exercise_stdout,
    "exercise_variant_sha256": hashlib.sha256(changed_raw).hexdigest(),
    "independent_record": independent_record,
    "lookup_cases": lookup_cases,
    "fence_imports": 0,
    "fence_named_calls": ["print"],
    "existing_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "existing_raw_rows_sha256": hashlib.sha256(rows_raw).hexdigest(),
    "existing_label_counts": labels,
    "scope": "Exact original fence and bounded variants, plus existing source-data schema/labels. The 100-row counts are auxiliary label checks, not model performance or a representative dataset estimate. No training, inference, network, download or weights in this execution.",
}
(BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
