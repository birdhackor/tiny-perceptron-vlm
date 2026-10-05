"""Independent, bounded C.2 arithmetic and existing raw-candidate audit.

No model is instantiated, no checkpoint is loaded, and no generation/training
or download is performed. Only named original methods are AST-extracted.
"""
import ast
import hashlib
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return {n.name: [n.lineno, n.end_lineno] for n in selected}


def local_checks(steps, truth=4):
    return {
        "equations": [a + b == claimed for a, b, claimed in steps],
        "linked": [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))],
        "final": steps[-1][2] == truth,
    }


original = local_checks([(2, 2, 5), (5, -1, 4)])
changed_claim = local_checks([(2, 2, 4), (5, -1, 4)])
wrong_task = local_checks([(2, 3, 5), (5, -1, 4)])
redundant = local_checks([(2, 2, 4), (4, 1, 5), (5, -1, 4)])
assert original == {"equations": [False, True], "linked": [True], "final": True}
assert changed_claim == {"equations": [True, True], "linked": [False], "final": True}
assert wrong_task == {"equations": [True, True], "linked": [True], "final": True}
assert redundant == {"equations": [True, True, True], "linked": [True, True], "final": True}

method_ns = {"re": re}
application_path = ART / "inputs/applications-experiment-revision.py"
locations = original_functions(application_path, ["_verify_reasoning", "_reasoning_records"], method_ns)
common_ns = {"random": random, "json": json, "hashlib": hashlib}
common_path = ART / "inputs/common.py"
locations.update(original_functions(common_path, ["split_records", "records_sha256"], common_ns))
verify = method_ns["_verify_reasoning"]

report = json.loads((ART / "inputs/reasoning-original.json").read_bytes())
assert sha(application_path) == report["code_sha256"]["scripts/course_experiments/applications.py"]
assert sha(common_path) == report["code_sha256"]["scripts/course_experiments/common.py"]
assert sha(ART / "inputs/data.py") == report["code_sha256"]["tiny_perceptron/data.py"]

records = method_ns["_reasoning_records"]()
splits = common_ns["split_records"](records, report["seed"])
families = {name: set(r["family"] for r in rows) for name, rows in splits.items()}
for name, rows in splits.items():
    observed = {"records": len(rows), "families": len(families[name]), "sha256": common_ns["records_sha256"](rows)}
    assert observed == report["results"]["split"][name]
assert not (families["train"] & families["test"])
assert not (families["validation"] & families["test"])

budgets = report["results"]["comparison"]["steps"]["budgets"]
denominators = []
checked = 0
for bi, budget in enumerate(budgets):
    k = budget["candidate_count"]
    assert len(budget["samples"]) == 24
    for si, row in enumerate(budget["samples"]):
        assert {key: row[key] for key in ["family", "a", "b", "c", "question", "truth"]} == splits["test"][si]
        assert len(row["candidates"]) == k
        for ci, candidate in enumerate(row["candidates"]):
            checked += 1
            recalculated = verify(candidate, row, "steps")
            assert all(candidate[key] == value for key, value in recalculated.items())
            ids = candidate["generated_ids"]
            raw_ids = ids[:-1] if ids and ids[-1] == 2 else ids
            decoded = bytes(i - 8 for i in raw_ids if i >= 8).decode("utf-8", errors="replace")
            assert decoded == candidate["generated"]
            assert len(ids) == candidate["generated_tokens"]
            assert candidate["eos"] == bool(ids and ids[-1] == 2)
            assert candidate["invalid_special_tokens"] == [i for i in raw_ids if i < 8]
    denominators.append({"budget_pointer": f"/results/comparison/steps/budgets/{bi}", "candidate_count": k,
                         "distinct_tasks": 24, "candidate_instances": 24 * k})

target_pointer = "/results/comparison/steps/budgets/1/samples/7/candidates/0"
row = budgets[1]["samples"][7]
target = row["candidates"][0]
assert row["question"] == "(5+0)+3=?" and row["truth"] == 8
assert target["generated"] == "5+0=5;5+3=10;answer=10"
assert target["parsed"] and target["linked"] and target["task_operands_valid"]
assert not target["equations_valid"] and not target["final_correct"] and not target["fully_verified"]

# Exercise separate contract branches without generating new model scores.
branch_row = {"a": 2, "b": 2, "c": 0, "truth": 4}
def synthetic(text, invalid=None):
    return verify({"generated": text, "invalid_special_tokens": invalid or []}, branch_row, "steps")
branch_cases = {
    "correct": synthetic("2+2=4;4+0=4;answer=4"),
    "correct_final_bad_equation": synthetic("2+2=5;5+0=4;answer=4"),
    "valid_equations_unlinked": synthetic("2+2=4;3+0=3;answer=3"),
    "wrong_operands": synthetic("2+1=3;3+1=4;answer=4"),
    "final_parse_only": synthetic("unstructured text;answer=4"),
    "invalid_special_token": synthetic("2+2=4;4+0=4;answer=4", [3]),
}
assert branch_cases["correct"]["fully_verified"]
assert branch_cases["correct_final_bad_equation"]["final_correct"] and not branch_cases["correct_final_bad_equation"]["fully_verified"]
assert branch_cases["valid_equations_unlinked"]["equations_valid"] and not branch_cases["valid_equations_unlinked"]["linked"]
assert branch_cases["wrong_operands"]["equations_valid"] and branch_cases["wrong_operands"]["linked"] and not branch_cases["wrong_operands"]["task_operands_valid"]
assert branch_cases["final_parse_only"]["final_correct"] and not branch_cases["final_parse_only"]["parsed"]
assert not branch_cases["invalid_special_token"]["parsed"]

result = {
    "scope": "Recalculate saved candidate labels and arithmetic; no training or model generation.",
    "local_examples": {"original": original, "only_first_claim_changed": changed_claim, "wrong_task": wrong_task,
                       "redundant_but_arithmetically_valid": redundant},
    "task_scope_note": "Local valid and linked flags alone do not check the problem or a rule for accepting extra steps. Redundant correct algebra is not inherently a false proof.",
    "original_method_locations": locations,
    "source_provenance": {key: report[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "unfinished_schedules"]},
    "split_reconstruction": report["results"]["split"],
    "denominators": denominators,
    "candidate_labels_and_token_records_checked": checked,
    "target_pointer": target_pointer,
    "target": {"question": row["question"], "truth": row["truth"], "generated": target["generated"],
               "result": verify(target, row, "steps"), "generated_tokens": target["generated_tokens"]},
    "synthetic_contract_branches": branch_cases,
    "software_environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                             "device": "CPU only; no model", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")},
    "hashes": {str(p.relative_to(ART)): sha(p) for p in [Path(__file__), application_path, common_path, ART / "inputs/data.py", ART / "inputs/reasoning-original.json"]},
}
(ART / "audit-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
