"""C.4 reviewer evidence: deterministic CPU-only checks; never trains or samples a model."""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[5]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, lesson):
    raw = (ROOT / path).read_bytes()
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, header in enumerate(headers):
        if header[0].startswith(("## " + lesson + " ").encode()):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
            return raw[header.start() : end].decode()
    raise ValueError(lesson)


def fence(body):
    matches = re.findall(r"(?ms)^```python\n(.*?)^```", body)
    assert len(matches) == 1
    return matches[0]


def run(code):
    out = io.StringIO()
    namespace = {"__name__": "__main__"}
    with redirect_stdout(out):
        exec(compile(code, "canonical-fence", "exec"), namespace)
    return out.getvalue().splitlines(), namespace


body = section("course/chapters/0C.md", "C.4")
assert hashlib.sha256(body.encode()).hexdigest() == "43c1977aa09ea08c588563f6b11c6c71a59ff5c2591204abf5d9090ecff507b2"
code = fence(body)
base_out, ns = run(code)
assert base_out == ["集合包含正確答案 True", "多數決 3 答對 False", "驗證器 4 答對 True"]
exercise_out, _ = run(code.replace("candidates = [3, 3, 4]", "candidates = [3, 3, 5]"))
assert exercise_out == ["集合包含正確答案 False", "多數決 3 答對 False", "驗證器 None 答對 False"]
assert Counter([3, 3, 4]).most_common(1) == [(3, 2)]
assert Counter([10, 8]).most_common(1) == [(10, 1)]
assert Counter([8, 10]).most_common(1) == [(8, 1)]

unigram_out, _ = run(fence(section("course/chapters/01.md", "1.4")))
bpe_code = fence(section("course/chapters/06.md", "6.2"))
bpe_out, bpe_ns = run(bpe_code)
assert bpe_ns["pairs"] == Counter({("a", "b"): 3, ("b", "a"): 2, ("a", "c"): 1})
assert bpe_ns["pair"] == ("a", "b")
assert [bpe_ns["merge"](row, bpe_ns["pair"]) for row in bpe_ns["corpus"]] == [["ab", "ab"], ["ab", "a", "c"]]
bpe_exercise, bpe_extra_ns = run(bpe_code.replace('corpus = [list("abab"), list("abac")]', 'corpus = [list("abab"), list("abac"), list("acacacac")]'))
assert bpe_extra_ns["pair"] == ("a", "c")
assert [bpe_extra_ns["merge"](row, bpe_extra_ns["pair"]) for row in bpe_extra_ns["corpus"]] == [["a", "b", "a", "b"], ["a", "b", "ac"], ["ac", "ac", "ac", "ac"]]

result_path = "docs/course-experiments/results/reasoning.json"
record = json.loads((ROOT / result_path).read_bytes())
application_path = "scripts/course_experiments/applications.py"
current = (ROOT / application_path).read_bytes()
old = subprocess.run(["git", "show", record["revision"] + ":" + application_path], cwd=ROOT, capture_output=True, check=True).stdout
assert hashlib.sha256(old).hexdigest() == record["code_sha256"][application_path]


def functions(raw):
    return {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}


old_functions, current_functions = functions(old), functions(current)
compared = ["_rate", "_verify_reasoning", "_reasoning_samples", "_reasoning_records", "run_reasoning"]
for name in compared:
    assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(current_functions[name], include_attributes=False)

# Execute only the two pure verifier/rate functions from the actual current source.
namespace = {"re": re}
exec(compile(ast.Module(body=[current_functions[name] for name in ("_verify_reasoning", "_rate")], type_ignores=[]), application_path, "exec"), namespace)
verify, rate = namespace["_verify_reasoning"], namespace["_rate"]
budget_summary = []
total_candidates = 0
selected_examples = {}
for mode in ("direct", "steps"):
    for budget in record["results"]["comparison"][mode]["budgets"]:
        k = budget["candidate_count"]
        assert len(budget["samples"]) == 24
        all_candidates, rows, covered = [], [], []
        for row in budget["samples"]:
            assert len(row["candidates"]) == k
            assert row["truth"] == row["a"] + row["b"] + row["c"]
            candidates = []
            for candidate in row["candidates"]:
                # Verify the raw saved byte IDs rather than assuming generated text is accurate.
                ids = candidate["generated_ids"]
                raw_ids = ids[:-1] if ids and ids[-1] == 2 else ids
                assert candidate["generated"] == bytes(i - 8 for i in raw_ids if i >= 8).decode("utf-8", errors="replace")
                assert candidate["invalid_special_tokens"] == [i for i in raw_ids if i < 8]
                calculated = verify(candidate, row, mode)
                for key, value in calculated.items():
                    assert candidate[key] == value, (mode, k, row["question"], key)
                candidates.append(calculated)
                total_candidates += 1
            answers = [c["final_answer"] for c in candidates if c["final_answer"] is not None]
            majority = Counter(answers).most_common(1)[0][0] if answers else None
            verified = next((c["final_answer"] for c in candidates if c["fully_verified"]), None)
            oracle = any(c["final_correct"] for c in candidates)
            recomputed = {"oracle_coverage": oracle, "majority_answer": majority, "majority_correct": majority == row["truth"], "verified_answer": verified, "verifier_correct": verified == row["truth"], "selection_failed_despite_coverage": oracle and majority != row["truth"]}
            for key, value in recomputed.items():
                assert row[key] == value, (mode, k, row["question"], key)
            rows.append(recomputed)
            if oracle:
                covered.append(recomputed)
            all_candidates.extend(candidates)
            if (mode, k, row["question"]) in [("direct", 4, "(3+3)+0=?"), ("steps", 2, "(5+0)+3=?")]:
                selected_examples[f"{mode}/k={k}/{row['question']}"] = {"truth": row["truth"], "answers": [c["final_answer"] for c in candidates], "texts": [c["generated"] for c in row["candidates"]], **recomputed}
        calculated_rates = {
            "oracle_coverage": rate(len(covered), 24),
            "majority_accuracy": rate(sum(r["majority_correct"] for r in rows), 24),
            "verifier_accuracy": rate(sum(r["verifier_correct"] for r in rows), 24),
            "majority_accuracy_given_coverage": rate(sum(r["majority_correct"] for r in covered), len(covered)),
            "final_correct_but_steps_invalid": rate(sum(c["final_correct"] and not c["fully_verified"] for c in all_candidates), len(all_candidates)),
        }
        for key, value in calculated_rates.items():
            assert budget[key] == value, (mode, k, key)
        budget_summary.append({"mode": mode, "k": k, "candidates": len(all_candidates), **calculated_rates})
assert total_candidates == 720
assert selected_examples["direct/k=4/(3+3)+0=?"]["answers"] == [5, 7, 5, 6]
assert selected_examples["steps/k=2/(5+0)+3=?"]["answers"] == [10, 8]

row = {"a": 2, "b": 2, "c": 0, "truth": 4}
synthetic = {}
for name, text in {
    "all_four_pass": "2+2=4;4+0=4;answer=4",
    "correct_final_wrong_equations": "2+2=5;5+0=4;answer=4",
    "correct_equations_broken_link": "2+2=4;5+0=5;answer=5",
    "correct_equations_link_but_wrong_operands": "1+3=4;4+0=4;answer=4",
    "correct_suffix_unparsed_steps": "not-an-equation;answer=4",
}.items():
    synthetic[name] = verify({"generated": text, "invalid_special_tokens": []}, row, "steps")
assert synthetic["all_four_pass"]["fully_verified"]
assert all(not v["fully_verified"] for k, v in synthetic.items() if k != "all_four_pass")
assert synthetic["correct_final_wrong_equations"]["final_correct"]
assert synthetic["correct_suffix_unparsed_steps"]["final_correct"]

report = {
    "reviewer_task": "/root/tech_c4_tokenizer_20261027",
    "command": "uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/c4/review_check.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; stdlib-only checks; no torch import, model generation, training or data download"},
    "input_sha256": {p: sha(p) for p in ["course/chapters/0C.md", "course/chapters/01.md", "course/chapters/06.md", result_path, application_path, "tiny_perceptron/data.py"]},
    "canonical_fence_output": base_out,
    "exercise_output": exercise_out,
    "counter_tie_checks": [[10, 8, 10], [8, 10, 8]],
    "prerequisite_1_4_output": unigram_out,
    "prerequisite_6_2_two_document_output": bpe_out,
    "prerequisite_6_2_exercise_output": bpe_exercise,
    "historical_experiment_source": {"revision": record["revision"], "recorded_sha256": record["code_sha256"][application_path], "actual_sha256": hashlib.sha256(old).hexdigest(), "relevant_functions_AST_equal_to_current": compared},
    "existing_experiment_environment": {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "step_scale", "evidence_status", "status"]},
    "rechecked_candidates": total_candidates,
    "rechecked_question_rows": 192,
    "budget_summary": budget_summary,
    "selected_examples": selected_examples,
    "synthetic_verifier_cases": synthetic,
    "numeric_derivation": {"conditional_selection": "12 / 14 = 0.8571428571428571", "all_test_questions": "12 / 24 = 0.5", "verifier": "14 / 24 = 0.5833333333333334", "reason_denominators_differ": "14 covered questions versus all 24 distinct test questions; both selected-success numerators equal 12"},
    "result": "PASS: canonical code, exercise, Counter tie order, current prerequisites, 720 raw saved candidates, 192 selection rows, eight aggregate groups, and synthetic verifier boundaries all agree.",
    "not_verified": ["No rerun of original CUDA training or stochastic generation", "No broader model, seed, dataset or unknown-fact generalization", "No full website desktop/mobile layout or reader-comprehension review"],
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
