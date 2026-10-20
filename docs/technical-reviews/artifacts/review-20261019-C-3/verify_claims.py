"""重算候選紀錄、透明算例與確定性生成的邊界。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record_path = ROOT / "docs/course-experiments/results/reasoning.json"
record = json.loads(record_path.read_bytes())
comparison = record["results"]["comparison"]
summary = []
questions = None
checked_candidates = 0
repeated_example = None
for mode in ("direct", "steps"):
    total = 0
    for budget in comparison[mode]["budgets"]:
        k = budget["candidate_count"]
        rows = budget["samples"]
        order = [row["question"] for row in rows]
        if questions is None:
            questions = order
        assert order == questions
        assert len(order) == len(set(order)) == 24
        hit_count = 0
        for row in rows:
            truth = row["a"] + row["b"] + row["c"]
            assert truth == row["truth"]
            generation = row["generation"]
            assert generation["temperature"] == 0.7
            assert generation["candidate_count"] == k
            assert generation["messages"] == [{"role": "user", "content": row["question"]}]
            assert len(row["candidates"]) == k
            successes = []
            for candidate in row["candidates"]:
                ids = candidate["generated_ids"]
                payload = ids[:-1] if ids and ids[-1] == 2 else ids
                decoded = bytes(value - 8 for value in payload if value >= 8).decode("utf-8", errors="replace")
                assert decoded == candidate["generated"]
                clean = not any(value < 8 for value in payload)
                text = decoded.strip()
                pattern = r"-?[0-9]+" if mode == "direct" else r".*;answer=(-?[0-9]+)"
                matched = re.fullmatch(pattern, text) if clean else None
                parsed = int(matched[0] if mode == "direct" else matched[1]) if matched else None
                correct = parsed == truth
                assert correct == candidate["final_correct"]
                assert parsed == candidate["final_answer"]
                successes.append(correct)
                checked_candidates += 1
            hit = any(successes)
            assert hit == row["oracle_coverage"]
            hit_count += hit
            if mode == "direct" and k == 8 and row["question"] == "(3+3)+0=?":
                answers = [item["final_answer"] for item in row["candidates"]]
                repeated_example = {"question": row["question"], "truth": truth, "answers": answers, "sevens": answers.count(7)}
        stored = budget["oracle_coverage"]
        assert stored["numerator"] == hit_count
        assert stored["denominator"] == len(rows)
        assert stored["rate"] == hit_count / len(rows)
        total += len(rows) * k
        summary.append({"mode": mode, "k": k, "hits": hit_count, "questions": len(rows), "candidates": len(rows) * k})
    assert total == 360
assert repeated_example["sevens"] == 6

source_path = ROOT / "scripts/course_experiments/applications.py"
source = source_path.read_text()
nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
namespace = {"torch": torch, "ByteTokenizer": ByteTokenizer, "copy": copy, "time": time}
for name in ("_sync", "_prompt_ids", "_sample"):
    exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), str(source_path), "exec"), namespace)


class FixedAnswer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=128)

    def forward(self, current, cache=None):
        logits = torch.full((current.shape[0], 1, 264), -100.0)
        logits[:, :, 2 if cache is not None else ord("7") + 8] = 100.0
        return {"logits": logits, "cache": True}


greedy = []
for k in (1, 2, 4, 8):
    generated = namespace["_sample"](FixedAnswer(), [{"role": "user", "content": "(3+3)+0=?"}], SimpleNamespace(device="cpu"), count=k, tokens=8, temperature=0)
    answers = [item["generated"] for item in generated["samples"]]
    assert answers == ["7"] * k
    greedy.append({"k": k, "answers": answers, "coverage": any(answer == "6" for answer in answers)})

freeze = ROOT / "docs/course-review-20261019/freeze-01/sources/C.3.md"
body = freeze.read_text()
fence = re.search(r"```python\n(.*?)```", body, re.S)[1]
exercise = io.StringIO()
with contextlib.redirect_stdout(exercise):
    exec(fence.replace("candidates = [3, 4, 3, 5]", "candidates = [3, 3, 3, 3]"), {})
assert exercise.getvalue().splitlines()[-1].endswith("False")

calculations = {
    "failure_once": str(Fraction(7, 10)),
    "failure_twice": str(Fraction(7, 10) ** 2),
    "success_twice": str(1 - Fraction(7, 10) ** 2),
    "coverage_by_k": {str(k): float(1 - Fraction(7, 10) ** k) for k in (1, 2, 4, 8)},
    "six_or_more_sevens_if_probability_is_point_eight": float(sum(Fraction(math.comb(8, n)) * Fraction(4, 5) ** n * Fraction(1, 5) ** (8 - n) for n in range(6, 9))),
    "plug_in_mean": float(1 - (1 - Fraction(1, 2)) ** 2),
    "mean_of_per_question_coverage": float(((1 - Fraction(1, 10) ** 2) + (1 - Fraction(9, 10) ** 2)) / 2),
    "candidates_per_branch": 24 * sum((1, 2, 4, 8)),
    "n_equals_k_estimator": {str(k): [1 - Fraction(math.comb(k - c, k), math.comb(k, k)) for c in range(k + 1)] for k in (1, 2, 4, 8)},
    "degenerate_success_probabilities": {str(p): {str(k): 1 - (1 - p) ** k for k in (1, 2, 4, 8)} for p in (0, 1)},
}
original_probe = subprocess.run(["git", "show", record["revision"] + ":scripts/course_experiments/applications.py"], cwd=ROOT, capture_output=True)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-3/verify_claims.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "inputs": {str(path.relative_to(ROOT)): digest(path) for path in (record_path, source_path, freeze, ROOT / "tiny_perceptron/data.py")},
    "recorded_run": {key: record[key] for key in ("revision", "seed", "device", "gpu", "torch_version", "python_version", "evidence_status")},
    "recorded_code_sha256": record["code_sha256"]["scripts/course_experiments/applications.py"],
    "original_code_probe": {"exit_code": original_probe.returncode, "stderr": original_probe.stderr.decode()},
    "calculations": calculations,
    "record_recount": summary,
    "checked_candidates": checked_candidates,
    "same_questions_in_all_budgets_and_branches": True,
    "repeated_example": repeated_example,
    "greedy_fixed_model": greedy,
    "exercise_stdout": exercise.getvalue(),
    "result": "所有紀錄、ID 解碼與算例斷言通過；確定性生成不增加集合命中。獨立性須另外按聯合機率的定義判斷。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", end="")
