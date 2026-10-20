"""核對成本推導、浮點邊界與有限世界服務的決策順序，不訓練模型。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products.data import answer, arithmetic_worlds, render
from tiny_perceptron.lesson_products.inference import auto_solve
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer


class Judge:
    def __init__(self, selected, complete):
        self.selected = selected
        self.complete = complete
        self.calls = []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.selected if question == "choice" else self.complete, "seconds": 0.0}


class Solver:
    auto_policy = {"version": 1, "low_max_events": 0, "medium_max_events": 4, "sum_depth": 1}

    def __init__(self):
        self.calls = []

    def synchronize(self):
        pass

    def solve(self, state, loops):
        self.calls.append(loops)
        return {"answer": 2, "loops": loops}


def main():
    snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/9.16.md"
    code = re.search(r"```python\n(.*?)```", snapshot.read_text(), re.S)[1]
    baseline, exercise = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(baseline):
        exec(code, {})
    with contextlib.redirect_stdout(exercise):
        exec(code.replace("1 - 1e-9", "0.5 - 1e-9"), {})
    assert baseline.getvalue() == "0.6 2.0 澄清\n0.8 1.0 澄清\n0.9 0.5 直接開頁\n"
    assert exercise.getvalue() == "0.6 2.0 澄清\n0.8 1.0 澄清\n0.9 0.5 澄清\n"
    exact = []
    for p in (Fraction(3, 5), Fraction(4, 5), Fraction(9, 10)):
        cost = 5 * (1 - p)
        exact.append({"p": str(p), "cost": str(cost), "float_cost": repr(5 * (1 - float(p)))})
    assert repr(5 * (1 - 0.8)) == "0.9999999999999998"
    assert 5 * (1 - 0.8) < 1
    assert not 5 * (1 - 0.8) < 1 - 1e-9
    assert 1 - Fraction(1, 5) == Fraction(4, 5)
    assert 1 - Fraction(1, 2) / 5 == Fraction(9, 10)
    probabilities = torch.tensor([2.0, -1.0, 0.0]).softmax(-1).tolist()
    assert round(probabilities[0], 3) == 0.844
    cases = []
    for task, complete, state, effort, cap in [
        ("card", "yes", "卡片進度;申請編號=123456", "auto", 5),
        ("boxes", "no", "兩盒追蹤;A=?;B=6;事件=交換;查詢=B", "auto", 5),
        ("boxes", "yes", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "auto", 5),
        ("sum", "yes", "三數加法;x=1;y=2;z=3", "auto", 5),
        ("boxes", "yes", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "medium", 2),
    ]:
        judge, solver = Judge(task, complete), Solver()
        result = auto_solve(solver, judge, state, effort=effort, max_loops=cap)
        assert judge.calls == ["choice", "info"]
        cases.append({"task": task, "complete": complete, "status": result["status"],
                      "solver_calls": solver.calls, "judge_calls": judge.calls})
    assert [c["status"] for c in cases] == [
        "unsupported", "clarification_needed", "answered", "answered", "budget_limited"]
    assert [c["solver_calls"] for c in cases] == [[], [], [3], [1], [2]]
    config = Config(len(Tokenizer()), width=64, ff=128)
    torch.manual_seed(7421)
    first = CandidateScorer(config)
    torch.manual_seed(7421)
    same = CandidateScorer(config)
    torch.manual_seed(7422)
    other = CandidateScorer(config)
    same_seed_equal = all(torch.equal(a, b) for a, b in zip(first.parameters(), same.parameters()))
    other_seed_differs = any(not torch.equal(a, b) for a, b in zip(first.parameters(), other.parameters()))
    assert same_seed_equal and other_seed_differs
    world = next(arithmetic_worlds())
    assert answer(world) == world["x"] + world["y"] + world["z"]
    groups = [(Fraction(3, 5), Fraction(2, 5)), (Fraction(9, 10), Fraction(9, 10))]
    predicted_policy_cost = sum(min(5 * (1 - p), 1) for p, truth in groups) / len(groups)
    actual_policy_cost = sum(1 if p <= Fraction(4, 5) else 5 * (1 - truth)
                             for p, truth in groups) / len(groups)
    assert any(p != truth for p, truth in groups)
    assert predicted_policy_cost == actual_policy_cost == Fraction(3, 4)
    concentration = (3 * Fraction(3, 5) - 1) / 2
    assert concentration == Fraction(2, 5)
    assert 5 * (1 - concentration) == 5 * (1 - Fraction(2, 5)) == 3
    paths = ["tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/model.py",
             "tiny_perceptron/lesson_products/train.py", "tiny_perceptron/lesson_products/data.py"]
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-16/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "platform": platform.platform()},
        "result": "全部斷言通過；Auto 的替身只核對路由，不代表模型準確率。",
        "baseline_stdout": baseline.getvalue(), "exercise_stdout": exercise.getvalue(),
        "exact_costs": exact, "thresholds": {"clarification_1": "4/5", "clarification_half": "9/10"},
        "float_rule": {"plain_less_than_one": 5 * (1 - 0.8) < 1,
                       "with_tolerance": 5 * (1 - 0.8) < 1 - 1e-9,
                       "effective_threshold": repr(0.8 + 1e-9 / 5)},
        "previous_example": {"probabilities": probabilities, "rounded_first": round(probabilities[0], 3)},
        "auto_cases": cases, "initialization": {"same_seed_equal": same_seed_equal,
                                                  "other_seed_differs": other_seed_differs},
        "finite_world": {"rendered": render(world), "answer": answer(world)},
        "calibration_counterexamples": {
            "description": "這是指定母體的精確反例，不是測得的模型品質。兩群等重，模型在第一群未校準，第二群校準。",
            "groups": [{"reported_probability": str(p), "true_success_rate": str(t)} for p, t in groups],
            "predicted_policy_cost": str(predicted_policy_cost), "actual_policy_cost": str(actual_policy_cost),
            "concentration_example": {"candidate_probabilities": ["3/5", "1/5", "1/5"],
                                       "concentration": str(concentration), "true_success_rate": "2/5",
                                       "using_concentration_cost": "3", "actual_direct_cost": "3"},
            "scope": "未校準或使用集中度都不能保證成本估計可靠；這不推出估計必然錯誤。"},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
