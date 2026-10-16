"""9.16 獨立技術核對：成本算例、Auto控制流程、有限世界及隨機初始化。

只跑CPU小型確定性fixture；不載入成品權重、不更新參數、不重現歷史訓練。
"""

import contextlib
import io
import json
import platform
from decimal import Decimal

import torch

from tiny_perceptron.lesson_products.data import answer, family, render, split_of
from tiny_perceptron.lesson_products.inference import auto_solve
from tiny_perceptron.lesson_products.model import CandidateScorer, Config


torch.set_num_threads(1)
output = {
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    },
    "scope": "無模型訓練；Auto使用stub判斷與stub解題器，只驗證實際auto_solve的控制流程。",
}

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for p in [0.6, 0.8, 0.9]:
        answer_cost = 5 * (1 - p)
        action = "回答" if answer_cost < 1 - 1e-9 else "澄清"
        print(p, round(answer_cost, 3), action)
assert buffer.getvalue() == "0.6 2.0 澄清\n0.8 1.0 澄清\n0.9 0.5 回答\n"
output["lesson_code_stdout"] = buffer.getvalue()
output["float_boundary"] = {
    "p_0.8_cost_repr": repr(5 * (1 - 0.8)),
    "answer_at_p_0.8000000001": 5 * (1 - 0.8000000001) < 1 - 1e-9,
    "answer_at_p_0.800000001": 5 * (1 - 0.800000001) < 1 - 1e-9,
    "note": "理論門檻p>0.8；程式依正文容差把距平手1e-9內的成本視作平手。",
}
assert output["float_boundary"]["answer_at_p_0.8000000001"] is False
assert output["float_boundary"]["answer_at_p_0.800000001"] is True
thresholds = {str(c): str(Decimal(1) - c / Decimal(5)) for c in (Decimal("1"), Decimal("0.5"))}
assert thresholds == {"1": "0.8", "0.5": "0.9"}
output["exact_decimal_thresholds"] = thresholds


class StubB:
    auto_policy = {"version": 1, "sum_depth": 3, "low_max_events": 1, "medium_max_events": 3}

    def synchronize(self):
        pass

    def solve(self, state, loops=3):
        events.append(["b.solve", loops])
        return {"fixture": True, "loops": loops}


class StubC:
    def __init__(self, choice, info):
        self.selected = {"choice": choice, "info": info}

    def synchronize(self):
        pass

    def judge(self, state, question):
        events.append(["c.judge", question])
        return {"selected": self.selected[question], "seconds": 0.0}


cases = []
for name, choice, info, expected, expected_loops in [
    ("unsupported", "none", "yes", "unsupported", 0),
    ("missing_info", "sum", "no", "clarification_needed", 0),
    ("complete_supported", "sum", "yes", "answered", 3),
]:
    events = []
    result = auto_solve(StubB(), StubC(choice, info), "三數加法;x=1;y=2;z=3")
    assert result["status"] == expected
    assert result["b_loops"] == expected_loops
    assert events[:2] == [["c.judge", "choice"], ["c.judge", "info"]]
    assert events[2:] == ([["b.solve", expected_loops]] if expected_loops else [])
    cases.append({"name": name, "status": result["status"], "b_loops": result["b_loops"], "events": events})
output["auto_control_fixtures"] = cases

world = {"task": "boxes", "a": 2, "b": 5, "events": [("swap", 0), ("A", 1)], "query": "B"}
assert answer(world) == 2
fid = family("boxes", world)
output["finite_world_fixture"] = {
    "input": render(world),
    "gold": answer(world),
    "family": fid,
    "split": split_of(fid, "review-only-fixture"),
    "note": "審閱者自訂小例，非歷史訓練資料或成績；答案由明示覆寫/交換規則決定。",
}

config = Config(vocab_size=8, width=8, heads=2, ff=16, max_length=8, c_layers=1)
torch.manual_seed(123)
model1 = CandidateScorer(config)
torch.manual_seed(123)
model2 = CandidateScorer(config)
torch.manual_seed(124)
model3 = CandidateScorer(config)
same_seed = all(torch.equal(a, b) for a, b in zip(model1.parameters(), model2.parameters(), strict=True))
changed_seed = any(not torch.equal(a, b) for a, b in zip(model1.parameters(), model3.parameters(), strict=True))
assert same_seed and changed_seed
output["random_initialization_fixture"] = {
    "parameters": sum(p.numel() for p in model1.parameters()),
    "same_seed_identical": same_seed,
    "different_seed_changes_parameters": changed_seed,
    "no_checkpoints_loaded": True,
    "optimizer_steps": 0,
}
output["all_assertions_passed"] = True
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
