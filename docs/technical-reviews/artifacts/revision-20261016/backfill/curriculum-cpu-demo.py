"""One fixed public-grammar request; this is not a heldout benchmark."""

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch

from tiny_perceptron.lesson_products.inference import Product, auto_solve, rule_depth

torch.set_num_threads(2)
b = Product(ROOT / "outputs/lesson-products/products/b", "b")
c = Product(ROOT / "outputs/lesson-products/products/c", "c")
state = "兩盒追蹤;A=1;B=2;事件=A設3;查詢=A"
result = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "2"},
    "scope": "One fixed complete public-grammar input. Reviewer did not train, calibrate, tune, or restart the formal test.",
    "state": state,
    "hand_expected_answer": 3,
    "frozen_rule": b.auto_policy,
    "rule_depth_by_event_count": {str(n): rule_depth(b.auto_policy, "boxes", n) for n in range(1, 7)},
    "observed": auto_solve(b, c, state),
}
Path(__file__).with_name("curriculum-cpu-demo-output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
