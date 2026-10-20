"""直接執行公開judge介面，以固定logits隔離選項決策及校準溫度。"""

import json
import platform
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.inference import Product, auto_solve
from tiny_perceptron.lesson_products.model import Config, Tokenizer


class FixedScores(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, tokens):
        assert tokens.shape[0] == len(self.values)
        return torch.tensor(self.values)


product = Product.__new__(Product)
product.product, product.device = "c", torch.device("cpu")
product.tokenizer = Tokenizer()
product.config = Config(vocab_size=len(product.tokenizer))
product.calibration = {"temperatures": {"choice": 4.0, "info": 4.0}}
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
cases = []
for question, scores, expected in [
    ("choice", [-0.3, 0.2, 0.5, 1.4, 1.1, -0.1], "boxes"),
    ("choice", [0.0] * 6, "address"),
    ("info", [-0.1, 0.1], "no"),
]:
    product.model = FixedScores(scores)
    result = product.judge(state, question)
    assert result["selected"] == expected
    if expected == "boxes":
        assert max(result["probabilities"].values()) < 0.5
    cases.append({"scores": scores, "result": result})


class NeverCalled:
    def synchronize(self):
        raise AssertionError("無效effort或上限必須在任何成品呼叫前被拒絕")


invalid = []
for effort, cap in [("unknown", 5), ("auto", 0), ("medium", 7), ("auto", True)]:
    try:
        auto_solve(NeverCalled(), NeverCalled(), state, effort, cap)
    except ValueError as error:
        invalid.append({"effort": effort, "max_loops": cap, "error": str(error)})
    else:
        raise AssertionError("無效輸入未被拒絕")

result = {
    "status": "passed",
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m docs.technical-reviews.artifacts.review-20261019-19-17.judge_probe_corrected",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "cases": cases,
    "invalid_inputs": invalid,
    "scope": "固定logits只驗證公開選擇規則及輸入驗證，不代表已訓練模型的品質。",
}
destination = Path(__file__).with_name("judge-execution-corrected.json")
with destination.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(result, ensure_ascii=False))
