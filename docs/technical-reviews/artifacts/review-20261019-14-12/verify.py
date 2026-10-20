"""核對手工遞迴、共享區塊的資料流與原始論文收據。"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bits = []

    def handle_data(self, data):
        self.bits.append(data)


def source_receipt(url, version, locators, excerpts):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    raw = subprocess.run(command, capture_output=True, check=True, timeout=45).stdout
    parser = PlainText()
    parser.feed(raw.decode("utf-8"))
    text = " ".join(" ".join(parser.bits).split())
    assert all(locator in text for locator in locators), "原文缺少指定定位"
    assert all(excerpt in text for excerpt in excerpts), "原文摘錄不符"
    return {
        "url": url,
        "version": version,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "fetched_bytes": len(raw),
        "fetched_sha256": hashlib.sha256(raw).hexdigest(),
        "locators": locators,
        "short_excerpts": excerpts,
        "full_text_saved": False,
    }


def trajectory(coefficient, rounds):
    e, state = 4.0, 0.0
    result = []
    for _ in range(rounds):
        state = state + coefficient * (e - state)
        result.append(state)
    return result


torch.set_num_threads(1)
torch.manual_seed(20261020)
tokenizer = Tokenizer()
config = Config(vocab_size=len(tokenizer), width=16, heads=4, ff=32)
model = LoopedSolver(config).eval()
prompt = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
tokens = padded([tokenizer.encode(prompt)], config.max_length)
saved_tokens = tokens.clone()
parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
calls = []


def observe(_module, args, output):
    hidden, valid = args
    calls.append((hidden.detach().clone(), valid.detach().clone(), output.detach().clone()))


handle = model.shared.register_forward_hook(observe)
with torch.inference_mode():
    original = model.embedding(tokens) + model.position(torch.arange(tokens.shape[1]))
    logits = model(tokens, loops=4, all_depths=True)
handle.remove()
assert len(calls) == 4, "共享區塊執行次數不符"
assert tuple(logits.shape) == (1, 4, 19), "逐輪答案分數維度不符"
injection_errors = []
for index, (hidden, valid, output) in enumerate(calls):
    previous = original if index == 0 else calls[index - 1][2]
    injection_errors.append(float((hidden - (previous + config.input_injection * original)).abs().max()))
    assert torch.equal(valid, tokens != 0), "有效位置遮罩不符"
    assert tuple(output.shape) == (1, tokens.shape[1], config.width), "隱藏狀態位置數改變"
assert injection_errors == [0.0] * 4, "原輸入再次注入的資料流不符"
with torch.inference_mode():
    last_logits = model(tokens, loops=4)
assert torch.equal(last_logits, logits[:, -1]), "最後一輪讀出不符"

product = Product.__new__(Product)
product.product = "b"
product.device = torch.device("cpu")
product.config, product.model, product.tokenizer = config, model, tokenizer
answer = product.solve(prompt, loops=4)
assert set(answer) == {"answer", "loops", "seconds", "stress_length"}, "解題器輸出契約不符"
assert answer["loops"] == 4 and 0 <= answer["answer"] < 19, "答案類別或輪數不符"
assert torch.equal(tokens, saved_tokens), "輸入 token 被改寫"
assert all(torch.equal(parameters[name], value) for name, value in model.named_parameters()), "推論改變參數"

half = trajectory(0.5, 4)
twice = trajectory(2.0, 8)
assert half == [2.0, 3.0, 3.5, 3.75], "半差距算例不符"
assert twice == [8.0, 0.0] * 4, "振盪算例不符"
assert [4.0 - value for value in half] == [2.0, 1.0, 0.5, 0.25], "誤差減半不符"

sources = [
    source_receipt(
        "https://arxiv.org/html/2502.05171v2",
        "arXiv:2502.05171v2，2025-02-17",
        ["3.1 Macroscopic Design", "3.2 Microscopic Design", "Motivation for this Design"],
        ["a core block that takes the current step as input"],
    ),
    source_receipt(
        "https://arxiv.org/html/1706.03762v7",
        "arXiv:1706.03762v7",
        ["3 Model Architecture", "3.4 Embeddings and Softmax"],
        [
            "consuming the previously generated symbols as additional input when generating the next.",
            "convert the input tokens and output tokens to vectors",
        ],
    ),
]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "seed": 20261020,
    "numeric": {"coefficient_half": half, "errors_half": [4.0 - v for v in half], "coefficient_two": twice},
    "software": {
        "prompt": prompt,
        "configuration": vars(config),
        "weights": "隨機初值；只核對資料流及推論介面，不評分答題能力",
        "token_shape": list(tokens.shape),
        "hidden_shapes": [list(call[2].shape) for call in calls],
        "logits_shape": list(logits.shape),
        "shared_calls": len(calls),
        "injection_max_absolute_errors": injection_errors,
        "last_readout_equal": True,
        "input_unchanged": True,
        "parameters_unchanged": True,
        "solve_result": answer,
    },
    "repository_inputs": [
        {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
        for path in ["tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/inference.py"]
    ],
    "source_receipts": sources,
    "result": "全部斷言通過",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
