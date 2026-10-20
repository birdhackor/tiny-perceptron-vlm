"""核對凍結算例、問題替換，以及本專案候選評分器的實際資料流。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer, candidate_batch, grouped_scores


def capture(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-example", "exec"), namespace)
    return namespace, output.getvalue()


def main():
    torch.set_num_threads(1)
    source = ROOT / "docs/course-review-20261019/freeze-01/sources/9.11.md"
    raw = source.read_bytes()
    code = re.search(r"```python\n(.*?)```", raw.decode(), re.S)[1]
    original, stdout = capture(code)
    changed_code = code.replace('"question": "下一步服務？"', '"question": "是否已有訂單編號？"')
    assert changed_code != code, "練習必須實際替換問題"
    changed, changed_stdout = capture(changed_code)
    expected_stdout = "查卡片進度 0.844\n更改地址 0.042\n以上皆不適合 0.114\n程式依規則採取的動作： 開啟卡片進度頁\n"
    assert stdout == expected_stdout, "原始輸出與教材預期不同"
    assert changed_stdout == stdout, "只改問題的輸出應相同"
    assert changed["judgment"]["question"] != original["judgment"]["question"], "題目文字應不同"
    scores = original["scores"]
    p = original["probability"]
    exact = [math.exp(s) / sum(math.exp(x) for x in [2.0, -1.0, 0.0]) for s in [2.0, -1.0, 0.0]]
    assert max(abs(a - b) for a, b in zip(exact, p.tolist())) < 1e-7, "softmax 與獨立算式不符"
    rounded = [round(x, 3) for x in p.tolist()]
    assert rounded == [0.844, 0.042, 0.114], "三位小數不符"
    assert sum(rounded) == 1.0, "四捨五入後的和不符"
    assert isinstance(p.tolist(), list) and len(p.tolist()) == 3, "預期一維普通清單"
    assert type(int(p.argmax())) is int and int(p.argmax()) == 0, "最高機率位置不符"
    order = sorted(range(len(p)), key=lambda i: float(p[i]), reverse=True)
    assert order == [0, 2, 1], "候選機率排序不符"
    tree = ast.parse(code)
    question_lookups = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "question"]
    assert not question_lookups, "示意程式不應讀取問題欄位"

    torch.manual_seed(0)
    tokenizer = Tokenizer()
    model = CandidateScorer(Config(vocab_size=len(tokenizer), width=64, ff=128)).eval()
    state = "卡片進度;申請編號=?"
    candidates = ["card", "address", "app", "none"]
    rows = [{"state": state, "question": "choice", "candidates": candidates}]
    tokens, slices, maximum, _ = candidate_batch(rows, tokenizer, model.config)
    with torch.no_grad():
        scores_model = model(tokens)
        one_by_one = torch.stack([model(row[None])[0] for row in tokens])
        probabilities = grouped_scores(scores_model, slices, maximum).softmax(-1)
    torch.testing.assert_close(scores_model, one_by_one, rtol=1e-5, atol=1e-6)
    q1 = tokenizer.encode(state, "choice", "none")
    q2 = tokenizer.encode(state, "info", "no")
    from tiny_perceptron.lesson_products.data import QUESTIONS
    separated = tokenizer.ids
    question_ids = {
        name: [separated[c] for c in QUESTIONS[name]] for name in ("choice", "info")
    }
    assert question_ids["choice"] != question_ids["info"], "不同問題應有不同的輸入編碼"
    for question, sequence in (("choice", q1), ("info", q2)):
        start = sequence.index(2) + 1
        end = sequence.index(2, start)
        assert sequence[start:end] == question_ids[question], "輸入必須包含問題文字"
    loss = torch.nn.functional.cross_entropy(grouped_scores(model(tokens), slices, maximum), torch.tensor([0]))
    loss.backward()
    assert model.score.weight.grad is not None and bool(torch.isfinite(model.score.weight.grad).all()), "共享評分權重應收到有效梯度"
    bound_files = [
        "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/data.py",
        "tiny_perceptron/lesson_products/training.py",
        "tiny_perceptron/lesson_products/train.py",
    ]
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads())},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": stdout,
        "exercise_stdout": changed_stdout,
        "scores": {"values": scores.tolist(), "shape": list(scores.shape), "dtype": str(scores.dtype)},
        "probability": p.tolist(),
        "softmax_float64_independent": exact,
        "softmax_max_absolute_error": max(abs(a-b) for a,b in zip(exact,p.tolist())),
        "probability_sum_float32": float(p.sum()),
        "rounded": rounded,
        "rounded_sum": sum(rounded),
        "argmax_index": int(p.argmax()),
        "best": original["best"],
        "second": original["judgment"]["candidates"][order[1]],
        "pairs": list(zip(original["judgment"]["candidates"], p.tolist())),
        "question_subscript_reads": len(question_lookups),
        "candidate_model": {"token_shape": list(tokens.shape), "score_shape": list(scores_model.shape), "scores": scores_model.tolist(), "one_by_one_scores": one_by_one.tolist(), "probability": probabilities.tolist(), "question_token_sequences": question_ids, "same_parameters_for_all_candidates": True, "cross_entropy": float(loss.detach()), "score_weight_gradient_norm": float(model.score.weight.grad.norm()), "initialization": "新建模型；未讀取權重，未更新參數"},
        "repository_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in bound_files},
        "result": "全部斷言通過。",
        "scope": "驗證人工算例與 CPU 上的一次前向／反向資料流；不宣稱模型品質、校準效果、商業 Jev 內部結構或跨裝置數值完全一致。",
    }
    with (BASE / "verification.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
