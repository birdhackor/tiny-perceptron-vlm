"""核對停止規則的人工算例、有限服務介面與原始來源收據。"""

import hashlib
import inspect
import itertools
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def sources():
    definitions = [
        (
            "act",
            "https://arxiv.org/html/1603.08983v6",
            "arXiv:1603.08983v6，2017-02-21；首次提交於 2016-03-29",
            "Adaptive Computation Time；Limiting Computation Time；Error Gradients",
            "We therefore take the more pragmatic approach of adding a time cost to the loss function",
            "停止分布由 sigmoid 單元導出；以加權狀態與輸出代替抽樣；目標加入 ponder cost，梯度忽略輪數跳變點。",
        ),
        (
            "calibration",
            "https://arxiv.org/html/1706.04599v2",
            "arXiv:1706.04599v2，2017-08-03",
            "Introduction；Definitions 的校準等式與 Negative log likelihood",
            "the probability associated with the predicted class label should reflect its ground truth correctness likelihood.",
            "校準描述群體機率與實際正確率的關係；NLL 需要真實標籤。高信心本身不是逐題正確保證。",
        ),
        (
            "validation",
            "https://scikit-learn.org/1.7/modules/cross_validation.html",
            "scikit-learn 1.7 版官方文件",
            "Cross-validation: evaluating estimator performance，validation set 說明",
            "final evaluation can be done on the test set",
            "調整超參數可能洩漏測試資訊；可用獨立驗證資料選設定，再作最後測試。",
        ),
        (
            "enumerate",
            "https://docs.python.org/3.13/library/functions.html",
            "Python 3.13 版官方文件；本機 Python 3.13.16",
            "Built-in Functions 的 enumerate",
            "a count (from start which defaults to 0)",
            "enumerate 逐次提供計數與元素，start 指定計數起點。",
        ),
    ]
    receipts = []
    for identifier, url, version, locator, quote, note in definitions:
        command = ["curl", "-fLsS", "--max-time", "40", url]
        result = subprocess.run(command, capture_output=True, check=True)
        parser = TextParser()
        parser.feed(result.stdout.decode("utf-8"))
        text = " ".join(" ".join(parser.parts).split())
        assert quote in text, "來源摘錄必須出現在當次下載的原文"
        receipts.append(
            {
                "id": identifier,
                "url": url,
                "version": version,
                "locator": locator,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "command_argv": command,
                "exit_code": result.returncode,
                "download_bytes": len(result.stdout),
                "download_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "checked_original": True,
                "short_excerpt": quote,
                "inspection_note": note,
                "retention": "全文只在記憶體處理；僅保存雜湊與支持主張的短摘錄。",
            }
        )
    return {"sources": receipts}


def execute():
    import torch

    from tiny_perceptron.lesson_products.inference import EFFORT, Product, auto_solve, rule_depth
    from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer

    torch.set_num_threads(1)
    torch.manual_seed(17)
    probabilities = torch.tensor([[0.95, 0.05], [0.97, 0.03]])
    indices = probabilities.argmax(-1).tolist()
    maxima = [round(float(x), 4) for x in probabilities.max(-1).values]
    enumeration = [(r, int(p.argmax())) for r, p in enumerate(probabilities, start=1)]
    assert indices == [0, 0] and maxima == [0.95, 0.97]
    assert enumeration == [(1, 0), (2, 0)]
    assert bool((probabilities.max(-1).values > 0.9).all())
    assert not any(i == 1 for i in indices)
    ce = {str(y): [-math.log(float(p[y])) for p in probabilities] for y in (0, 1)}
    best = {y: min(range(2), key=lambda r: values[r]) + 1 for y, values in ce.items()}
    assert best == {"0": 2, "1": 1}
    threshold_cases = []
    for threshold in (0.9, 0.99, 0.9999):
        p = ((1 + threshold) / 2, (1 - threshold) / 2)
        assert p[0] > threshold and p[0] > p[1] and math.isclose(sum(p), 1)
        threshold_cases.append({"threshold": threshold, "p": p, "predicted": 0, "gold": 1})

    tokenizer = Tokenizer()
    b = Product.__new__(Product)
    b.product, b.device, b.tokenizer = "b", torch.device("cpu"), tokenizer
    b.config = Config(vocab_size=len(tokenizer), width=16, heads=4, ff=32)
    b.model = LoopedSolver(b.config).eval()
    policy_path = "docs/lesson-products/results/b-auto-policy.json"
    b.auto_policy = json.loads((ROOT / policy_path).read_bytes())
    state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
    block_inputs, embedding_calls = [], []
    hooks = [
        b.model.embedding.register_forward_hook(lambda *args: embedding_calls.append(1)),
        b.model.shared.register_forward_pre_hook(lambda _, args: block_inputs.append(args[0].detach().clone())),
    ]
    low = b.solve(state, loops=1)
    high = b.solve(state, loops=5)
    assert len(block_inputs) == 6 and len(embedding_calls) == 2
    assert torch.equal(block_inputs[0], block_inputs[1])
    restart = {
        "low_loops": low["loops"],
        "high_loops": high["loops"],
        "shared_calls": len(block_inputs),
        "embedding_calls": len(embedding_calls),
        "first_block_inputs_identical": torch.equal(block_inputs[0], block_inputs[1]),
        "forward_parameters": list(inspect.signature(LoopedSolver.forward).parameters),
        "solve_parameters": list(inspect.signature(Product.solve).parameters),
        "scope": "隨機小模型只核對介面與計算路徑，不用其答案推論成品品質或速度。",
    }
    for hook in hooks:
        hook.remove()

    class Judge:
        def __init__(self, selected, info, events):
            self.selected, self.info, self.events = selected, info, events

        def synchronize(self):
            pass

        def judge(self, state, question):
            self.events.append("judge:" + question)
            return {"selected": self.selected if question == "choice" else self.info, "seconds": 0.0}

    solve = b.solve
    trace = []

    def traced_solve(state, loops):
        trace.append("solve:" + str(loops))
        return solve(state, loops=loops)

    b.solve = traced_solve
    cases = []
    for effort, choice, info, max_loops in [
        ("auto", "boxes", "yes", 5),
        ("low", "boxes", "yes", 5),
        ("medium", "boxes", "yes", 5),
        ("high", "boxes", "yes", 5),
        ("medium", "boxes", "yes", 2),
        ("auto", "none", "yes", 5),
        ("auto", "boxes", "no", 5),
    ]:
        trace.clear()
        result = auto_solve(b, Judge(choice, info, trace), state, effort=effort, max_loops=max_loops)
        assert trace[:2] == ["judge:choice", "judge:info"]
        expected = 0 if choice == "none" or info == "no" else min(3 if effort == "auto" else EFFORT[effort], max_loops)
        assert result["b_loops"] == expected
        assert len(trace) == (3 if expected else 2)
        cases.append({"effort": effort, "choice": choice, "info": info, "max_loops": max_loops,
                      "status": result["status"], "b_loops": result["b_loops"], "trace": list(trace)})

    c = Product.__new__(Product)
    c.product, c.device, c.tokenizer = "c", torch.device("cpu"), tokenizer
    c.config = Config(vocab_size=len(tokenizer), width=64, ff=128)
    c.model = CandidateScorer(c.config).eval()
    c.calibration = {"temperatures": {"choice": 1.0, "info": 1.0}}
    choices = c.judge(state, "choice")
    information = c.judge(state, "info")
    assert set(choices["probabilities"]) == {"card", "address", "app", "boxes", "sum", "none"}
    assert set(information["probabilities"]) == {"yes", "no"}
    assert sum(p.numel() for p in c.model.parameters()) == 84545

    logs = {
        "low": {"ok": [1, 1, 0, 0], "work": [1, 1, 1, 1]},
        "medium": {"ok": [1, 1, 1, 1], "work": [3, 3, 3, 3]},
        "high": {"ok": [1, 1, 1, 0], "work": [5, 5, 5, 5]},
    }
    all_policies = []
    for names in itertools.product(logs, repeat=4):
        all_policies.append((sum(logs[n]["ok"][i] for i, n in enumerate(names)),
                             sum(logs[n]["work"][i] for i, n in enumerate(names))))
    highest_correct = max(ok for ok, _ in all_policies)
    least_work = min(work for ok, work in all_policies if ok == highest_correct)
    assert (highest_correct, least_work) == (4, 8)
    record = json.loads((ROOT / "docs/lesson-products/results/initial-c-execution.json").read_bytes())
    training = record["training"]
    assert record["status"] == "completed" and training["random_initialization"] is True
    assert training["weights_changed"] is True and training["parameters"] == 84545
    files = [
        "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/inference.py",
        "tiny_perceptron/lesson_products/evaluation.py", "tiny_perceptron/lesson_products/train.py",
        policy_path, "docs/lesson-products/results/initial-c-execution.json",
    ]
    return {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torch_git_version": str(torch.version.git_version), "device": "cpu",
                        "platform": platform.platform(), "seed": "17", "threads": "1"},
        "sequence": {"argmax": indices, "maxima_rounded": maxima, "enumeration": enumeration,
                     "gold": 1, "errors": 2, "first_round_exceeds_0_9": True},
        "label_dependent_ce": {"ce_by_label": ce, "best_round_by_label": best},
        "threshold_counterexamples": threshold_cases,
        "restart": restart,
        "auto_cases": cases,
        "frozen_depths": {"effort": EFFORT, "boxes_by_events": {str(n): rule_depth(b.auto_policy, "boxes", n)
                          for n in range(1, 7)}, "sum": rule_depth(b.auto_policy, "sum")},
        "candidate_scorer": {"parameters": 84545, "choice_candidates": list(choices["probabilities"]),
                             "info_candidates": list(information["probabilities"]),
                             "scope": "隨機初值實跑只驗證候選介面；實際訓練來自另存的既有執行紀錄。"},
        "training_record": {"status": record["status"], "random_initialization": training["random_initialization"],
                            "weights_changed": training["weights_changed"], "steps": training["steps_this_attempt"],
                            "parameters": training["parameters"]},
        "oracle": {"enumerated_policies": len(all_policies), "highest_correct": highest_correct,
                   "least_total_work_at_highest_correct": least_work, "mean_work": least_work / 4,
                   "scope": "四題人工固定紀錄；上界只對所列預算及相同評分目標成立。"},
        "repository_sha256": {p: sha(p) for p in files},
        "result": "全部斷言通過。未載入成品權重、未更新模型、未測量訓練效益或服務速度。",
    }


if __name__ == "__main__":
    value = sources() if "--sources" in sys.argv else execute()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
