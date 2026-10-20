"""核對凍結算例、既存實驗紀錄及原始方法來源，不重新訓練。"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def read_remote(url):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    raw = subprocess.check_output(command)
    return raw.decode(), {
        "url": url,
        "command": command,
        "accessed_at": datetime.now(UTC).isoformat(),
        "raw_sha256": sha(raw),
        "bytes": len(raw),
        "full_text_saved": False,
    }


def record(path):
    raw = (ROOT / path).read_bytes()
    return json.loads(raw), {"path": path, "sha256": sha(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.section_run.resolve()
    allowed = ROOT / "outputs/reviewer-tools/runs"
    assert run.is_relative_to(allowed), "執行紀錄須來自指定工具的輸出目錄"
    copied = {}
    for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
        raw = (run / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "content": raw.decode()}
    extraction = json.loads(copied["extraction.json"]["content"])
    execution = json.loads(copied["execution.json"]["content"])
    frozen = (ROOT / "docs/course-review-20261019/freeze-04/sources/14.16.md").read_bytes()
    assert extraction["source_sha256"] == sha(frozen)
    assert execution["exit_code"] == 0
    assert extraction["figure_sha256"] == {}
    assert copied["stdout.txt"]["content"] == "早輪 0.2 後輪 0.4 懲罰 0.2\n早輪 0.5 後輪 0.4 懲罰 0.0\n"

    final, final_ref = record("docs/lesson-products/results/final-report-v1.json")
    empirical = {}
    for mode, filename in (("fixed3", "mechanism-fixed3-runner-result.json"), ("odd", "mechanism-odd135-runner-result.json")):
        runner, runner_ref = record("docs/lesson-products/results/" + filename)
        training = runner["execution"]["training"]
        mechanism = final["mechanisms"][mode]
        assert mechanism["actual_last_step"] == training["last_step"] == 1500
        assert training["seed"] == 8159
        values = {depth: result["boxes"] for depth, result in mechanism["evaluation"].items()}
        for value in values.values():
            assert value["count"] == 2956 and value["families"] == 1478
            assert abs(value["accuracy"] - value["correct"] / value["count"]) < 1e-7
        empirical[mode] = {
            "runner": runner_ref,
            "seed": training["seed"],
            "batch_size": training["batch_size"],
            "steps": training["last_step"],
            "training_environment": training["environment"],
            "initial_weights_sha256": training["initial_weights_sha256"],
            "terminal_supervision": mechanism["terminal_supervision"],
            "same_update_count_last_weights": mechanism["same_update_count_last_weights"],
            "boxes": values,
        }
    assert empirical["fixed3"]["initial_weights_sha256"] == empirical["odd"]["initial_weights_sha256"]
    fixed = empirical["fixed3"]["boxes"]
    odd = empirical["odd"]["boxes"]
    assert [f"{fixed[d]['cross_entropy']:.4f}" for d in ("3", "6")] == ["0.0238", "0.1072"]
    assert [f"{100 * fixed[d]['accuracy']:.2f}%" for d in ("3", "6")] == ["99.49%", "97.67%"]
    assert odd["3"]["correct"] == odd["4"]["correct"] == 2916
    assert odd["3"]["cross_entropy"] < odd["4"]["cross_entropy"]

    wrong_probabilities = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float64)
    wrong_ce = F.cross_entropy(wrong_probabilities.log(), torch.tensor([0, 0]), reduction="none")
    assert wrong_ce[1] < wrong_ce[0]
    assert wrong_probabilities.argmax(-1).tolist() == [1, 1]
    average_before, average_after = [0.2, 0.8], [0.4, 0.4]
    assert sum(average_after) / 2 < sum(average_before) / 2
    assert average_after[0] > average_before[0]
    coefficient = 0.5
    optimum = 1 - coefficient / 2
    objective = (optimum - 1) ** 2 + coefficient * max(0, optimum)
    assert optimum > 0 and objective < 1
    costs = [1.2, 0.8, 0.6, 0.7, 0.9]
    assert min(range(len(costs)), key=costs.__getitem__) + 1 == 3
    derivations = {
        "validation_minimum": {"costs": costs, "minimum_round": 3, "minimum": min(costs)},
        "penalty": {"before_after": [[0.2, 0.4], [0.5, 0.4]], "penalties": [0.2, 0.0], "unweighted_ce_sums": [0.6, 0.9]},
        "soft_constraint": {"objective": "(theta-1)^2 + 0.5*max(0,theta)", "intended_constraint": "theta <= 0", "global_minimizer": optimum, "objective_at_minimizer": objective, "best_feasible_objective": 1.0, "proof": "theta>0 時導數為 2*(theta-1)+0.5，二階導數為 2；theta<=0 的最佳可行點為 0。故有限懲罰仍可保留違反量。"},
        "average_vs_sample": {"losses_before": average_before, "losses_after": average_after, "means": [sum(average_before) / 2, sum(average_after) / 2]},
        "ce_vs_accuracy": {"probabilities": wrong_probabilities.tolist(), "gold_class": 0, "cross_entropy": wrong_ce.tolist(), "argmax": wrong_probabilities.argmax(-1).tolist(), "correct": [False, False], "formula": "對 one-hot 正確類別 y，CE=-ln(p_y)；p_y 提高但仍低於競爭類別時，CE 下降而答案仍錯。"},
        "fixed_point_vs_answer": {"update": "F(h)=0", "fixed_point": 0, "lipschitz_constant": 0, "readout": "對所有 h 輸出類別 1，正確類別為 0", "proof": "從任何初始狀態一次抵達固定點，但固定 readout 仍錯；狀態收斂與答對是不同命題。"},
        "unseen_inputs": "僅在訓練輸入上要求代價關係，沒有增加關聯不同輸入的假設時，可在新輸入任意指定反向的代價關係；訓練上的關係本身無法推出任意新題的逐題關係。",
        "selection_objective": "argmin 驗證 CE 選的是該批資料、該指標的最小值；若產品指標為正確率，應比較 argmax 正確率，兩者排序無須相同。",
    }

    paper_html, paper_receipt = read_remote("https://arxiv.org/html/2605.11011v1")
    text = Text()
    text.feed(paper_html)
    paper = " ".join(" ".join(text.parts).split())
    assert "guaranteeing a formal global contraction is intractable" in paper
    assert "SiLU" in paper and "Confidence Loss." in paper
    assert "Random Deep Supervision for Loop Training." in paper
    paper_receipt.update(
        version="arXiv:2605.11011v1，2026-05-10",
        locators=["Recasting LLM as a Looped LLM / Selective Gating for Stable Loop Dynamics", "Adaptive Computation via Early Stopping Mechanism", "Random Deep Supervision for Loop Training", "Training Objective / Refinement Losses / Confidence Loss", "Conclusion"],
        excerpt="Although this convex combination does not mathematically guarantee the entire composite operator is a strict contraction",
        inspection="已讀原文更新式、門控、隨機深度監督、總目標、單調性項及停止信心頭。單調性項使用 SiLU(後輪CE-前輪CE)，並和 CE、信心損失相加；停止時比較由目前狀態算出的信心值與門檻。論文以實測支持效益，未提供任意新題逐輪代價下降定理；門控原文明說不保證整個算子嚴格收縮。",
    )
    commit = torch.version.git_version
    loss_text, loss_receipt = read_remote(f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/loss.py")
    assert "class CrossEntropyLoss" in loss_text and "w_{y_n}" in loss_text
    loss_receipt.update(version=f"PyTorch {torch.__version__}; git {commit}", locator="CrossEntropyLoss：class-index target 的 unreduced loss 與 mean reduction", inspection="已讀輸入 logits、正確類別索引、負對數 softmax 及預設 mean 分母。未加類別權重且沒有忽略樣本時為各樣本 CE 算術平均。")
    argmax_text, argmax_receipt = read_remote(f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_torch_docs.py")
    assert "torch.argmax," in argmax_text
    argmax_receipt.update(version=f"PyTorch {torch.__version__}; git {commit}", locator="add_docstr(torch.argmax)：dim 參數的語義", inspection="已讀沿指定維度回傳最大值索引的定義，以及最大值並列時回傳第一個索引的規則。")

    code_refs = []
    for path in ("tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/evaluation.py"):
        raw = (ROOT / path).read_bytes()
        code_refs.append({"path": path, "sha256": sha(raw)})
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-16/verification.py --section-run " + args.section_run.as_posix(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": commit, "device": "cpu"},
        "result": "全部斷言通過；小節執行、數值推導、實驗紀錄及原始來源核對完成。",
        "source_sha256": sha(frozen),
        "section_execution_copies": copied,
        "final_report": final_ref,
        "empirical": empirical,
        "derivations": derivations,
        "external_sources": [paper_receipt, loss_receipt, argmax_receipt],
        "repository_code": code_refs,
        "limitations": "未重新訓練，未讀取既有 outputs 實驗；實測主張核對保存的最終成績、訓練收據與計分程式，未從逐題 logits 重算 CE，也不把單一 seed 當成一般品質保證。",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
