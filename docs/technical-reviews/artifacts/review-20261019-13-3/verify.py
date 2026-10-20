"""核對序列分數、遮罩與既有實驗的抽樣計數，不執行訓練。"""

import ast
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import (
    _pair_examples,
    _preference_evaluate,
    _preference_parts,
)
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


torch.set_num_threads(1)
torch.set_default_device("cpu")
environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "device": "cpu",
    "platform": platform.platform(),
}
logits = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
labels = torch.tensor([[IGNORE, 0, 2]])
valid = labels != IGNORE
safe = labels.masked_fill(~valid, 0)
log_probs = logits.log_softmax(-1)
gathered = log_probs.gather(-1, safe.unsqueeze(-1))
selected = gathered.squeeze(-1)
score = sequence_log_probability(logits, labels)
target_probability = math.exp(2) / (math.exp(2) + 2)
target_logp = 2 - math.log(math.exp(2) + 2)
ensure(IGNORE == -100, "忽略標籤必須符合常數")
ensure(valid.tolist() == [[False, True, True]], "有效位置不符")
ensure(safe.tolist() == [[0, 0, 2]], "替換後標籤不符")
ensure(abs(score.item() - 2 * target_logp) < 1e-6, "加總與獨立公式不符")
ensure(round(score.item(), 4) == -0.4791, "加總顯示值不符")
ensure(round(score.exp().item(), 4) == 0.6193, "機率顯示值不符")
ensure(abs(math.log(0.8 * 0.8) - 2 * math.log(0.8)) < 1e-15, "log 恆等式不符")

alternate = sequence_log_probability(logits, torch.tensor([[1, 0, 2]]))
extended_logits = torch.cat([logits, logits[:, 2:3]], dim=1)
extended_labels = torch.tensor([[IGNORE, 0, 2, 2]])
extended = sequence_log_probability(extended_logits, extended_labels)
ensure(round(alternate.item(), 4) == -0.7186, "練習顯示值不符")
ensure(abs(alternate.item() - extended.item()) < 1e-6, "新增有效項與練習不符")
ensure(abs(score.item() / 2 - extended.item() / 3) < 1e-7, "逐步平均不符")

try:
    log_probs.gather(-1, labels.unsqueeze(-1))
except RuntimeError as error:
    unsafe_gather = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("非法候選編號應被拒絕")

empty_row_errors = []
for invalid_labels in (
    torch.tensor([[IGNORE, IGNORE, IGNORE]]),
    torch.tensor([[IGNORE, 0, 2], [IGNORE, IGNORE, IGNORE]]),
):
    batch_logits = logits.repeat(invalid_labels.shape[0], 1, 1)
    try:
        sequence_log_probability(batch_logits, invalid_labels)
    except ValueError as error:
        empty_row_errors.append({"rows": invalid_labels.shape[0], "message": str(error)})
    else:
        raise AssertionError("無有效目標的筆數應被拒絕")

two_rows = sequence_log_probability(
    logits.repeat(2, 1, 1), torch.tensor([[IGNORE, 0, 2], [1, 0, 2]])
)
ensure(two_rows.shape == (2,), "每筆分數形狀不符")
ensure(torch.allclose(two_rows, torch.cat([score, alternate])), "逐筆分數不符")
gradient_logits = logits.clone().requires_grad_(True)
sequence_log_probability(gradient_logits, labels).sum().backward()
ensure(gradient_logits.grad[0, 0].eq(0).all().item(), "忽略位置仍有梯度")

source = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.3.md").read_text()
fences = source.split("```python\n")[1:]
excerpt_tree = ast.parse(fences[1].split("```", 1)[0])
repo_tree = ast.parse((ROOT / "tiny_perceptron/alignment.py").read_text())
excerpt_fn = next(n for n in excerpt_tree.body if isinstance(n, ast.FunctionDef))
repo_fn = next(n for n in repo_tree.body if isinstance(n, ast.FunctionDef) and n.name == excerpt_fn.name)
excerpt_identical = ast.dump(excerpt_fn, include_attributes=False) == ast.dump(repo_fn, include_attributes=False)
ensure(excerpt_identical, "教材節錄與實作不符")

tok = ByteTokenizer()
examples = []
answer_target_records = []
for answer in ("4", "10"):
    x, y = render_chat([{"role": "user", "content": "問題"}, {"role": "assistant", "content": answer}])
    answer_ids = tok.encode(answer) + [tok.eos_id]
    ensure(y[y != IGNORE].tolist() == answer_ids, "回答標籤未包含完整內容與 EOS")
    answer_target_records.append({"answer": answer, "targets": answer_ids, "count": len(answer_ids)})
    examples.append((x, y))
padded_x, padded_y, attention_valid = pad_batch(examples)
ensure((padded_y[~attention_valid] == IGNORE).all().item(), "PAD 仍被計分")


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_length=128)

    def forward(self, x, valid=None):
        return {"logits": torch.zeros((*x.shape, tok.vocab_size), device=x.device)}


uniform = UniformModel()
probe_records = [{"prompt": "2+2=?", "chosen": "4", "rejected": "10"}]
evaluation = _preference_evaluate(uniform, uniform, probe_records)
sample = evaluation["samples"][0]
ensure(abs(sample["policy_chosen_logp"] + 2 * math.log(tok.vocab_size)) < 2e-6, "評估未使用整段加總")
ensure(abs(sample["policy_rejected_logp"] + 3 * math.log(tok.vocab_size)) < 2e-6, "評估疑似長度正規化")
ensure(sample["relative_margin"] == 0, "相同參考與策略的相對分數應為零")
policy_chosen, policy_rejected = torch.tensor([-2.0]), torch.tensor([-5.0])
reference_chosen, reference_rejected = torch.tensor([-3.0]), torch.tensor([-4.0])
actual_loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1)
expected_loss = math.log1p(math.exp(-0.1 * ((-2 + 3) - (-5 + 4))))
ensure(abs(actual_loss.item() - expected_loss) < 1e-7, "參考相減的 DPO 公式不符")

experiment_path = ROOT / "docs/course-experiments/results/dpo.json"
experiment = json.loads(experiment_path.read_bytes())
parts = _preference_parts(split_records(arithmetic_records(), seed=experiment["seed"]))
split_receipts = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(raw).hexdigest()
    recorded = experiment["results"]["data"][name]
    ensure(digest == recorded["sha256"], "重建資料與實驗切分不符")
    split_receipts[name] = {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": digest}

pair_examples = _pair_examples(parts["train"], max_length=128)
replayed_runs = {}
for name in ("model", "beta1"):
    run = experiment["results"]["runs"][name]
    training = run["training"]
    sampler = random.Random(experiment["seed"])
    counts = [0, 0]
    selected_ids = []
    for _ in range(training["steps"]):
        indices = sampler.choices(range(len(pair_examples)), k=8)
        selected_ids.extend(indices)
        for side in (0, 1):
            _, targets, _ = pad_batch([pair_examples[i][side] for i in indices])
            observed = int((targets != IGNORE).sum())
            field = ("chosen", "rejected")[side]
            independent = sum(len(tok.encode(parts["train"][i][field])) + 1 for i in indices)
            ensure(observed == independent, "抽樣計數與文字加 EOS 的獨立計算不符")
            counts[side] += observed
    ensure(training["steps"] == 250, "記錄的更新數不符")
    ensure(sum(counts) == training["effective_answer_tokens_both_sides"] == 9448, "有效目標曝光量不符")
    replayed_runs[name] = {
        "beta": run["beta"],
        "steps": training["steps"],
        "pairs_per_step": 8,
        "pair_draws": len(selected_ids),
        "unique_training_pairs": len(set(selected_ids)),
        "chosen_answer_targets": counts[0],
        "rejected_answer_targets": counts[1],
        "effective_answer_tokens_both_sides": sum(counts),
        "checkpoint": training["checkpoint"],
    }

notes = {}


def inspect_notes(node, path):
    if isinstance(node, dict):
        if "chosen_higher_absolute_probability" in node:
            ensure("sequence sums include EOS" in node["note"], "候選比較記錄缺少 EOS 約定")
            for row in node["samples"]:
                for field in ("chosen", "rejected"):
                    ensure(row[field + "_answer_tokens"] == len(tok.encode(row[field])) + 1, "候選有效長度未包含 EOS")
            notes[path] = {"records": node["records"], "note": node["note"]}
        for key, value in node.items():
            inspect_notes(value, path + "." + key)


inspect_notes(experiment["results"], "results")
paths = (
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/data.py",
    "scripts/course_experiments/behavior.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py",
    "docs/course-experiments/results/dpo.json",
)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-3/verify.py",
    "environment": environment,
    "result": "全部核對成立；只執行算例、固定 logits 的評估及資料抽樣重放，沒有執行訓練更新。",
    "source_sha256": sha(ROOT / "docs/course-review-20261019/freeze-01/sources/13.3.md"),
    "repository_file_sha256": {p: sha(ROOT / p) for p in paths},
    "numeric": {
        "probability_product": 0.8 * 0.8,
        "log_product": math.log(0.8 * 0.8),
        "sum_of_logs": 2 * math.log(0.8),
        "softmax_exact_formula": target_probability,
        "log_probability_exact_formula": target_logp,
        "softmax": logits.softmax(-1).tolist(),
        "log_softmax": log_probs.tolist(),
        "valid": valid.tolist(),
        "safe_labels": safe.tolist(),
        "selected": selected.tolist(),
        "score": score.tolist(),
        "restored_probability": score.exp().tolist(),
        "extended_score": extended.tolist(),
        "alternate_labels_score": alternate.tolist(),
        "two_target_mean": (score / 2).tolist(),
        "three_target_mean": (extended / 3).tolist(),
        "rounded_selected": [round(value, 4) for value in selected[0].tolist()],
    },
    "shapes": {"logits": list(logits.shape), "labels": list(labels.shape), "gather_index": list(safe.unsqueeze(-1).shape), "gathered": list(gathered.shape), "selected": list(selected.shape), "output": list(score.shape)},
    "guards": {"unsafe_gather": unsafe_gather, "empty_row_errors": empty_row_errors},
    "two_rows": two_rows.tolist(),
    "ignored_position_gradient": gradient_logits.grad[0, 0].tolist(),
    "excerpt_ast_identical": excerpt_identical,
    "answer_targets": answer_target_records,
    "padded_labels": padded_y.tolist(),
    "attention_valid": attention_valid.tolist(),
    "preference_evaluation_uniform_model": evaluation,
    "dpo_loss_formula_probe": {"actual": actual_loss.item(), "expected": expected_loss},
    "experiment_record": {
        "sha256": sha(experiment_path),
        "revision": experiment["revision"],
        "recorded_environment": {k: experiment[k] for k in ("device", "seed", "torch_version", "python_version", "gpu", "step_scale", "status")},
        "reconstructed_splits": split_receipts,
        "replayed_runs": replayed_runs,
        "candidate_comparison_notes": notes,
    },
    "limitations": "沒有重跑既有 GPU 訓練或檢查遠端模型權重；重放只核對此固定 seed、切分、抽樣流程的有效目標曝光量，並以保存的訓練紀錄核對步數。有限 logits、合法整數標籤及事先對齊的下一 token 目標是算例條件。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
