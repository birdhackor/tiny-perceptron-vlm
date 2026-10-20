"""重算純量梯度、逐題評估統計與既有實驗的設定；不執行訓練。"""

import ast
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.course_experiments.behavior import _pair_examples, _preference_parts  # noqa: E402
from scripts.course_experiments.common import split_records  # noqa: E402
from scripts.course_experiments.text import arithmetic_records  # noqa: E402
from tiny_perceptron.alignment import dpo_loss  # noqa: E402
from tiny_perceptron.data import ByteTokenizer  # noqa: E402
from tiny_perceptron.model import ModelConfig, TinyLM  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def fetch(url):
    command = ["curl", "-L", "--fail", "--max-time", "30", "-sS", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    return completed.stdout, {"command_argv": command, "exit_code": completed.returncode}


class PaperParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.formulas = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "math" and attrs.get("id") in {"S3.E3.m1", "S4.E4.m1", "S4.E7.m1"}:
            self.formulas[attrs["id"]] = attrs["alttext"]


torch.set_num_threads(1)
torch.manual_seed(42)
environment = {
    "python": sys.version,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "platform": platform.platform(),
    "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
}
inputs = [
    "course/chapters/13.md",
    "docs/course-review-20261019/freeze-01/sources/13.6.md",
    "docs/course-review-20261019/freeze-04.json",
    "docs/course-experiments/results/dpo.json",
    "docs/course-experiments/results/style.json",
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/training.py",
    "scripts/course_experiments/behavior.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py",
]
input_sha256 = {p: digest((ROOT / p).read_bytes()) for p in inputs}
dpo = read_json("docs/course-experiments/results/dpo.json")
style = read_json("docs/course-experiments/results/style.json")
scalar = []
for chosen_value in (-2.0, -6.0):
    margin = chosen_value - (-4.0) - ((-3.0) - (-3.0))
    for beta in (0.1, 1.0, 5.0):
        chosen = torch.tensor([chosen_value], requires_grad=True)
        loss = dpo_loss(chosen, torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-3.0]), beta)
        loss.backward()
        room = 1 - torch.sigmoid(torch.tensor(beta * margin))
        exact_room = 1 / (1 + math.exp(beta * margin))
        exact_loss = math.log1p(math.exp(-beta * margin))
        exact_grad = -beta * exact_room
        assert abs(loss.item() - exact_loss) < 1e-6, "純量代價不符合公式"
        assert abs(chosen.grad.item() - exact_grad) < 1e-6, "純量梯度不符合公式"
        scalar.append({
            "chosen": chosen_value,
            "margin": margin,
            "beta": beta,
            "loss_float32": loss.item(),
            "room_float32": room.item(),
            "gradient_float32": chosen.grad.item(),
            "loss_float64": exact_loss,
            "room_float64": exact_room,
            "gradient_float64": exact_grad,
            "printed": [round(loss.item(), 6), round(room.item(), 6), round(chosen.grad.item(), 6)],
            "plain_descent_delta_lr_0.1": -0.1 * chosen.grad.item(),
        })
assert [r["printed"] for r in scalar[:3]] == [
    [0.598139, 0.450166, -0.045017],
    [0.126928, 0.119203, -0.119203],
    [0.000045, 0.000045, -0.000227],
], "正文表格與實跑不一致"
assert 4.5e-05 == 4.5 * 10**-5 == 0.000045, "科學記號換算不一致"
chosen = torch.tensor([-2.0], requires_grad=True)
for _ in range(2):
    dpo_loss(chosen, torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-3.0])).backward()
accumulated_grad = chosen.grad.item()
assert abs(accumulated_grad - 2 * scalar[0]["gradient_float32"]) < 1e-7, "梯度累積不符合預期"
rejected_beta = {}
for beta in (0, -1):
    try:
        dpo_loss(*(torch.zeros(1) for _ in range(4)), beta=beta)
    except ValueError as error:
        rejected_beta[str(beta)] = str(error)
    else:
        raise AssertionError("非正 beta 未被拒絕")

arithmetic = split_records(arithmetic_records(), seed=42)
pairs = _preference_parts(arithmetic)
split_audit = {}
for name, rows in pairs.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    observed = {"records": len(rows), "families": len({row["family"] for row in rows}), "sha256": digest(serialized)}
    assert all(dpo["results"]["data"][name][k] == v for k, v in observed.items()), "偏好資料重建與紀錄不一致"
    split_audit[name] = observed

metrics = {}
expected = {
    "before.validation": [0, 0, 0.0],
    "before.test": [1, 0, 0.0],
    "model.validation": [3, 5, 2.39470],
    "model.test": [1, 1, -16.27347],
    "beta1.validation": [0, 5, 2.20619],
    "beta1.test": [0, 0, -5.03428],
}
for name in ("before", "model", "beta1"):
    for split in ("validation", "test"):
        report = (dpo["results"]["before"][split] if name == "before"
                  else dpo["results"]["runs"][name]["preference"][split])
        samples = report["samples"]
        assert len(samples) == report["records"] == len(pairs[split]), "評估分母不一致"
        for sample in samples:
            assert sample["policy_margin"] == sample["policy_chosen_logp"] - sample["policy_rejected_logp"], "策略差距錯誤"
            assert sample["relative_margin"] == sample["policy_margin"] - sample["reference_margin"], "相對差距錯誤"
            original = next(r for r in pairs[split] if r["prompt"] == sample["prompt"])
            assert all(sample[k] == original[k] for k in original), "逐題紀錄與重建資料不一致"
        observed = {
            "records": len(samples),
            "chosen_higher_absolute_probability": sum(r["policy_margin"] > 0 for r in samples),
            "relative_preference_improved": sum(r["relative_margin"] > 0 for r in samples),
            "mean_relative_margin": sum(r["relative_margin"] for r in samples) / len(samples),
        }
        assert all(observed[k] == report[k] for k in observed), "逐題重算不符合紀錄"
        values = [observed["chosen_higher_absolute_probability"], observed["relative_preference_improved"], round(observed["mean_relative_margin"], 5)]
        assert values == expected[f"{name}.{split}"], "逐題重算不符合正文表格"
        observed["samples"] = samples
        metrics[f"{name}.{split}"] = observed

generation = {}
tok = ByteTokenizer()
for name in ("before", "model", "beta1"):
    evaluation = (style["results"]["content_evaluation"] if name == "before"
                  else dpo["results"]["runs"][name]["arithmetic"])
    for split in ("validation", "test"):
        report = evaluation[split]
        samples = report["samples"]
        observed = {"records": len(samples), "matches": sum(r["exact"] for r in samples), "eos_rate": sum(r["eos"] for r in samples) / len(samples)}
        assert all(report[k] == v for k, v in observed.items()), "生成彙總與逐題紀錄不一致"
        assert observed == {"records": 8 if split == "validation" else 7, "matches": 0, "eos_rate": 1.0}, "生成結果不符合正文"
        for sample in samples:
            ids = sample["generated_ids"]
            ended = tok.eos_id in ids
            raw = ids[:ids.index(tok.eos_id)] if ended else ids
            assert sample["eos"] == ended, "EOS 欄位與 token 不一致"
            assert sample["generated"] == tok.decode(raw), "生成文字與 token 不一致"
            assert sample["exact"] == (raw == tok.encode(sample["expected"])), "正確率欄位與 token 不一致"
            prompt = sample["messages"][0]["content"]
            matching = next(r for r in pairs[split] if r["prompt"] == prompt)
            assert matching["chosen"] == sample["expected"], "生成與偏好候選的正解不一致"
        if name != "before" and split == "test":
            sample = next(r for r in samples if r["messages"][0]["content"] == "2+2=?")
            assert sample["generated"] == "8", "範例生成答案不符合正文"
        observed["samples"] = samples
        generation[f"{name}.{split}"] = observed

config = ModelConfig(width=64, layers=2)
model = TinyLM(config)
parameter_count = sum(p.numel() for p in model.parameters())
assert parameter_count == style["results"]["content_training"]["parameters"] == 141568, "參數數量不符合紀錄"
examples = _pair_examples(pairs["train"], config.max_length)
sampler = random.Random(42)
effective = sum(int((y != -100).sum()) for _ in range(250) for pair in sampler.choices(examples, k=8) for _, y in pair)
for name, beta in (("model", 0.1), ("beta1", 1.0)):
    run = dpo["results"]["runs"][name]
    assert run["beta"] == beta and run["training"]["steps"] == 250, "訓練設定不符合正文"
    assert run["training"]["effective_answer_tokens_both_sides"] == effective, "抽樣與有效 token 數不符合紀錄"
assert dpo["seed"] == style["seed"] == 42, "種子不同"
assert dpo["results"]["reference_unchanged"], "參考模型未保持固定"
for split in ("validation", "test"):
    nll_sum = -sum(r["policy_chosen_logp"] for r in metrics[f"before.{split}"]["samples"])
    assert abs(nll_sum - style["results"]["content_evaluation"][split]["nll_sum"]) < 2e-5, "兩份紀錄的底座分數不一致"

original_sources = []
paper_url = "https://arxiv.org/html/2305.18290v3"
raw, receipt = fetch(paper_url)
parser = PaperParser()
parser.feed(raw.decode())
assert len(parser.formulas) == 3, "原論文定位未找到"
original_sources.append({"url": paper_url, "version": "arXiv:2305.18290v3", "accessed_on": "2026-10-20", "retrieved_bytes_sha256": digest(raw), "receipt": receipt, "selected_formulas": parser.formulas, "inspection_note": "已閱讀 Preliminaries 與 Direct Preference Optimization 的原文；核對 KL 係數、獎勵到最優策略的變數代換與 DPO 代價。只保存對應公式。"})
commit = str(torch.version.git_version)
for filename, locator, excerpt in (
    ("torch/_tensor.py", "Tensor.backward", "This function accumulates gradients in the leaves"),
    ("torch/optim/sgd.py", "SGD.__doc__", r"\theta_t \leftarrow \theta_{t-1} - \gamma g_t"),
):
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{filename}"
    raw, receipt = fetch(url)
    text = raw.decode()
    assert excerpt in text, "官方原始碼未找到指定引文"
    original_sources.append({"url": url, "version": f"PyTorch {torch.__version__}; git {commit}", "accessed_on": "2026-10-20", "retrieved_bytes_sha256": digest(raw), "receipt": receipt, "locator": locator, "excerpt": excerpt})

historical_code = {}
path = "scripts/course_experiments/behavior.py"
current = (ROOT / path).read_text()
current_nodes = {n.name: n for n in ast.parse(current).body if isinstance(n, ast.FunctionDef)}
for result, names in (
    (dpo, ["_dpo_train", "_preference_evaluate", "run_dpo", "_pair_examples", "_preference_parts"]),
    (style, ["run_style"]),
):
    revision = result["revision"]
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{revision}/{path}"
    raw, receipt = fetch(url)
    assert digest(raw) == result["code_sha256"][path], "歷史程式不符合實驗所記雜湊"
    text = raw.decode()
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    matching = {name: ast.dump(nodes[name], include_attributes=False) == ast.dump(current_nodes[name], include_attributes=False) for name in names}
    assert all(matching.values()), "相關函式與實驗版本不同"
    historical_code[revision] = {"url": url, "receipt": receipt, "sha256": digest(raw), "matching_current_functions": matching, "inspection_note": "已核對指定函式全文及其相依資料、評估與最佳化程式；只記錄原檔雜湊與函式比對結果，不保存全文。"}

artifact = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-6/verify_facts.py",
    "environment": environment,
    "checked_at": datetime.now(UTC).isoformat(),
    "input_sha256": input_sha256,
    "scalar": scalar,
    "repeated_leaf_accumulated_gradient": accumulated_grad,
    "invalid_beta_errors": rejected_beta,
    "scientific_notation": {"4.5e-05": 4.5e-05},
    "split_audit": split_audit,
    "preference_metrics": metrics,
    "generation": generation,
    "model_config": asdict(config),
    "parameter_count": parameter_count,
    "training_settings": {"betas": [0.1, 1.0], "steps_each": 250, "pairs_per_step": 8, "learning_rate": 0.001, "seed": 42, "effective_answer_tokens_both_sides_each": effective, "optimizer": "AdamW", "reference_unchanged_recorded": True},
    "historical_code": historical_code,
    "recorded_experiment_environment": {name: {k: r[k] for k in ("device", "gpu", "python_version", "torch_version", "seed", "revision")} for name, r in (("dpo", dpo), ("style", style))},
    "scope": "只重跑純量與統計核對，不重訓 CUDA 模型、不讀取權重；實驗品質只支持兩份既有逐題紀錄的單一 seed 與切分。局部更新按梯度縮放限於純量普通梯度下降，並不聲稱 AdamW 更新量與原始梯度成比例。",
    "result": "所有明列斷言通過。",
}
(HERE / "facts-execution.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
(HERE / "original-source-excerpts.json").write_text(json.dumps(original_sources, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": artifact["result"], "scalar_cases": len(scalar), "preference_groups": len(metrics), "generation_groups": len(generation), "parameter_count": parameter_count, "effective_tokens_each": effective, "input_sha256": input_sha256}, ensure_ascii=False))
