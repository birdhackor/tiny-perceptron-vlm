"""核對混合代價、梯度、有效位置及固定實驗配方；不執行模型訓練。"""

import ast
import hashlib
import inspect
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

import torch

from scripts.course_experiments import compression
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip and value.strip():
            self.parts.append(value.strip())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "-L", "--fail", "--max-time", "35", "-sS", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


torch.set_num_threads(1)
torch.manual_seed(7)
student = torch.zeros(1, 2, 2, requires_grad=True)
teacher = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).log()
labels = torch.tensor([[0, 1]])
ce = masked_loss(student, labels)
kl = distillation_kl(student, teacher, labels, temperature=1)
expected_ce = math.log(2)
expected_kl = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
assert abs(ce.item() - expected_ce) < 1e-7, "CE 與自然對數推導不符"
assert abs(kl.item() - expected_kl) < 1e-7, "KL 與逐候選求和不符"
mixtures = {}
gradients = {}
for alpha in (0.0, 0.25, 0.5, 1.0):
    value = distillation_loss(student, teacher, labels, alpha=alpha, temperature=1)
    expected = (1 - alpha) * expected_ce + alpha * expected_kl
    assert abs(value.item() - expected) < 1e-7, "混合代價與線性組合不符"
    mixtures[str(alpha)] = {"expected": expected, "observed": value.item()}
    gradients[str(alpha)] = torch.autograd.grad(value, student)[0].tolist()
assert torch.allclose(distillation_loss(student, teacher, labels, alpha=0, temperature=1), ce)
assert torch.allclose(distillation_loss(student, teacher, labels, alpha=1, temperature=1), kl)
assert student.grad is None, "前向算例不應更新梯度欄位"
invalid_alpha = {}
for alpha in (-0.25, 1.25):
    try:
        distillation_loss(student, teacher, labels, alpha=alpha, temperature=1)
    except ValueError as error:
        invalid_alpha[str(alpha)] = str(error)
    else:
        raise AssertionError("範圍外的 alpha 沒有被拒絕")

# 明寫未經範圍檢查的公式，檢查正文對非法係數所作的數學敘述。
counterexamples = []
for alpha in (-0.25, 1.25):
    logits = torch.tensor([[[0.01, 0.99], [0.01, 0.99]]]).log().requires_grad_()
    hard = masked_loss(logits, labels)
    soft = distillation_kl(logits, teacher, labels, temperature=1)
    mixture = (1 - alpha) * hard + alpha * soft
    gradient = torch.autograd.grad(mixture, logits, retain_graph=True)[0]
    monitored = soft if alpha < 0 else hard
    monitored_gradient = torch.autograd.grad(monitored, logits)[0]
    directional_derivative = -(monitored_gradient * gradient).sum().item()
    updated = logits.detach() - 0.01 * gradient
    after = (
        distillation_kl(updated, teacher, labels, temperature=1)
        if alpha < 0
        else masked_loss(updated, labels)
    )
    assert directional_derivative < 0 and after.item() < monitored.item()
    counterexamples.append({
        "alpha": alpha,
        "student_probabilities": logits.detach().softmax(-1).tolist(),
        "teacher_probabilities": teacher.softmax(-1).tolist(),
        "labels": labels.tolist(),
        "monitored_objective": "教師 KL" if alpha < 0 else "正確答案 CE",
        "before": monitored.item(),
        "after_step_size_0.01": after.item(),
        "directional_derivative_along_negative_total_gradient": directional_derivative,
        "total_gradient": gradient.tolist(),
        "scope": "未經範圍檢查的數學式反例；正式函式會拒絕此 alpha",
    })

masked_student = torch.zeros(1, 3, 2, requires_grad=True)
masked_teacher = torch.tensor([[[0.8, 0.2], [0.01, 0.99], [0.8, 0.2]]]).log()
masked_labels = torch.tensor([[0, IGNORE, 1]])
masked_value = distillation_loss(masked_student, masked_teacher, masked_labels, alpha=0.5, temperature=1)
masked_gradient = torch.autograd.grad(masked_value, masked_student)[0]
assert torch.allclose(masked_value, distillation_loss(student, teacher, labels, alpha=0.5, temperature=1))
assert masked_gradient[:, 1].abs().sum().item() == 0
temperature_probes = []
for temperature in (1.0, 2.0, 4.0):
    p = (teacher / temperature).softmax(-1)
    log_q = (student / temperature).log_softmax(-1)
    raw = (p * (p.log() - log_q)).sum(-1).mean()
    scaled = distillation_kl(student, teacher, labels, temperature)
    combined = distillation_loss(student, teacher, labels, 0.5, temperature)
    assert torch.allclose(scaled, raw * temperature**2, atol=1e-7)
    assert torch.allclose(combined, 0.5 * ce + 0.5 * scaled, atol=1e-7)
    temperature_probes.append({"temperature": temperature, "teacher_probability": p.tolist(), "raw_kl": raw.item(), "scaled_kl": scaled.item()})

frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/18.9.md"
section = frozen.read_text()
displayed = section.split("```python\n")[2].split("```")[0]
assert ast.dump(ast.parse(displayed)) == ast.dump(ast.parse(inspect.getsource(distillation_loss)))
records_path = ROOT / "docs/course-experiments/results/distillation.json"
records = json.loads(records_path.read_bytes())
experiment_rows = []
for task_name, task in records["results"]["tasks"].items():
    for name, run in task["runs"].items():
        if name.endswith("ce_kl"):
            training = run["training"]
            assert training["alpha"] == 0.5 and training["temperature"] == 2.0
            assert training["temperature_squared_applied_once"] is True
            for entry in training["loss_trace"]:
                assert abs(entry["loss"] - 0.5 * entry["ce"] - 0.5 * entry["kl"]) < 1e-6
            experiment_rows.append({
                "task": task_name, "run": name,
                "split_counts": task["data"]["counts"],
                "training": {key: training[key] for key in ("steps", "batch_size", "training_examples", "training_sequence_chunks", "effective_supervised_tokens", "objective", "alpha", "temperature", "temperature_squared_applied_once")},
            })

# 原實驗版本以記錄內的程式指紋核對，避免把目前修改過的檔案冒充原版。
revision = records["revision"]
original_url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{revision}/scripts/course_experiments/compression.py"
original_raw, original_command = fetch(original_url)
assert sha(original_raw) == records["code_sha256"]["scripts/course_experiments/compression.py"]
original_text = original_raw.decode()
original_tree = ast.parse(original_text)
original_functions = {
    node.name: ast.get_source_segment(original_text, node)
    for node in original_tree.body
    if isinstance(node, ast.FunctionDef)
}
fit = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_text")
defaults = dict(zip([arg.arg for arg in fit.args.args][-len(fit.args.defaults):], [ast.literal_eval(v) for v in fit.args.defaults]))
assert defaults["alpha"] == 0.5 and defaults["temperature"] == 2.0
assert 'generated if method == "teacher_hard" else train_records' in original_functions["_distill_case"]
assert 'teacher_cache=cache if method == "ce_kl" else None' in original_functions["_distill_case"]
assert 'logits = model(x, valid=valid)["logits"]' in original_functions["_fit_text"]
assert 'logits[i, y[i] != IGNORE].detach().cpu()' in original_functions["_cache_text"]
assert '"logits": cache' in original_functions["_distill_case"]

tiny_records = [{"question": "q", "answer": "a"}, {"question": "q", "answer": "b"}]
tiny_teacher = TinyLM(ModelConfig(width=8, heads=2, max_length=32))
cache, _ = compression._cache_text(tiny_teacher, tiny_records, "cpu")
examples = compression._examples(tiny_records, 32)
x, y, valid = pad_batch(examples)
with torch.no_grad():
    expected_teacher = tiny_teacher(x, valid=valid)["logits"]
assert all(torch.equal(value, expected_teacher[i, y[i] != IGNORE]) for i, value in enumerate(cache))
assert all(not value.requires_grad for value in cache)

receipts = []
source_specs = [
    ("https://arxiv.org/html/1503.02531v1", "arXiv:1503.02531v1，2015-03-09", "Distillation；式 (1)、(2)", "a weighted average of two different objective functions", "核對教師與學生使用相同溫度、正確答案項使用溫度 1、軟目標梯度乘 T²。原文使用軟目標 CE；固定教師時它和 forward KL 對學生的梯度相同。"),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", "PyTorch 2.14 documentation", "CrossEntropyLoss；class-index targets 與 ignore_index", "The input is expected to contain the unnormalized logits for each class", "核對 class-index 交叉熵公式、忽略標記與有效位置分母；本地函式另外自行使用 sum/count。"),
    ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html", "PyTorch 2.14 documentation", "KLDivLoss；pointwise formula 與 reduction", "input in the log-space", "核對 input 是學生 log 機率、target 是教師比例；先逐候選求和，再平均有效位置才對應本文。"),
    ("https://www.deeplearningbook.org/contents/ml.html", "Goodfellow、Bengio、Courville，Deep Learning，2016 線上版", "Hyperparameters and Validation Sets；The Performance Measure", "The subset of data used to guide the selection", "核對驗證資料用於選擇超參數，測試資料保留作完成選擇後的評估；訓練目標本身不是任務品質。"),
]
for url, version, locator, quote, note in source_specs:
    raw, command = fetch(url)
    parser = PlainText()
    parser.feed(raw.decode())
    text = " ".join(" ".join(parser.parts).split())
    assert quote in text, f"原始來源沒有找到指定短摘錄：{url}"
    receipts.append({"url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(), "locator": locator, "command_argv": command, "response_sha256": sha(raw), "excerpt": quote, "inspection_note": note})

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-9/verify_facts.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_git_version": torch.version.git_version},
    "source_sha256": sha(frozen.read_bytes()),
    "example": {"student_shape": list(student.shape), "teacher_shape": list(teacher.shape), "labels": labels.tolist(), "valid_positions": int((labels != IGNORE).sum()), "student_probabilities": student.softmax(-1).tolist(), "teacher_probabilities": teacher.softmax(-1).tolist(), "ce": ce.item(), "kl": kl.item(), "mixtures": mixtures, "gradients": gradients, "student_unchanged": torch.equal(student.detach(), torch.zeros_like(student)), "alpha_quarter_teacher_term": 0.25 * expected_kl, "alpha_quarter_teacher_fraction": 0.25 * expected_kl / (0.75 * expected_ce + 0.25 * expected_kl)},
    "invalid_alpha_rejections": invalid_alpha,
    "negative_coefficient_counterexamples": counterexamples,
    "masked_probe": {"ignore_index": IGNORE, "valid_positions": int((masked_labels != IGNORE).sum()), "loss": masked_value.item(), "ignored_gradient": masked_gradient[:, 1].tolist()},
    "temperature_probes": temperature_probes,
    "displayed_implementation_ast_matches": True,
    "experiment_record": {"path": records_path.relative_to(ROOT).as_posix(), "sha256": sha(records_path.read_bytes()), "revision": revision, "seed": records["seed"], "environment": {key: records[key] for key in ("device", "gpu", "python_version", "torch_version")}, "ce_kl_runs": experiment_rows, "scope": "只核對紀錄與固定配方，不重新執行歷史訓練，也不證明不存在未記錄的人為探索"},
    "original_experiment_code": {"url": original_url, "command_argv": original_command, "sha256": sha(original_raw), "version": revision, "inspected_functions": ["_example", "_examples", "_cache_text", "_fit_text", "_distill_case", "_evaluate", "run_distillation"], "fit_defaults": defaults, "relevant_excerpts": {"cache": original_functions["_cache_text"], "fit": original_functions["_fit_text"]}, "inspection_note": "CE+KL 用原始 train_records；teacher_hard 才用生成回答。教師 logits 在訓練前快取；每次更新重算學生 logits。固定預設 alpha=0.5、temperature=2；呼叫沒有參數搜尋或測試回饋調參。"},
    "cache_probe": {"records": tiny_records, "cached_shapes": [list(value.shape) for value in cache], "matches_gold_prefix_logits": True, "cached_requires_grad": [value.requires_grad for value in cache], "scope": "兩筆人工對話的 CPU 前向核對，沒有模型訓練"},
    "external_source_receipts": receipts,
    "result": "數值、有效位置、溫度與固定配方核對完成；兩個負係數的總梯度反例成立",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
