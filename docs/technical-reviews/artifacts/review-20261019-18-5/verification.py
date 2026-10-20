"""以固定分數核對溫度、梯度、圖寬與實驗設定；不執行訓練。"""

import ast
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.alignment import distillation_kl, distillation_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gradient(teacher, student, temperature):
    t = torch.tensor(teacher, dtype=torch.float64)
    s = torch.tensor(student, dtype=torch.float64, requires_grad=True)
    p = (t / temperature).softmax(0)
    q = (s / temperature).softmax(0)
    loss = -(p * (s / temperature).log_softmax(0)).sum()
    loss.backward()
    exact = (q.detach() - p) / temperature
    assert torch.allclose(s.grad, exact, atol=1e-14, rtol=1e-12)
    size = s.grad.norm().item()
    return {
        "temperature": temperature,
        "p": p.tolist(),
        "q": q.detach().tolist(),
        "probability_difference_norm": (q.detach() - p).norm().item(),
        "gradient": s.grad.tolist(),
        "gradient_norm": size,
        "manual_norm": math.sqrt(sum(x * x for x in s.grad.tolist())),
        "scaled_gradient_norm": size * temperature**2,
    }


def fetch(url):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return raw, {"command": command, "url": url, "bytes": len(raw), "sha256": sha(raw)}


result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "torch.float64（公式與反例）；torch.float32（專案函式）",
    },
    "source_files": {},
}
for name in (
    "course/figures/p7-18-temperature.svg",
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/multimodal.py",
    "scripts/course_experiments/compression.py",
    "scripts/course_experiments/modalities.py",
):
    result["source_files"][name] = sha((ROOT / name).read_bytes())

result["probabilities"] = []
for temperature in (1.0, 2.0, 4.0):
    scores = torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64)
    p = (scores / temperature).softmax(0)
    doubled = (scores * 2 / (temperature * 2)).softmax(0)
    assert torch.equal(p, doubled)
    result["probabilities"].append(
        {
            "temperature": temperature,
            "probabilities": p.tolist(),
            "percentages": [round(100 * x, 2) for x in p.tolist()],
            "ratio": (p[0] / p[1]).item(),
            "ratio_from_exp": math.exp(3 / temperature),
            "sum_after_dividing_probabilities": (p / temperature).sum().item(),
            "doubling_scores_and_temperature_equal": True,
        }
    )

ns = {"s": "http://www.w3.org/2000/svg"}
svg = ET.fromstring((ROOT / "course/figures/p7-18-temperature.svg").read_bytes())
result["figure_widths"] = []
for group, row in zip(svg.findall("s:g", ns), result["probabilities"], strict=True):
    rects = group.findall("s:rect", ns)
    widths = [float(r.attrib["width"]) for r in rects]
    expected = [352 * x for x in row["probabilities"]]
    errors = [abs(x - y) for x, y in zip(widths, expected, strict=True)]
    assert max(errors) < 0.000001
    assert abs(sum(widths) - 352) < 0.000002
    result["figure_widths"].append(
        {
            "temperature": row["temperature"],
            "widths": widths,
            "expected_widths": expected,
            "maximum_absolute_error": max(errors),
            "total_width": sum(widths),
            "colors": [r.attrib["fill"] for r in rects],
        }
    )

result["example_gradients"] = [gradient([4, 1, 0], [0, 0, 0], t) for t in (1, 2, 4)]
result["counterexample"] = {
    "teacher_logits": [10, 0, 0],
    "student_logits": [20, 0, 0],
    "rows": [gradient([10, 0, 0], [20, 0, 0], t) for t in (1, 2, 4, 100, 1000)],
    "reason": "兩邊本來都極偏第一名，升溫可先拉大兩者的機率差，梯度也可增加。",
}
rows = result["counterexample"]["rows"]
result["counterexample"]["gradient_ratio_t4_over_t1"] = rows[2]["gradient_norm"] / rows[0]["gradient_norm"]
result["counterexample"]["scaled_gradient_ratio_t4_over_t1"] = (
    rows[2]["scaled_gradient_norm"] / rows[0]["scaled_gradient_norm"]
)
assert rows[2]["gradient_norm"] > rows[0]["gradient_norm"]

result["library_scaling"] = []
for temperature in (1.0, 2.0, 4.0):
    teacher = torch.tensor([[[4.0, 1.0, 0.0]]])
    student = torch.zeros_like(teacher, requires_grad=True)
    labels = torch.tensor([[0]])
    loss = distillation_kl(student, teacher, labels, temperature)
    loss.backward()
    p = (teacher / temperature).softmax(-1)
    q = (student.detach() / temperature).softmax(-1)
    unscaled = (p * (p.log() - q.log())).sum()
    expected = temperature * (q - p)
    assert torch.allclose(student.grad, expected, atol=1e-6)
    mixed = distillation_loss(student, teacher, labels, 0.5, temperature)
    ce = -student.log_softmax(-1)[0, 0, 0]
    assert torch.allclose(mixed, 0.5 * ce + 0.5 * loss)
    result["library_scaling"].append(
        {
            "temperature": temperature,
            "kl_with_scaling": loss.item(),
            "kl_without_scaling": unscaled.item(),
            "ratio": (loss / unscaled).item(),
            "gradient": student.grad.tolist(),
            "mixed_loss": mixed.item(),
        }
    )

result["experiment_settings"] = []
historical_command = [
    "git", "show",
    "dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17:scripts/course_experiments/compression.py",
]
historical_raw = subprocess.run(historical_command, check=True, capture_output=True).stdout
historical_tree = ast.parse(historical_raw.decode())
current_tree = ast.parse((ROOT / "scripts/course_experiments/compression.py").read_text())
historical_functions = {n.name: n for n in historical_tree.body if isinstance(n, ast.FunctionDef)}
current_functions = {n.name: n for n in current_tree.body if isinstance(n, ast.FunctionDef)}
result["historical_training_code"] = {
    "command": historical_command,
    "sha256": sha(historical_raw),
    "functions": {},
}
for name in ("_fit_text", "_fit_modal", "_distill_case", "run_distillation", "_evaluate"):
    old = ast.dump(historical_functions[name], include_attributes=False)
    current = ast.dump(current_functions[name], include_attributes=False)
    assert old == current
    result["historical_training_code"]["functions"][name] = {"ast_equal": True}
for name in ("distillation", "multimodal_distillation"):
    path = ROOT / f"docs/course-experiments/results/{name}.json"
    record = json.loads(path.read_bytes())
    assert record["code_sha256"]["scripts/course_experiments/compression.py"] == sha(historical_raw)
    result["source_files"][path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
    for task_name, task in record["results"]["tasks"].items():
        for run_name, run in task["runs"].items():
            training = run.get("training")
            if not training or not training["temperature_squared_applied_once"]:
                continue
            assert training["temperature"] == 2.0
            assert training.get("alpha", 0.5) == 0.5
            code_match = {
                p: sha((ROOT / p).read_bytes()) == record["code_sha256"][p]
                for p in (
                    "tiny_perceptron/alignment.py",
                    "scripts/course_experiments/compression.py",
                    "scripts/course_experiments/modalities.py",
                    "tiny_perceptron/model.py",
                    "tiny_perceptron/multimodal.py",
                )
            }
            assert code_match["tiny_perceptron/alignment.py"]
            result["experiment_settings"].append(
                {
                    "experiment": name,
                    "task": task_name,
                    "run": run_name,
                    "seed": record["seed"],
                    "device": record["device"],
                    "torch": record["torch_version"],
                    "training_examples": training["training_examples"],
                    "steps": training["steps"],
                    "objective": training["objective"],
                    "temperature": training["temperature"],
                    "alpha": training.get("alpha", "由 objective 的兩個 0.5 核對"),
                    "temperature_squared_applied_once": training["temperature_squared_applied_once"],
                    "test_examples": run["test"]["examples"],
                    "generation": run["test"].get("generation", "由 evaluate_modal → _evaluate → generate_modal 預設值核對"),
                    "recorded_code_matches_current": code_match,
                }
            )

result["source_receipts"] = []
raw, receipt = fetch("https://arxiv.org/src/1503.02531v1")
with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
    tex = archive.extractfile("distillation11.tex").read().decode()
excerpt = "If the temperature is high compared with the magnitude of the logits, we can approximate:"
assert excerpt in tex
receipt.update(
    version="arXiv:1503.02531v1，2015-03-09",
    accessed_on="2026-10-20",
    locator="Distillation；Matching logits is a special case of distillation；公式 (1)–(4)",
    excerpt=excerpt,
    inspection="已讀 softmax、同溫度、硬軟目標混合、梯度精確式，以及高溫展開的前提；只保存此短摘錄。",
)
result["source_receipts"].append(receipt)

url = "https://raw.githubusercontent.com/pytorch/pytorch/" + torch.version.git_version + "/torch/nn/functional.py"
raw, receipt = fetch(url)
tree = ast.parse(raw.decode())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
for name in ("softmax", "log_softmax"):
    assert name in functions
receipt.update(
    version="PyTorch " + str(torch.__version__) + "；git " + torch.version.git_version,
    accessed_on="2026-10-20",
    locator="softmax、log_softmax",
    excerpt="It is applied to all slices along dim",
    inspection="已讀 softmax 公式、dim、總和為一的契約，以及轉呼叫 Tensor.softmax／Tensor.log_softmax 的實作。",
)
result["source_receipts"].append(receipt)

raw, receipt = fetch("https://scikit-learn.org/1.7/modules/cross_validation.html")
assert b"validation set" in raw and b"test set" in raw
receipt.update(
    version="scikit-learn 1.7.2 documentation",
    accessed_on="2026-10-20",
    locator="Cross-validation: evaluating estimator performance；開頭超參數、validation set、最終 test set 的說明",
    excerpt="evaluation is done on the validation set",
    inspection="已讀訓練、驗證、測試的用途，以及反覆按測試集調參會造成資訊洩漏的段落。",
)
result["source_receipts"].append(receipt)

print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
