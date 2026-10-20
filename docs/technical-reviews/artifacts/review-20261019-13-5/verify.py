"""核對偏好損失的算例、梯度、程式及既有測量紀錄；不重新訓練模型。"""

import ast
import hashlib
import html
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = HERE / name
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == content
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


torch.set_num_threads(1)
command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT))
environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}
run = ROOT / "outputs/reviewer-tools/runs/20261020T060551736482Z"
copied = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
    raw = (run / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "utf8_content": raw.decode("utf-8")}
assert json.loads(copied["execution.json"]["utf8_content"])["exit_code"] == 0
assert json.loads(copied["extraction.json"]["utf8_content"])["source_sha256"] == (
    "c68aff1fc53934f5fc4d62ef289d8906e4c82c7b481767d6758b386692fd16eb"
)
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/13.md#13.5" --execute',
    "result": "三個原始 Python 區塊依序執行成功，stderr 為空。",
    "environment": environment,
    "copied_original_files": copied,
})

source = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.5.md").read_text()
fences = re.findall(r"```python\n(.*?)```", source, re.S)
assert len(fences) == 3
assert ast.dump(ast.parse(fences[1])) == ast.dump(ast.parse(inspect.getsource(dpo_loss)))

cases = []
for initial in (-4.0, -5.0):
    chosen = torch.tensor([initial], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    reference_chosen = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    reference_rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    before = [value.detach().clone() for value in (chosen, rejected, reference_chosen, reference_rejected)]
    loss = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1)
    loss.backward()
    margin = initial + 4.0
    expected_loss = math.log1p(math.exp(-0.1 * margin))
    expected_gradient = -0.1 / (1 + math.exp(0.1 * margin))
    h = 1e-5
    plus = math.log1p(math.exp(-0.1 * (margin + h)))
    minus = math.log1p(math.exp(-0.1 * (margin - h)))
    finite_difference = (plus - minus) / (2 * h)
    assert math.isclose(loss.item(), expected_loss, abs_tol=1e-12)
    assert math.isclose(chosen.grad.item(), expected_gradient, abs_tol=1e-12)
    assert math.isclose(rejected.grad.item(), -expected_gradient, abs_tol=1e-12)
    assert math.isclose(finite_difference, expected_gradient, abs_tol=1e-10)
    assert reference_chosen.grad is None and reference_rejected.grad is None
    assert all(torch.equal(a, b) for a, b in zip(before, (chosen, rejected, reference_chosen, reference_rejected)))
    cases.append({
        "policy_chosen": initial, "policy_rejected": -3.0,
        "reference_chosen": -4.0, "reference_rejected": -3.0,
        "beta": 0.1, "margin": margin, "loss": loss.item(),
        "chosen_gradient": chosen.grad.item(), "rejected_gradient": rejected.grad.item(),
        "formula_gradient": expected_gradient, "finite_difference_gradient": finite_difference,
        "reference_gradients": [None, None], "backward_preserves_all_input_values": True,
        "dtype": str(chosen.dtype), "shape": list(chosen.shape),
    })

batch_chosen = torch.tensor([-4.0, -5.0], dtype=torch.float64, requires_grad=True)
batch_loss = dpo_loss(batch_chosen, torch.tensor([-3.0, -3.0]), torch.tensor([-4.0, -4.0]), torch.tensor([-3.0, -3.0]))
batch_loss.backward()
assert math.isclose(batch_loss.item(), sum(x["loss"] for x in cases) / 2, abs_tol=1e-12)
assert torch.allclose(batch_chosen.grad, torch.tensor([x["chosen_gradient"] / 2 for x in cases], dtype=torch.float64))

invalid = []
for beta in (0.0, -0.1):
    try:
        dpo_loss(torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-4.0]), torch.tensor([-3.0]), beta=beta)
    except ValueError as error:
        invalid.append({"beta": beta, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("非正的實數 beta 應中止計算")

z = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=torch.float32)
stable = F.logsigmoid(z)
direct = torch.sigmoid(z).log()
assert torch.isfinite(stable).all()
assert torch.isneginf(direct[:2]).all()
stable_record = {"z": z.tolist(), "dtype": str(z.dtype), "logsigmoid": stable.tolist(),
                 "log_after_sigmoid": [str(float(x)) for x in direct]}

paper_update = {"learning_rate": 0.1, "chosen": -4.0 - 0.1 * cases[0]["chosen_gradient"],
                "rejected": -3.0 - 0.1 * cases[0]["rejected_gradient"]}
paper_update["policy_margin"] = paper_update["chosen"] - paper_update["rejected"]
paper_update["relative_margin"] = paper_update["policy_margin"] - (-1.0)
for key, expected in {"chosen": -3.995, "rejected": -3.005, "policy_margin": -0.99, "relative_margin": 0.01}.items():
    assert math.isclose(paper_update[key], expected, abs_tol=1e-12)

logits = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], dtype=torch.float64)
labels = torch.tensor([[IGNORE, 0, 2]])
score = sequence_log_probability(logits, labels)
assert math.isclose(score.item(), 2 * math.log(math.exp(2) / (math.exp(2) + 2)), abs_tol=1e-12)
tok = ByteTokenizer()
targets = {}
for answer in ("6", "7"):
    x, y = render_chat([{"role": "user", "content": "4+2=?"}, {"role": "assistant", "content": answer}])
    targets[answer] = y[y != IGNORE].tolist()
    assert targets[answer] == tok.encode(answer) + [tok.eos_id]

record_path = ROOT / "docs/course-experiments/results/dpo.json"
record = json.loads(record_path.read_bytes())
model_run = record["results"]["runs"]["model"]
preference = model_run["preference"]["test"]
arithmetic = model_run["arithmetic"]["test"]
pair = next(x for x in preference["samples"] if x["prompt"] == "4+2=?")
generated = next(x for x in arithmetic["samples"] if x["messages"] == [{"role": "user", "content": "4+2=?"}])
assert model_run["beta"] == 0.1 and model_run["training"]["steps"] == 250
assert (pair["chosen"], pair["rejected"]) == ("6", "7")
assert pair["policy_chosen_logp"] > pair["policy_rejected_logp"]
assert abs(pair["policy_chosen_logp"] - (-24.88083)) < 0.000005
assert abs(pair["policy_rejected_logp"] - (-28.11597)) < 0.000005
assert generated["expected"] == "6" and generated["generated"] == "8"
assert generated["generated_ids"] == tok.encode("8") + [tok.eos_id]
assert generated["eos"] is True and generated["exact"] is False
repo_files = {}
historical_code = []
for name in ("tiny_perceptron/alignment.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
             "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"):
    actual = sha((ROOT / name).read_bytes())
    expected = record["code_sha256"][name]
    if actual != expected:
        assert name == "scripts/course_experiments/behavior.py"
        url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{record['revision']}/{name}"
        argv = ["curl", "-L", "--fail", "--max-time", "45", "-sS", url]
        response = subprocess.run(argv, capture_output=True, check=True)
        assert sha(response.stdout) == expected
        old_functions = {node.name: ast.dump(node) for node in ast.parse(response.stdout).body if isinstance(node, ast.FunctionDef)}
        current_functions = {node.name: ast.dump(node) for node in ast.parse((ROOT / name).read_bytes()).body if isinstance(node, ast.FunctionDef)}
        checked = {}
        for identifier in ("_preference_parts", "_pair_examples", "_dpo_train", "_preference_evaluate", "run_dpo"):
            assert old_functions[identifier] == current_functions[identifier]
            checked[identifier] = {"ast_sha256": sha(old_functions[identifier].encode()), "equal": True}
        historical_code.append({"url": url, "version": record["revision"], "command_argv": argv,
                                "accessed_on": datetime.now(UTC).date().isoformat(), "exit_code": response.returncode,
                                "stderr": response.stderr.decode(), "original_sha256": expected,
                                "current_sha256": actual, "checked_functions": checked, "fulltext_saved": False})
    repo_files[name] = actual

save("verification.json", {
    "command": command, "environment": environment, "result": "全部斷言通過；既有實驗只讀取紀錄，未重新訓練或載入權重。",
    "source_sha256": sha(source.encode()), "excerpt_ast_equals_repository_function": True,
    "single_pair_cases": cases, "batch": {"pairs": 2, "mean_loss": batch_loss.item(), "chosen_gradients": batch_chosen.grad.tolist()},
    "nonpositive_beta": invalid, "numerical_stability": stable_record,
    "paper_gradient_descent": paper_update, "masked_sequence_score": score.tolist(),
    "answer_targets_with_eos": targets,
    "record": {"path": record_path.relative_to(ROOT).as_posix(), "sha256": sha(record_path.read_bytes()),
               "revision": record["revision"], "seed": record["seed"], "device": record["device"],
               "torch_version": record["torch_version"], "python_version": record["python_version"], "gpu": record["gpu"],
               "beta": model_run["beta"], "training_steps": model_run["training"]["steps"],
               "effective_answer_tokens_both_sides": model_run["training"]["effective_answer_tokens_both_sides"],
               "test_records": preference["records"], "selected_prompt_count": 1,
               "preference_sample": pair, "generated_sample": generated,
               "checked_current_code_sha256": repo_files, "historical_code_receipts": historical_code},
    "scope": "本機 CPU 的確定性驗證，以及指定既有單 seed 實驗紀錄的逐欄重算；不證明一般回答品質。",
})

commit = str(torch.version.git_version)
specs = [
    ("dpo_paper", "https://arxiv.org/html/2305.18290v3", "arXiv:2305.18290v3", ["(7)", "gradient of the loss"],
     "DPO 目標可整理為負 log-sigmoid 的相對 log 機率差；梯度由偏好差距調節。",
     "What does the DPO update do?"),
    ("tensor_api", f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_tensor.py", commit,
     ["    def backward(", "    detach ="], "backward 累積葉節點的梯度；detach 的結果不再需要梯度，仍共用儲存。",
     "Computes the gradient of current tensor wrt graph leaves."),
    ("functional_api", f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/functional.py", commit,
     ["logsigmoid =", "def sigmoid("], "F.logsigmoid 的介面定義等於 log(sigmoid(x))，sigmoid 為 logistic 函數。",
     "logsigmoid(input) -> Tensor"),
    ("stable_kernel", f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/aten/src/ATen/native/cpu/Activation.cpp", commit,
     ["void log_sigmoid_cpu_kernel", "Vec output_vec = min_vec - buffer_vec.log1p();"],
     "CPU kernel 使用 min(x,0)-log1p(exp(-abs(x)))，避免先計算接近零的 sigmoid。",
     "Vec output_vec = min_vec - buffer_vec.log1p();"),
]
receipts = []
for identifier, url, version, locators, note, excerpt in specs:
    argv = ["curl", "-L", "--fail", "--max-time", "45", "-sS", url]
    response = subprocess.run(argv, capture_output=True, check=True)
    raw = response.stdout
    text = raw.decode("utf-8")
    searchable = html.unescape(re.sub("<[^>]+>", " ", text)) if identifier == "dpo_paper" else text
    if identifier == "dpo_paper":
        searchable = re.sub(r"\s+", " ", searchable)
    assert all(x in searchable for x in locators)
    assert excerpt in searchable
    receipts.append({"id": identifier, "url": url, "version": version,
                     "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": argv,
                     "exit_code": response.returncode, "stderr": response.stderr.decode(),
                     "downloaded_bytes": len(raw), "original_sha256": sha(raw),
                     "locators": locators, "short_excerpt": excerpt, "inspection_note": note,
                     "checked_original": True, "fulltext_saved": False})
save("original-sources.json", {"sources": receipts, "scope": "curl 在記憶體讀取原文；只保存收據與必要短摘錄。"})
print(json.dumps({"result": "全部驗證通過", "environment": environment,
                  "artifacts": ["section-execution.json", "verification.json", "original-sources.json"]}, ensure_ascii=False))
