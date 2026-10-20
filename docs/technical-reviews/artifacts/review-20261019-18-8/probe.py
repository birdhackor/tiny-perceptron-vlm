"""核對蒸餾算例、有效目標、梯度與前文的作用。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.posttraining import exact_kl

torch.set_num_threads(1)
torch.manual_seed(97)


def case(labels, temperature=1.0, dtype=torch.float32):
    student = torch.zeros(1, 3, 2, dtype=dtype, requires_grad=True)
    teacher = torch.tensor([[[0.8, 0.2]] * 3], dtype=dtype).log().requires_grad_()
    loss = distillation_kl(student, teacher, labels, temperature)
    loss.backward()
    return {
        "loss": loss.item(),
        "gradient": student.grad.tolist(),
        "teacher_gradient": None if teacher.grad is None else teacher.grad.tolist(),
        "dtype": str(loss.dtype),
    }


labels = torch.tensor([[-100, 0, 1]])
base = case(labels)
all_valid = case(torch.tensor([[0, 0, 1]]))
relabeled = case(torch.tensor([[-100, 123, -17]]))
assert base == relabeled
assert abs(base["loss"] - 0.19274475702175753) < 1e-7
assert torch.allclose(torch.tensor(base["gradient"]), torch.tensor([[[0., 0.], [-.15, .15], [-.15, .15]]]))
assert torch.allclose(torch.tensor(all_valid["gradient"]), torch.tensor([[[-.1, .1]] * 3]))
assert base["teacher_gradient"] is None

errors = {}
for name, student, teacher, targets, temperature in [
    ("shape", torch.zeros(1, 3, 2), torch.zeros(1, 3, 3), labels, 1.0),
    ("zero_temperature", torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), labels, 0.0),
    ("negative_temperature", torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), labels, -1.0),
    ("all_ignored", torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), torch.full((1, 3), IGNORE), 1.0),
]:
    try:
        distillation_kl(student, teacher, targets, temperature)
    except ValueError as error:
        errors[name] = str(error)
    else:
        raise AssertionError("預期的輸入檢查没有報錯")

temperature_results = []
for temperature in (1.0, 2.0, 4.0):
    teacher = torch.tensor([[[1.2, -0.3], [0.4, 1.1], [2., -1.]]], requires_grad=True)
    student = torch.tensor([[[0.1, 0.7], [0., 0.], [-1., 1.]]], requires_grad=True)
    p = (teacher.detach() / temperature).softmax(-1)
    log_q = (student / temperature).log_softmax(-1)
    manual = (p * (p.log() - log_q)).sum(-1)[labels != IGNORE].mean() * temperature**2
    kl = distillation_kl(student, teacher, labels, temperature)
    ce = -(p * log_q).sum(-1)[labels != IGNORE].mean() * temperature**2
    kl_gradient = torch.autograd.grad(kl, student, retain_graph=True)[0]
    ce_gradient = torch.autograd.grad(ce, student, retain_graph=True)[0]
    difference = (kl_gradient - ce_gradient).abs().max().item()
    assert torch.allclose(manual, kl, atol=1e-6)
    assert difference == 0
    temperature_results.append({"temperature": temperature, "loss": kl.item(), "explicit_formula": manual.item(), "ce_kl_gradient_max_difference": difference})

student = torch.zeros(1, 3, 2)
teacher = torch.tensor([[[.8, .2]] * 3]).log()
perturbed = student.clone()
perturbed[:, 0] = torch.tensor([400., -400.])
assert torch.equal(distillation_kl(student, teacher, labels), distillation_kl(perturbed, teacher, labels))

raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/18.8.md").read_text()
fences = re.findall(r"```python\n(.*?)```", raw, re.S)
notebook_path = ROOT / "notebooks/18/18.8.ipynb"
notebook = json.loads(notebook_path.read_bytes())
notebook_cells = [
    "".join(cell["source"]).strip()
    for cell in notebook["cells"]
    if cell["cell_type"] == "code" and not cell.get("metadata", {}).get("course_setup")
]
assert notebook_cells == [code.strip() for code in fences]
tree = ast.parse(fences[1])
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
excerpt = ast.get_source_segment(fences[1], function)
library_function = inspect.getsource(distillation_kl).strip()
assert excerpt == library_function
namespace = {}
notebook_output = io.StringIO()
with contextlib.redirect_stdout(notebook_output):
    exec(fences[0], namespace)
    library_loss = namespace["loss"]
    exec(fences[1], namespace)
    assert namespace["distillation_kl"] is not distillation_kl
    exec(fences[2], namespace)
assert torch.equal(namespace["again"], library_loss)

messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
inputs, targets = render_chat(messages)
batch_inputs, batch_targets, attention_valid = pad_batch([(inputs, targets), (inputs[:-1], targets[:-1])])
assert (batch_targets[~attention_valid] == IGNORE).all()
model = TinyLM(ModelConfig(vocab_size=264, width=8, heads=1, layers=1, max_length=32))
embedded = model.embedding(inputs.unsqueeze(0)).detach().requires_grad_()
logits = model(embeddings=embedded)["logits"]
teacher_logits = torch.zeros_like(logits)
context_loss = distillation_kl(logits, teacher_logits, targets.unsqueeze(0), 1.0)
context_loss.backward()
context_gradient = embedded.grad[0].norm(dim=-1)
ignored_prefix = (targets == IGNORE).nonzero().flatten()
assert (context_gradient[ignored_prefix] > 0).any()

p, q = [.8, .2], [.5, .5]
terms = [a * (math.log(a) - math.log(b)) for a, b in zip(p, q, strict=True)]
ce = -sum(a * math.log(b) for a, b in zip(p, q, strict=True))
entropy = -sum(a * math.log(a) for a in p)
numeric = {"terms": terms, "kl": sum(terms), "reverse_kl": sum(b * math.log(b / a) for a, b in zip(p, q, strict=True)), "ce": ce, "entropy": entropy, "ce_minus_entropy": ce - entropy}
policy = torch.tensor([[.8, .2]]).log().requires_grad_()
reference = torch.tensor([[.5, .5]]).log().requires_grad_()
policy_kl = exact_kl(policy, reference)
policy_kl.sum().backward()
assert policy.grad is not None and reference.grad is None
assert abs(policy_kl.item() - numeric["kl"]) < 1e-7
table = []
for length in (1, 2, 3):
    standard = "小貓追球"[:length]
    teacher_prefix = "小貓追球"[:length]
    student_prefix = "小狗跑開"[:length]
    table.append({"standard_prefix_for_both": standard, "free_teacher_prefix": teacher_prefix, "free_student_prefix": student_prefix})
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-8/probe.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "platform": platform.platform(), "device": "cpu", "seed": "97", "threads": str(torch.get_num_threads())},
    "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    "base": base, "all_valid": all_valid, "relabeled": relabeled,
    "numeric": numeric, "input_errors": errors,
    "policy_reference_example": {"kl": policy_kl.item(), "policy_gradient": policy.grad.tolist(), "reference_gradient": None},
    "prefix_table": table,
    "float64_input_result": case(labels, dtype=torch.float64),
    "temperature_and_ce": temperature_results,
    "ignored_logits_perturbation_preserves_loss": True,
    "excerpt_matches_library": True,
    "notebook_code_cells_match_frozen_fences_in_order": True,
    "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    "excerpt_physical_lines": len(excerpt.splitlines()),
    "excerpt_redefines_imported_function": True,
    "excerpt_loss_equal_library_loss": True,
    "notebook_output": notebook_output.getvalue(),
    "ignore_constant": IGNORE,
    "padding_targets_ignored": True,
    "attention_valid": attention_valid.tolist(),
    "ignored_context_positions": ignored_prefix.tolist(),
    "input_embedding_gradient_norms": context_gradient.tolist(),
    "scope": "只驗證 CPU 確定性算例。忽略目標不等於遮住前文；PAD 的 attention 遮罩另由模型處理。本程式没有訓練模型。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
