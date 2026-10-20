"""核對修文、沿用證據與 LoRA 梯度的成立條件。"""

import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def text(path):
    return (ROOT / path).read_text()


original_path = "docs/course-review-20261019/reviews/rv1019_technical_8-8/technical-8-8-original-01.json"
assert digest(original_path) == "c2534bc11d875e6fc0fecb5488118d0df84adb7ad0ab71aa5047da6072924aee"
original = json.loads(text(original_path))
old_path = "docs/course-review-20261019/freeze-01/sources/8.8.md"
new_path = "docs/course-review-20261019/freeze-05/sources/8.8.md"
assert digest(new_path) == "e343fddd2b29b8a98dcdbd557292948b735e2e9e7705e14b47eb9580c9e5eb43"
assert digest("docs/course-review-20261019/freeze-05.json") == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
old = text(old_path)
new = text(new_path)
old_sentence = "B的輸入則是A算出的那2個數；A是隨機數，這2個數不是0，B的梯度也就不是0。"
changed_sentence = "B的輸入則是A算出的那2個數；A是隨機數，這2個數不是0，B才可能收到非零的梯度。"
added_sentence = "B的梯度還要看從輸出傳回的梯度，以及幾筆輸入加總後會不會互相抵消；本例的輸入與`sum()`目標下，B的梯度不是0。"
assert old.count(old_sentence) == 1
assert new == old.replace(old_sentence, changed_sentence + added_sentence)
chapter = (ROOT / "course/chapters/08.md").read_bytes()
start = re.search(rb"(?m)^## 8\.8 ", chapter).start()
next_heading = re.search(rb"(?m)^## ", chapter[start + 1:])
end = start + 1 + next_heading.start() if next_heading else len(chapter)
assert chapter[start:end] == (ROOT / new_path).read_bytes()
old_fences = re.findall(r"```python\n(.*?)```", old, re.S)
new_fences = re.findall(r"```python\n(.*?)```", new, re.S)
assert old_fences == new_fences and len(new_fences) == 2
quoted = next(n for n in ast.parse(new_fences[1]).body if isinstance(n, ast.ClassDef))
actual = next(n for n in ast.parse(text("tiny_perceptron/alignment.py")).body if isinstance(n, ast.ClassDef) and n.name == "LoRALinear")
assert ast.dump(quoted) == ast.dump(actual)

reused = []
for item in original["artifacts"]:
    actual_sha = digest(item["path"])
    assert actual_sha == item["sha256"]
    reused.append({"path": item["path"], "sha256": actual_sha, "unchanged": True})
repo_sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual_sha = digest(item["path"])
        assert actual_sha == item["sha256"]
        repo_sources.append({"path": item["path"], "sha256": actual_sha, "unchanged": True})
old_audit = json.loads(text("docs/technical-reviews/artifacts/review-20261019-8-8/verify-output.json"))
inputs = []
for path, expected_sha in old_audit["inputs_sha256"].items():
    actual_sha = digest(path)
    assert actual_sha == expected_sha
    inputs.append({"path": path, "sha256": actual_sha, "unchanged": True})
assert digest("course/figures/rewrite-08-lora-paths.svg") == original["figure_sha256"]["course/figures/rewrite-08-lora-paths.svg"]
png = "outputs/course-review-20261019/figure-renders-04/4349f1c9a64ae732be807f4ca9ec7cb084123465daec984a608d8c37ff35ba81.png"
assert digest(png) == "ba790169fccc500bf9a2fee999412cdf75900a474f879034c0ecbcb4b1527268"

torch.set_num_threads(1)
torch.manual_seed(0)
layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
x = torch.randn(3, 16)
h = x @ layer.a.T
initial_equal = torch.equal(layer(x), layer.base(x))
layer(x).sum().backward()
g = torch.ones(3, 12)
expected_b_grad = g.T @ h.detach()
max_formula_error = float((layer.b.grad - expected_b_grad).abs().max())
assert h.count_nonzero().item() == 6
assert max_formula_error < 1e-7
assert layer.a.grad.count_nonzero() == 0 and layer.b.grad.count_nonzero() > 0
example = {"seed": 0, "input_shape": list(x.shape), "rank": 2, "alpha": 2,
           "objective": "layer(x).sum()", "hidden": h.detach().tolist(),
           "initial_equal": initial_equal,
           "a_gradient_max": layer.a.grad.abs().max().item(),
           "b_gradient_max": layer.b.grad.abs().max().item(),
           "formula_error_max": max_formula_error}

layer.zero_grad(set_to_none=True)
(layer(x) * 0).sum().backward()
zero_upstream = {"hidden_nonzero": h.count_nonzero().item() == 6,
                 "objective": "(layer(x) * 0).sum()",
                 "b_gradient_max": layer.b.grad.abs().max().item()}
assert zero_upstream["b_gradient_max"] == 0

layer.zero_grad(set_to_none=True)
probe_g = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 10
(layer(x) * probe_g).sum().backward()
formula_error = float((layer.b.grad - probe_g.T @ (x @ layer.a.T).detach()).abs().max())
assert formula_error < 1e-7
weighted_upstream = {"objective": "(layer(x) * probe_g).sum()",
                     "upstream_gradient": probe_g.tolist(),
                     "b_gradient_max": layer.b.grad.abs().max().item(),
                     "formula_error_max": formula_error}

v = x[:1]
paired = torch.cat((v, -v))
layer.zero_grad(set_to_none=True)
paired_hidden = paired @ layer.a.T
layer(paired).sum().backward()
cancellation = {"input": paired.tolist(), "hidden": paired_hidden.detach().tolist(),
                "objective": "layer(paired).sum()", "b_gradient_max": layer.b.grad.abs().max().item()}
assert paired_hidden.count_nonzero().item() == 4 and cancellation["b_gradient_max"] == 0

result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-8/recheck-current-01.py",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "float32", "threads": "1"},
          "current_source": {"path": new_path, "sha256": digest(new_path)},
          "current_section_equals_freeze": True,
          "changed_sentences": [changed_sentence, added_sentence],
          "no_other_text_changes": True, "python_fences_unchanged": True,
          "quoted_class_equals_library": True,
          "original_report": {"path": original_path, "sha256": digest(original_path)},
          "reused_artifacts": reused, "repository_sources": repo_sources,
          "previous_audit_inputs": inputs,
          "figure": {"svg_sha256": digest("course/figures/rewrite-08-lora-paths.svg"), "png_sha256": digest(png)},
          "derivation": "Y=XW^T+b+s(XA^T)B^T，H=XA^T，G=∂L/∂Y。微分為dL=s tr(G^T H dB^T)，故∂L/∂B=s G^T H=s sum_i g_i h_i^T。若所有h_i=0則梯度必零；某些h_i非零只讓梯度可能非零。還需看G，各樣本的外積也可能抵消。對本例s=1、G=ones(3,12)，實際B梯度非零。",
          "seeded_example": example, "zero_upstream": zero_upstream,
          "weighted_upstream": weighted_upstream, "paired_cancellation": cancellation,
          "all_assertions_passed": True}
print(json.dumps(result, ensure_ascii=False, indent=2))
