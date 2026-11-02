"""Bounded independent checks of the frozen first-steps page; no training."""
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SNAP = ROOT / "docs/course-review-20261019/freeze-02/sources/first-steps.md"
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/collect_original.py"
torch.set_num_threads(1)
body = SNAP.read_text()
results = {"created_at": datetime.now(timezone.utc).isoformat(), "command": COMMAND,
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "torch_git": torch.version.git_version, "device": "cpu", "platform": platform.platform()},
           "source_sha256": hashlib.sha256(SNAP.read_bytes()).hexdigest(), "inline_blocks": [], "checks": {}}
namespace = {}
for n, match in enumerate(re.finditer(r"```python\n(.*?)```", body, re.S), 1):
    code = match[1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(SNAP) + f":python-block-{n}", "exec"), namespace)
    results["inline_blocks"].append({"id": n, "line": body[:match.start()].count("\n") + 1,
                                    "code": code, "stdout": buf.getvalue(), "error": None})

def check(name, fn):
    results["checks"][name] = fn()

def python_examples():
    animals = ["貓", "狗"]
    scores = [5, 9, 7]
    expressions = ['[5, 2, 7][:-1]', '[5, 2, 7][1:]', 'list(range(3))',
                   'len(set("貓狗貓"))', 'sorted([3, 1, 2])', 'sorted([3, 1, 2], reverse=True)',
                   'sorted(set("貓狗貓"))', 'max(["貓", "小花貓"], key=len)',
                   'max(range(3), key=lambda i: scores[i])', 'list(zip(["貓", "狗"], [3, 1]))',
                   'dict(zip(["貓", "狗"], [0, 1]))', 'list(enumerate(["貓", "狗"]))',
                   'list(reversed(["貓", "狗", "鳥"]))', '"".join(["貓", "看", "狗"])',
                   '"、".join(["貓", "狗"])', '[x * 2 for x in [1, 2, 3]]',
                   'sum(x * 2 for x in [1, 2, 3])', 'next(x for x in [3, 8, 12] if x > 5)',
                   'next((x for x in [3] if x > 5), None)', '7 // 2', '-7 // 2',
                   'ord("狗")', 'ord("貓")']
    env = {"scores": scores}
    values = {expression: repr(eval(expression, env)) for expression in expressions}
    values["function_first"] = animals[0]
    values["exercise_first_line"] = "鳥 3"
    changed = body[body.index('animals = ["貓", "狗"]\nrecord'):body.index('```', body.index('animals = ["貓", "狗"]\nrecord'))]
    changed = changed.replace('["貓", "狗"]', '["鳥", "貓", "狗"]').replace('== 2', '== 3')
    output = io.StringIO()
    with contextlib.redirect_stdout(output): exec(changed, {})
    values["exercise_stdout"] = output.getvalue()
    return values
check("W2_table_and_exercise", python_examples)

def tensor_examples():
    s = torch.tensor([[60., 70., 80.], [80., 90., 100.]])
    original = s.clone()
    result = {"shape": list(s.shape), "row": s[0].tolist(), "item": s[1, 2].item(),
              "means1": s.mean(dim=1).tolist(), "means0": s.mean(dim=0).tolist(),
              "means_negative1": s.mean(dim=-1).tolist(), "tuple_shape": repr(tuple(s.shape)),
              "tolist": s.tolist(), "reshape": s.reshape(6).tolist(),
              "reshape_roundtrip": torch.equal(s.reshape(6).reshape(2, 3), s),
              "no_change": torch.equal(original, s), "batch_slice": list(torch.zeros(2, 4, 3)[0].shape)}
    for key, fn in [("whole_item_error", lambda: s.item()), ("reshape8_error", lambda: s.reshape(8))]:
        try: fn()
        except Exception as e: result[key] = type(e).__name__ + ": " + str(e)
    s[0, 2] = 110
    result["exercise"] = [s.mean(dim=1).tolist(), s.mean(dim=0).tolist()]
    return result
check("W3_tensor", tensor_examples)

def log_examples():
    result = {"costs": {str(p): -math.log(p) for p in [0.9, 0.5, 0.1, 0.25]},
              "e": math.e, "inverse_e": math.e ** -1,
              "joint": 0.5 * 0.5, "log_identity": [-math.log(0.25), -math.log(0.5)-math.log(0.5)]}
    try: math.log(0)
    except Exception as e: result["log0_error"] = type(e).__name__ + ": " + str(e)
    logits = torch.tensor([[1., 2., 3.]])
    result["softmax_sum"] = logits.softmax(-1).sum().item()
    result["ce_raw"] = torch.nn.functional.cross_entropy(logits, torch.tensor([2])).item()
    result["negative_log_probability"] = -logits.log_softmax(-1)[0, 2].item()
    return result
check("W4_log", log_examples)

def linear_examples():
    recipes = torch.tensor([[2., 1., 0.], [1., 2., 1.]])
    weights = torch.tensor([[10., 0.], [20., 1.], [30., 4.]])
    layer = torch.nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(weights.T)
        layer.bias.copy_(torch.tensor([8., 0.]))
    result = {"matmul": (recipes @ weights).tolist(), "shape": list((recipes @ weights).shape),
              "weight_shape": list(layer.weight.shape), "bias8": layer(recipes).tolist()}
    try: recipes @ weights[:2]
    except Exception as e: result["incompatible_error"] = type(e).__name__ + ": " + str(e)
    return result
check("W5_linear", linear_examples)

def gradient_examples():
    result = {"loss_at1.1": (1.1-3)**2, "loss_decrease": 4-(1.1-3)**2,
              "derivatives": {str(x): 2*(x-3) for x in [1., 1.4]}, "steps": []}
    for lr in [.1, .6, 1.1]:
        w = torch.tensor(1., requires_grad=True)
        loss = (w-3).square()
        loss.backward()
        before = w.item()
        gradient = w.grad.item()
        with torch.no_grad(): w -= lr * w.grad
        result["steps"].append({"lr": lr, "before": before, "gradient": gradient,
                                "after": w.item(), "new_loss": (w-3).square().item()})
    w = torch.tensor(1., requires_grad=True)
    (w-3).square().backward()
    (w-3).square().backward()
    result["accumulated_gradient"] = w.grad.item()
    return result
check("W6_gradient", gradient_examples)

def paths_and_errors():
    previous = Path.cwd()
    result = {}
    for name, folder in [("root", ROOT), ("notebooks", ROOT / "notebooks")]:
        os.chdir(folder)
        output = io.StringIO()
        with contextlib.redirect_stdout(output): exec(results["inline_blocks"][-1]["code"], {})
        result[name] = output.getvalue()
    os.chdir(previous)
    for label, code in [("key", '{}["x"]'), ("index", '["貓", "狗"][2]'),
                        ("file", 'open("/nonexistent-firststeps-review-file")')]:
        try: exec(code)
        except Exception as e: result[label] = type(e).__name__ + ": " + str(e)
    result["directory_presence"] = {x: (ROOT / x).exists() for x in
                                    ["course/chapters", "notebooks/01/1.1.ipynb", "tiny_perceptron", "scripts", "pyproject.toml", ".venv"]}
    result["ignored_paths"] = subprocess.run(["git", "check-ignore", "data/", "checkpoints/", "outputs/", ".venv/"],cwd=ROOT,capture_output=True,text=True).stdout
    return result
check("W7_paths_errors", paths_and_errors)

spec = importlib.util.spec_from_file_location("independent_build_course", ROOT / "scripts/build_course.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
bootstrap = {}
for name, folder in [("root", ROOT), ("notebooks", ROOT / "notebooks"), ("lesson_folder", ROOT / "notebooks/01")]:
    original = Path.cwd()
    os.chdir(folder)
    context = {}
    exec(builder.BOOTSTRAP, context)
    bootstrap[name] = str(context["root"])
    os.chdir(original)
chapter = (ROOT / "course/chapters/01.md").read_text()
first_section = re.search(r'^## 1\.1 .*?\n(.*?)(?=^## )', chapter, re.S | re.M)[1]
first_code = re.search(r'```python\n(.*?)```', first_section, re.S)[1]
for label, code in [("original", first_code), ("bird_exercise", first_code.replace('貓看狗，狗看貓。', '鳥看狗，狗看鳥。'))]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output): exec(code, {})
    bootstrap[label] = {"code": code, "stdout": output.getvalue()}
results["checks"]["W1_bootstrap_and_exercise"] = bootstrap
results["result"] = "All eight frozen inline Python blocks executed; table/exercises and explicit error cases recorded. No GPU, training, external model, or Colab execution."
target = OUT / "cpu-execution-v1.json"
with target.open("x") as f: json.dump(results, f, ensure_ascii=False, indent=2); f.write("\n")
print(json.dumps({"output": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "inline_blocks": len(results["inline_blocks"]), "checks": list(results["checks"]), "result": results["result"]}, ensure_ascii=False))
