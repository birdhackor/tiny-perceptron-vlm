"""核對凍結暖身的程式、練習、導覽與本機環境；只輸出證據。"""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

ROOT = Path.cwd()
ARTIFACT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/course-review-20261019/freeze-02/sources/first-steps.md"
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                       text=True, capture_output=True, timeout=45)
    return {"command": args, "cwd": str(cwd), "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr}


result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
          "environment": {"python": platform.python_version(), "torch": torch.__version__,
                          "os": platform.platform(), "device": "cpu"}}
result["packages"] = {}
for name in ["torch", "numpy", "pillow", "soundfile", "tokenizers", "datasets", "huggingface-hub",
             "jupyterlab", "ipykernel", "nbclient"]:
    try:
        result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][name] = "未安裝"
result["versions"] = [run(["git", "--version"]), run(["uv", "--version"])]
freeze = ROOT / "docs/course-review-20261019/freeze-04.json"
page = next(p for p in json.loads(freeze.read_bytes())["pages"] if p["page_id"] == "first-steps")
result["freeze"] = {"manifest_sha256": sha(freeze), "page": page,
                    "snapshot_sha256": sha(SOURCE), "current_sha256": sha(ROOT / "course/first-steps.md")}
result["figures"] = {path: sha(ROOT / path) for path in page["figures_sha256"]}
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
result["figure_receipt"] = receipt if not isinstance(receipt, dict) else {
    k: v for k, v in receipt.items() if not isinstance(v, list)
}
result["png_sha256"] = {}
for digest in page["figures_sha256"].values():
    path = ROOT / "outputs/course-review-20261019/figure-renders-04" / (digest + ".png")
    result["png_sha256"][str(path.relative_to(ROOT))] = sha(path)

namespace = {"__name__": "__main__"}
blocks = []
text = SOURCE.read_text()
for match in re.finditer(r"```python\n(.*?)\n```", text, re.S):
    code = match.group(1)
    heading = re.findall(r"^## (.+)$", text[:match.start()], re.M)[-1]
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "暖身程式", "exec"), namespace)
    blocks.append({"heading": heading, "code": code, "stdout": stream.getvalue()})
result["python_blocks"] = blocks

table = [
    ("[5,2,7][:-1]", [5,2]), ("[5,2,7][1:]", [2,7]), ("list(range(3))", [0,1,2]),
    ('len(set("貓狗貓"))', 2), ("sorted([3,1,2])", [1,2,3]),
    ("sorted([3,1,2],reverse=True)", [3,2,1]), ('sorted(set("貓狗貓"))', ["狗","貓"]),
    ('ord("狗")', 29399), ('ord("貓")', 35987), ('max(["貓","小花貓"],key=len)', "小花貓"),
    ("max(range(3),key=lambda i:[5,9,7][i])", 1),
    ('list(zip(["貓","狗"],[3,1]))', [("貓",3),("狗",1)]),
    ('dict(zip(["貓","狗"],[0,1]))', {"貓":0,"狗":1}),
    ('list(enumerate(["貓","狗"]))', [(0,"貓"),(1,"狗")]),
    ('list(reversed(["貓","狗","鳥"]))', ["鳥","狗","貓"]),
    ('"".join(["貓","看","狗"])', "貓看狗"), ('"、".join(["貓","狗"])', "貓、狗"),
    ("[x*2 for x in [1,2,3]]", [2,4,6]), ("sum(x*2 for x in [1,2,3])", 12),
    ("next(x for x in [3,8,12] if x>5)", 8), ("next((x for x in [3] if x>5),None)", None),
    ('f"看到{animal}"', "看到貓"), ("7//2", 3), ("(3<<4)|5", 53)
]
result["table"] = []
for expression, expected in table:
    observed = eval(expression, {"animal": "貓"})
    assert observed == expected, expression
    result["table"].append({"expression": expression, "expected": expected, "observed": observed})
result["division_boundary"] = {"positive": 7 // 2, "negative": -7 // 2,
                               "truncate_negative": math.trunc(-7 / 2), "negative_remainder": -7 % 2,
                               "floating_operands": 7.0 // 2}
animals = ["鳥", "貓", "狗"]
result["animals_exercise"] = {"first": animals[0], "length": len(animals),
                              "loop": ["看到 " + x for x in animals]}
record = {"animal": "貓", "caption": "更換說明"}
assert animals == ["鳥", "貓", "狗"]
result["function"] = {"first": (lambda items: items[0])(["貓", "狗"])}
scores = namespace["scores"]
before = scores.clone()
result["tensor"] = {"shape_tuple": tuple(scores.shape), "tolist": scores.tolist(),
                    "row_tolist": scores[0].tolist(), "dim_last": scores.mean(-1).tolist(),
                    "reshape": scores.reshape(6).tolist(),
                    "restored": scores.reshape(6).reshape(2,3).tolist(),
                    "index_batch_shape": list(torch.zeros(2,4,3)[0].shape)}
assert torch.equal(scores, before)
for label, operation in [("item_multiple", lambda: scores.item()), ("reshape_mismatch", lambda: scores.reshape(8)),
                         ("matmul_mismatch", lambda: namespace["recipes"] @ torch.ones(2,2)),
                         ("index_outside", lambda: animals[3]), ("assert_false", lambda: exec("assert False")),
                         ("log_zero", lambda: math.log(0))]:
    try:
        operation()
    except Exception as error:
        result[label] = {"type": type(error).__name__, "message": str(error)}
scores[0,2] = 110
result["scores_exercise"] = {"per_person": scores.mean(1).tolist(), "per_exam": scores.mean(0).tolist()}
assert result["scores_exercise"] == {"per_person": [80.,90.], "per_exam": [70.,80.,105.]}
result["log_values"] = {str(p): -math.log(p) for p in [0.9,0.5,0.1,0.25]}
result["log_identity"] = {"joint": -math.log(0.5*0.5), "sum": -math.log(0.5)-math.log(0.5)}
logits = torch.tensor([[2.,-1.],[-1.,2.]])
labels = torch.tensor([0,1])
result["cross_entropy"] = {"raw_logits": logits.tolist(), "labels": labels.tolist(),
    "softmax": logits.softmax(-1).tolist(), "loss": torch.nn.functional.cross_entropy(logits,labels).item(),
    "manual_mean_nll": (-logits.log_softmax(-1)[torch.arange(2),labels]).mean().item()}
layer = namespace["layer"]
with torch.no_grad():
    layer.bias[0] = 8
result["bias_exercise"] = layer(namespace["recipes"]).tolist()
result["gradient_exercises"] = []
for lr in [0.1,0.6,1.1]:
    w = torch.tensor(1.,requires_grad=True)
    loss = (w-3).square()
    loss.backward()
    old_w, grad = w.item(), w.grad.item()
    with torch.no_grad():
        w -= lr*w.grad
    result["gradient_exercises"].append({"lr": lr, "before": old_w, "loss": loss.item(), "gradient": grad,
        "after": w.item(), "new_loss": (w-3).square().item()})
w = torch.tensor(1.4,requires_grad=True)
(w-3).square().backward()
first = w.grad.item()
(w-3).square().backward()
result["gradient_accumulation"] = {"at_1_4": first, "after_second_backward": w.grad.item()}
result["finite_differences"] = [{"h": h, "slope": (((1+h)-3)**2-(1-3)**2)/h}
                                 for h in [0.1,0.01,0.001]]
notebook = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_bytes())
setup = "".join(next(c for c in notebook["cells"] if c.get("metadata",{}).get("course_setup"))["source"])
result["notebook_bootstrap"] = {"path": "notebooks/01/1.1.ipynb", "sha256": sha(ROOT / "notebooks/01/1.1.ipynb"),
                                 "source": setup}
result["notebook_locations"] = []
for directory in [ROOT, ROOT / "notebooks", ROOT / "notebooks/01"]:
    start = directory
    while not (start / "pyproject.toml").exists() and start != start.parent:
        start = start.parent
    result["notebook_locations"].append({"cwd": str(directory.relative_to(ROOT)), "found": str(start),
        "project_exists": (directory / "pyproject.toml").exists(),
        "package_exists": (directory / "tiny_perceptron").exists()})
result["roundtrip_exercise"] = []
for text_input in ["貓看狗，狗看貓。", "鳥看狗，狗看鳥。"]:
    ns = {}
    output = io.StringIO()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and not cell.get("metadata"):
            code = "".join(cell["source"]).replace('text = "貓看狗，狗看貓。"', "text = " + repr(text_input))
            with contextlib.redirect_stdout(output):
                exec(compile(code, "字元還原練習", "exec"), ns)
    result["roundtrip_exercise"].append({"input": text_input,"stdout": output.getvalue(),"restored":ns["restored"]})
result["check_env"] = run(["uv","run","--no-sync","python","scripts/check_env.py"])
result["train_dry_run"] = run(["uv","run","--no-sync","python","scripts/train.py","--device","cpu",
    "--steps","1","--width","8","--max-length","8","--batch-size","1",
    "--output",str(ARTIFACT / "unused-checkpoint.pt")])
result["checkpoint_created"] = (ARTIFACT / "unused-checkpoint.pt").exists()

for path in ["course/chapters", "notebooks", "tiny_perceptron", "scripts"]:
    assert (ROOT / path).is_dir()
result["ignored_paths"] = run(["git","check-ignore","data/example","checkpoints/example","outputs/example",".venv/example"])
result["navigation"] = []
for link in re.findall(r"\]\(([^)]+)\)", SOURCE.read_text()):
    if "://" in link:
        continue
    path_text, _, anchor = link.partition("#")
    path = ROOT / "course" / path_text if path_text else ROOT / "course/first-steps.md"
    exists = path.exists()
    fragment = True
    excerpt = ""
    if anchor and path.suffix == ".md" and exists:
        raw = path.read_text()
        m = re.search(r"^## " + re.escape(anchor) + r"\s.*?(?=^## |\Z)",raw,re.M|re.S)
        fragment = bool(m)
        if m:
            part = m.group(0)
            excerpt = "\n".join(line for line in part.splitlines()
                                 if "tiny_perceptron" in line or "節錄自" in line or "精簡自" in line)[:1800]
    result["navigation"].append({"link":link,"exists":exists,"anchor_exists":fragment,"implementation_mentions":excerpt})
print(json.dumps(result,ensure_ascii=False,indent=2))
