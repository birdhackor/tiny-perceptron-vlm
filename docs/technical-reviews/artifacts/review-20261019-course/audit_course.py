"""唯讀核對入口頁的連結、教材對應、短算例、流程及原始來源。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch

from scripts.build_course import notebook
from tiny_perceptron.lesson_products.inference import auto_solve, rule_depth


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-course/audit_course.py",
    "executed_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
}
frozen_path = "docs/course-review-20261019/freeze-01/sources/course.md"
body = (ROOT / frozen_path).read_text()
result["source"] = {
    "path": frozen_path,
    "sha256": digest(frozen_path),
    "current_source_equal": (ROOT / frozen_path).read_bytes() == (ROOT / "course/README.md").read_bytes(),
    "figures": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
}
links = []
for label, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", body):
    path_text, _, fragment = target.partition("#")
    path = ((ROOT / "course" / path_text) if path_text else ROOT / "course/README.md").resolve()
    text = path.read_text() if path.is_file() else ""
    anchors = re.findall(r"^## ([^ ]+) ", text, re.M)
    links.append({"label": label, "target": target, "exists": path.is_file(), "anchor_exists": not fragment or fragment in anchors})
result["links"] = links
result["links_all_valid"] = all(x["exists"] and x["anchor_exists"] for x in links)

chapters = []
for path in sorted((ROOT / "course/chapters").glob("*.md")):
    raw = path.read_text()
    chapters.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "title": raw.splitlines()[0], "headings": re.findall(r"^## .+$", raw, re.M)})
result["chapter_contents"] = chapters
index = read_json("course/lesson-index.json")
notebooks = []
for item in index:
    source = ROOT / item["source"]
    raw = source.read_text()
    matches = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", raw, re.M))
    pos = next(i for i, m in enumerate(matches) if m[1] == item["id"])
    section = raw[matches[pos].end():matches[pos + 1].start() if pos + 1 < len(matches) else len(raw)]
    generated = notebook(item["id"], item["title"], section, source)
    saved = read_json(item["notebook"])
    notebooks.append({"id": item["id"], "path": item["notebook"], "same_as_generated": generated == saved,
                      "listed_in_lessons": f"../{item['notebook']}" in (ROOT / "course/lessons.md").read_text()})
result["notebooks"] = {"count": len(notebooks), "all_source_synchronized": all(x["same_as_generated"] for x in notebooks),
                       "all_listed_in_lessons": all(x["listed_in_lessons"] for x in notebooks),
                       "mismatches": [x for x in notebooks if not x["same_as_generated"] or not x["listed_in_lessons"]]}

namespace = {}
first = (ROOT / "course/chapters/01.md").read_text()
code = re.search(r"```python\n(.*?)```", first, re.S)[1]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code, "course/chapters/01.md", "exec"), namespace)
result["character_roundtrip"] = {"input": namespace["text"], "table": namespace["to_id"], "ids": namespace["ids"], "restored": namespace["restored"], "stdout": output.getvalue(), "equal": namespace["text"] == namespace["restored"]}
torch.set_num_threads(1)
recipes = torch.tensor([[2., 1., 0.], [1., 2., 1.]])
weights = torch.tensor([[10., 0.], [20., 1.], [30., 4.]])
result["weighted_sum"] = {"input": recipes.tolist(), "weights": weights.tolist(), "observed": (recipes @ weights).tolist(), "expected": [[40., 1.], [80., 6.]]}
torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Embedding(5, 3), torch.nn.Linear(3, 5))
x, y = torch.tensor([3]), torch.tensor([2])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = torch.nn.functional.cross_entropy(model(x), y)
before.backward()
optimizer.step()
after = torch.nn.functional.cross_entropy(model(x), y)
result["torch_update"] = {"loss_before": before.item(), "loss_after": after.item(), "finite_gradients": all(torch.isfinite(p.grad).all().item() for p in model.parameters()), "scope": "單次固定 CPU 算例，只核對 API 與資料流。"}


class Solver:
    def __init__(self):
        self.auto_policy = {"version": 1, "low_max_events": 0, "medium_max_events": 4, "sum_depth": 1}
        self.calls = []

    def synchronize(self):
        pass

    def solve(self, state, loops):
        self.calls.append({"state": state, "loops": loops})
        return {"loops": loops}


class Judge:
    def __init__(self, choice, info):
        self.choice, self.info, self.calls = choice, info, []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.choice if question == "choice" else self.info, "seconds": 0.0}


branches = []
for choice, info, effort, cap in [("card", "yes", "auto", 6), ("boxes", "no", "auto", 6), ("boxes", "yes", "auto", 6), ("boxes", "yes", "medium", 2)]:
    b, c = Solver(), Judge(choice, info)
    actual = auto_solve(b, c, "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", effort=effort, max_loops=cap)
    branches.append({"inputs": {"choice": choice, "info": info, "effort": effort, "cap": cap}, "judge_calls": c.calls, "solver_calls": b.calls, "result": actual})
result["auto_flow"] = {"branches": branches, "scope": "替身只隔離流程控制；沒有執行成品神經推論或量測品質。", "rule_depths": [rule_depth(Solver().auto_policy, "boxes", i) for i in range(1, 7)]}

training = []
for name in ["initial-b-execution.json", "initial-c-execution.json"]:
    path = "docs/lesson-products/results/" + name
    d = read_json(path)
    t = d["training"]
    training.append({"path": path, "sha256": digest(path), "code_revision": d["code_revision"], "status": d["status"], "job": d["job"], "training": {k: t.get(k) for k in ["random_initialization", "weights_changed", "initial_weights_sha256", "selected_weights_sha256", "steps_this_attempt", "seed", "data_manifest_sha256", "environment"]}})
result["product_training"] = training
moe = []
for path in sorted((ROOT / "docs/selftrained/results/training-raw").glob("moe-*/raw/train-receipt.json")):
    d = json.loads(path.read_bytes())
    moe.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "origin_kind": d.get("origin", {}).get("kind"), "seed": d.get("origin", {}).get("seed"), "steps": d.get("steps"), "stage": d.get("stage"), "history": [{k: row.get(k) for k in ["stage", "step", "checkpoint_sha256"]} for row in d.get("stage_history", [])]})
result["moe_training"] = moe
final = read_json("docs/lesson-products/results/final-report-v1.json")
result["product_final"] = {"path": "docs/lesson-products/results/final-report-v1.json", "sha256": digest("docs/lesson-products/results/final-report-v1.json"), "gates": final["gates"], "b": final["b"]["3"], "c": final["c"], "c_wording": final["shifts"]["c_wording"], "scope": final["scope"]}
moe_metrics = read_json("docs/selftrained/results/public-raw/moe/test/metrics.json")
result["moe_final"] = {"path": "docs/selftrained/results/public-raw/moe/test/metrics.json", "sha256": digest("docs/selftrained/results/public-raw/moe/test/metrics.json"), **{k:moe_metrics[k] for k in ["split", "count", "expected_count", "evaluation_complete", "thresholds", "per_task_final_reply", "notes"]}}

sources = {
    "phone": ("https://support.apple.com/zh-tw/104995", "顯示你接下來可能會輸入的字詞選項"),
    "autograd": ("https://docs.pytorch.org/docs/2.14/notes/autograd.html", "By tracing this graph from roots to leaves"),
    "embedding": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html", "A simple lookup table"),
    "linear": ("https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "Applies an affine linear transformation"),
    "validation": ("https://scikit-learn.org/1.7/modules/cross_validation.html", "training proceeds on the training set"),
    "notebook": ("https://jupyter.org/", "They contain a complete record"),
    "colab": ("https://research.google.com/colaboratory/faq.html", "Code is executed in a virtual machine private to your account"),
    "colab_signin": ("https://colab.research.google.com/notebooks/welcome.ipynb", "Sign in"),
}


def fetch(item):
    key, (url, phrase) = item
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url]
    r = subprocess.run(command, capture_output=True)
    raw = r.stdout.decode("utf-8", "replace")
    cleaned = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", " ", cleaned)))
    position = text.lower().find(phrase.lower())
    title = re.search(r"<title>(.*?)</title>", raw, re.S)
    excerpt = phrase if key == "phone" else " ".join(text[position:].split()[:24])
    return key, {"url": url, "command": command, "returncode": r.returncode, "stderr": r.stderr.decode(), "full_response_sha256": hashlib.sha256(r.stdout).hexdigest(), "title": unescape(title[1]).strip() if title else "", "phrase_found": position >= 0, "excerpt": excerpt if position >= 0 else "", "accessed_on": datetime.now(timezone.utc).date().isoformat()}


with ThreadPoolExecutor(max_workers=7) as executor:
    result["external_originals"] = dict(executor.map(fetch, sources.items()))
result["scope"] = "僅讀取本頁、連結教材、模型程式、原始實驗收據及官方來源；未讀任何舊教材審閱結論，未讀本機 outputs。未重訓 GPU 成品，未使用 Colab 帳號執行。"
print(json.dumps(result, ensure_ascii=False, indent=2))
