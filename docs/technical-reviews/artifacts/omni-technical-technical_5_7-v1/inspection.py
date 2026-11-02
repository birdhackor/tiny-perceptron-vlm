"""保存親讀來源、必要前提切片與歷史實驗版本的精確收據。"""

import ast
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-technical_5_7-v1")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def function(raw, name):
    text = raw.decode()
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


receipt = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_5_7", "cache_root": str(CACHE), "prerequisites": [], "official_sources": [], "historical_sources": [], "historical_function_comparisons": []}
for path, ids in [("course/chapters/05.md", ["5.4", "5.6"]), ("course/first-steps.md", ["W.7"]), ("course/training.md", ["T.4"])]:
    raw = (ROOT / path).read_bytes()
    for lesson in ids:
        match = re.search(rb"^## " + re.escape(lesson.encode()) + rb"\s.*?(?=^## |\Z)", raw, re.M | re.S)
        receipt["prerequisites"].append({"source": path + "#" + lesson, "sha256": digest(match.group()), "bytes": len(match.group()), "inspection_scope": "實讀當前完整小節；僅用於本節需要的 Adam 狀態、總計畫、路徑或固定實驗入口，不採整章依賴。"})

pin = torch.version.git_version
mapping = {
    "pytorch-adam.py": "torch/optim/adam.py",
    "pytorch-torch_optim_adamw.py": "torch/optim/adamw.py",
    "pytorch-torch_optim_optimizer.py": "torch/optim/optimizer.py",
    "pytorch-torch_serialization.py": "torch/serialization.py",
    "pytorch-torch_nn_modules_module.py": "torch/nn/modules/module.py",
    "pytorch-torch_random.py": "torch/random.py",
    "pytorch-docs_source_notes_serialization.md": "docs/source/notes/serialization.md",
    "pytorch-docs_source_notes_randomness.md": "docs/source/notes/randomness.md",
}
for filename, path in mapping.items():
    raw = (CACHE / filename).read_bytes()
    local = Path(torch.__file__).parent.parent / path
    receipt["official_sources"].append({"url": f"https://github.com/pytorch/pytorch/blob/{pin}/{path}", "version": f"torch {torch.__version__}; git {pin}", "cache": str(CACHE / filename), "sha256": digest(raw), "matches_installed": local.read_bytes() == raw if local.exists() else "documentation not shipped in wheel"})
for name in ["tempfile", "os", "pathlib"]:
    path = CACHE / f"python-{name}.html"
    receipt["official_sources"].append({"url": f"https://docs.python.org/3.13/library/{name}.html", "version": "Python 3.13.18 documentation, read 2026-10-29; executed Python 3.13.5", "cache": str(path), "sha256": digest(path.read_bytes())})

with urllib.request.urlopen("https://api.github.com/repos/pytorch/tutorials/commits/main", timeout=25) as response:
    tutorial_pin = json.load(response)["sha"]
tutorial_url = f"https://raw.githubusercontent.com/pytorch/tutorials/{tutorial_pin}/beginner_source/saving_loading_models.py"
with urllib.request.urlopen(tutorial_url, timeout=25) as response:
    tutorial = response.read()
assert tutorial == (CACHE / "pytorch-save-load.py").read_bytes()
receipt["official_sources"].append({"url": tutorial_url, "version": tutorial_pin, "cache": str(CACHE / "pytorch-save-load.py"), "sha256": digest(tutorial), "matches_previously_read_main_bytes": True})

record = json.loads((ROOT / "docs/course-experiments/results/text_foundation.json").read_bytes())
for path, names in [("scripts/course_experiments/text.py", ["_resume_probe", "_steps"]), ("scripts/course_experiments/common.py", ["seed", "new_lm", "text_examples"]), ("scripts/course_experiments/run.py", [])]:
    historical = (CACHE / ("historical-" + path.replace("/", "_"))).read_bytes()
    assert digest(historical) == record["code_sha256"][path]
    receipt["historical_sources"].append({"url": f"https://github.com/birdhackor/tiny-perceptron-vlm/blob/{record['revision']}/{path}", "retrieval": f"git show {record['revision']}:{path}", "cache": str(CACHE / ("historical-" + path.replace("/", "_"))), "sha256": digest(historical), "matches_experiment_record": True})
    for name in names:
        old, current = function(historical, name), function((ROOT / path).read_bytes(), name)
        receipt["historical_function_comparisons"].append({"source": path, "function": name, "identical": old == current, "historical_function_sha256": digest(old.encode()), "current_function_sha256": digest(current.encode())})

help_result = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=25)
assert help_result.returncode == 0 and "--experiment" in help_result.stdout and "--device {cpu,cuda}" in help_result.stdout
receipt["recipe_cli_help"] = {"command": ".venv/bin/python -m scripts.course_experiments.run --help", "returncode": help_result.returncode, "experiment_flag_present": True, "cpu_cuda_device_choices_present": True, "gpu_recipe_executed": False}
receipt["network_fallbacks"] = ["docs.pytorch.org rendered docs returned HTTP 403; read official raw GitHub Markdown at the installed wheel's git commit instead.", "The .rst notes URLs returned 404; the commit uses .md files, which were fetched and read."]
receipt["figure_inspection"] = {"freeze_figures": {}, "inline_images": re.findall(r"!\[[^\]]*\]\([^)]*\)", (ROOT / "docs/course-review-20261019/freeze-01/sources/5.7.md").read_text()), "rendered_or_viewed": False, "reason": "5.7 本身沒有圖片；前提 5.6 的圖未作為 5.7 證據，排程依函式及正文核對。"}
receipt["security_excerpt"] = {"source": f"https://github.com/pytorch/pytorch/blob/{pin}/docs/source/notes/serialization.md#weights_only-security", "lines": "264-278", "quote": "weights_only=True narrows the surface of remote code execution attacks", "limitation": "We try to prevent memory corruptions during torch.load(weights_only=True) but they might still be possible."}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
