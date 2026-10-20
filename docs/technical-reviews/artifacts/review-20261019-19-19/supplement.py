"""查核 CLI、Git 保存範圍與有限語法內的失敗例；不啟動正式訓練。"""

import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

AREA = Path(__file__).resolve().parent
ROOT = AREA.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.lesson_products.delivery import download_product
from tiny_perceptron.lesson_products.inference import Product

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-19/supplement.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "commands": [],
}
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
for args in (
    ["-m", "tiny_perceptron.lesson_products", "judge", "--help"],
    ["-m", "tiny_perceptron.lesson_products", "solve", "--help"],
    ["-m", "tiny_perceptron.lesson_products", "auto", "--help"],
    ["-m", "scripts.lesson_products.train_local", "--help"],
    ["scripts/selftrained/chat.py", "--help"],
):
    command = [sys.executable, *args]
    record = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    result["commands"].append({"command": command, "exit_code": record.returncode, "stdout": record.stdout, "stderr": record.stderr})

paths = [
    "tiny_perceptron/lesson_products/release.json", "tiny_perceptron/lesson_products/__main__.py",
    "tiny_perceptron/lesson_products/delivery.py", "tiny_perceptron/lesson_products/inference.py",
    "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/data.py",
    "tiny_perceptron/lesson_products/train.py", "tiny_perceptron/lesson_products/training.py",
    "tiny_perceptron/lesson_products/evaluation.py", "scripts/lesson_products/train_local.py",
    "scripts/selftrained/train.py", "tiny_perceptron/selftrained/inference.py",
    "scripts/selftrained/chat.py", "docs/lesson-products/README.md", ".gitignore", "course/first-steps.md",
]
result["source_files"] = [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for path in paths]
command = ["git", "ls-files", "--", *paths]
record = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
result["git_tracking"] = {"command": command, "exit_code": record.returncode, "stdout": record.stdout}
result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
with tempfile.TemporaryDirectory(prefix=".supplement-", dir=AREA) as scratch:
    directory = Path(scratch)
    command = [sys.executable, "-m", "scripts.lesson_products.train_local", "--job", "docs/lesson-products/jobs/train-b-v1.json", "--output", str(directory), "--device", "cpu"]
    record = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    result["existing_output_rejection"] = {"command": command, "exit_code": record.returncode, "stdout": record.stdout, "stderr": record.stderr, "directory_still_empty": not any(directory.iterdir())}
    release = json.loads((ROOT / "tiny_perceptron/lesson_products/release.json").read_bytes())
    item = release["products"]["b"]
    download_product(release["repo"], release["revision"], item["prefix"], item["manifest_sha256"], directory / "b", "b")
    torch.set_num_threads(2)
    b = Product(directory / "b", "b")
    result["finite_examples"] = []
    for x, y, z in ((0, 0, 0), (1, 1, 1), (1, 2, 3), (2, 3, 4), (6, 6, 6)):
        state = f"三數加法;x={x};y={y};z={z}"
        observed = b.solve(state, loops=3)
        result["finite_examples"].append({"state": state, "gold": x + y + z, "observed": observed, "correct": observed["answer"] == x + y + z})
    result["finite_examples_scope"] = "事先指定的五個三數加法診斷題，每題一次、固定三輪、CPU、固定公開權重；不是獨立測試集成績或成功率估計。"
    attempted = 0
    witness = None
    for x, y, z in itertools.product(range(7), repeat=3):
        state = f"三數加法;x={x};y={y};z={z}"
        observed = b.solve(state, loops=3)
        attempted += 1
        if observed["answer"] != x + y + z:
            witness = {"state": state, "gold": x + y + z, "observed": observed}
            break
    result["failure_witness"] = {
        "attempted": attempted, "witness": witness,
        "selection": "按 x、y、z 字典序列舉 0–6 的完整三數輸入，遇第一個錯誤即停；此搜尋只建立失敗見證，不估計正確率。",
    }
(AREA / "supplement-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "完成短程式核對", "finite_examples": result["finite_examples"]}, ensure_ascii=False, indent=2))
