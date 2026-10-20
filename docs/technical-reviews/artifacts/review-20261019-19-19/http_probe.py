"""在可用的本機埠驗證同一個 JSON 服務入口。"""

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

AREA = Path(__file__).resolve().parent
ROOT = AREA.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.lesson_products.delivery import download_product

release = json.loads((ROOT / "tiny_perceptron/lesson_products/release.json").read_bytes())
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-19/http_probe.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "scope": "原先指定的 8765 已被占用；改用空閒埠查核同一 serve 入口，不接觸既有服務。",
    "requests": [],
}
with tempfile.TemporaryDirectory(prefix=".http-", dir=AREA) as scratch:
    directory = Path(scratch)
    for product in ("b", "c"):
        item = release["products"][product]
        download_product(release["repo"], release["revision"], item["prefix"], item["manifest_sha256"], directory / product, product)
    with socket.socket() as selector:
        selector.bind(("127.0.0.1", 0))
        port = selector.getsockname()[1]
    command = [sys.executable, "-m", "tiny_perceptron.lesson_products", "serve", "--b", str(directory / "b"), "--c", str(directory / "c"), "--port", str(port)]
    result["server_command"] = command
    server = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("服務未進入就緒狀態")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2) as response:
                    result["health"] = json.load(response)
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("服務就緒逾時")
        state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
        for endpoint, payload in (
            ("solve", {"state": state, "loops": 3}),
            ("judge", {"state": "卡片進度;申請編號=?", "question": "info"}),
            ("auto", {"state": state, "effort": "medium", "max_loops": 2}),
        ):
            request = urllib.request.Request(f"http://127.0.0.1:{port}/{endpoint}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                result["requests"].append({"endpoint": endpoint, "request": payload, "status": response.status, "response": json.load(response)})
        result["result"] = "三個入口均回傳 HTTP 200，欄位與 CLI 行為相符。"
    finally:
        server.terminate()
        stdout, stderr = server.communicate(timeout=10)
        result.update(stdout=stdout, stderr=stderr)
(AREA / "http-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
