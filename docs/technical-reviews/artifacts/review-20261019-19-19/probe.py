"""以短時間 CPU 操作查核公開成品；暫存權重只留在本次證據目錄內。"""

import hashlib
import inspect
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

ROOT = Path(__file__).resolve().parents[4]
AREA = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.lesson_products.data import answer, judgment
from tiny_perceptron.lesson_products.inference import Product, auto_solve, parse_state
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer
from tiny_perceptron.lesson_products.training import checkpoint, restore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(arguments, offline=False):
    if offline:
        wrapper = (
            "import socket,sys,runpy; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled')); "
            "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled')); "
            "sys.argv=['tiny_perceptron.lesson_products']+sys.argv[1:]; "
            "runpy.run_module('tiny_perceptron.lesson_products',run_name='__main__')"
        )
        command = [sys.executable, "-c", wrapper, *arguments]
    else:
        command = [sys.executable, "-m", "tiny_perceptron.lesson_products", *arguments]
    value = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=180)
    record = {
        "command": command,
        "network_disabled": offline,
        "exit_code": value.returncode,
        "stdout": value.stdout,
        "stderr": value.stderr,
    }
    if value.returncode == 0:
        record["parsed"] = json.loads(value.stdout)
    return record


ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
torch.set_num_threads(2)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-19/probe.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"},
    "freeze_source_sha256": digest(ROOT / "docs/course-review-20261019/freeze-01/sources/19.19.md"),
    "freeze_manifest_sha256": digest(ROOT / "docs/course-review-20261019/freeze-04.json"),
    "downloads": [],
    "offline_commands": [],
}

with tempfile.TemporaryDirectory(prefix=".probe-", dir=AREA) as scratch:
    work = Path(scratch)
    for product in ("b", "c"):
        record = run(["download", "--product", product, "--destination", str(work / product)])
        result["downloads"].append(record)
        print(json.dumps({"download": product, "exit_code": record["exit_code"]}), flush=True)
        if record["exit_code"]:
            result["result"] = "download failed; no claim of successful inference"
            (AREA / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            raise SystemExit(1)

    state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
    commands = [
        ["solve", "--b", str(work / "b"), "--state", state, "--loops", "3"],
        ["judge", "--c", str(work / "c"), "--state", "卡片進度;申請編號=?", "--question", "info"],
        ["auto", "--b", str(work / "b"), "--c", str(work / "c"), "--state", state, "--effort", "auto"],
        ["auto", "--b", str(work / "b"), "--c", str(work / "c"), "--state", state, "--effort", "medium", "--max-loops", "2"],
    ]
    result["offline_commands"] = [run(command, offline=True) for command in commands]
    b, c = Product(work / "b", "b"), Product(work / "c", "c")
    result["models"] = {}
    for product, instance in (("b", b), ("c", c)):
        directory = work / product
        weights = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        receipt = json.loads((directory / "training-receipt.json").read_bytes())
        result["models"][product] = {
            "all_modules_eval": all(not module.training for module in instance.model.modules()),
            "weights_container": type(weights).__name__,
            "all_weight_values_tensor": all(isinstance(value, torch.Tensor) for value in weights.values()),
            "weight_entries": len(weights),
            "optimizer_or_rng_keys": [key for key in weights if any(word in key.lower() for word in ("optimizer", "rng"))],
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in directory.iterdir()},
            "training_receipt": {key: receipt.get(key) for key in ("random_initialization", "seed", "last_step", "selected_step", "data_manifest_sha256", "environment", "resume_verification")},
        }
    result["typed_questions"] = {question: c.judge("卡片進度;申請編號=?", question) for question in ("choice", "info", "rubric")}
    result["gold_examples"] = {
        "swap_answer": answer({"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0)], "query": "B"}),
        "missing_application_info": judgment({"task": "card", "application": "123456"}, ("application",), "info", ("yes", "no")),
    }
    result["invalid_state"] = {}
    try:
        parse_state("請跟我聊聊天")
    except ValueError as error:
        result["invalid_state"]["rejected"] = str(error)

    config_path = work / "b" / "config.json"
    raw = config_path.read_bytes()
    config_path.write_bytes(raw + b" ")
    try:
        Product(work / "b", "b")
    except ValueError as error:
        result["tamper_detection"] = str(error)
    finally:
        config_path.write_bytes(raw)

    server_command = [sys.executable, "-m", "tiny_perceptron.lesson_products", "serve", "--b", str(work / "b"), "--c", str(work / "c"), "--port", "8765"]
    server = subprocess.Popen(server_command, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result["http"] = {"command": server_command, "requests": []}
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("HTTP server exited before readiness")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=0.2) as response:
                    result["http"]["health"] = json.load(response)
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("HTTP readiness timeout")
        for endpoint, payload in (
            ("solve", {"state": state, "loops": 3}),
            ("judge", {"state": "卡片進度;申請編號=?", "question": "info"}),
            ("auto", {"state": state, "effort": "medium", "max_loops": 2}),
        ):
            request = urllib.request.Request("http://127.0.0.1:8765/" + endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                result["http"]["requests"].append({"endpoint": endpoint, "status": response.status, "response": json.load(response)})
    except Exception as error:
        result["http"]["error"] = repr(error)
    finally:
        server.terminate()
        stdout, stderr = server.communicate(timeout=10)
        result["http"].update(stdout=stdout, stderr=stderr)

    model = LoopedSolver(Config(len(Tokenizer()), width=16, heads=2, ff=32))
    optimizer = torch.optim.AdamW(model.parameters())
    generator = torch.Generator().manual_seed(7)
    checkpoint(work / "checkpoint.pt", model, optimizer, generator, {"step": 0})
    stored = torch.load(work / "checkpoint.pt", weights_only=True)
    restored, restored_optimizer, restored_generator, metadata = restore(work / "checkpoint.pt", "b", torch.device("cpu"))
    result["checkpoint_roundtrip"] = {
        "keys": sorted(stored),
        "model_equal": all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()),
        "sampler_rng_equal": torch.equal(generator.get_state(), restored_generator.get_state()),
        "optimizer_restored": restored_optimizer.state_dict() == optimizer.state_dict(),
        "metadata": metadata,
        "scope": "未做訓練更新；只查核保存與還原所需欄位，不宣稱完整正式續訓重現。",
    }

torch.manual_seed(42)
dropout = torch.nn.Dropout(0.9).eval()
result["eval_and_sampling"] = {
    "dropout_eval_identity": torch.equal(dropout(torch.ones(32)), torch.ones(32)),
    "sampled_classes": sorted(set(torch.multinomial(torch.tensor([0.5, 0.5]), 32, replacement=True).tolist())),
    "scope": "評估模式不會停用顯式抽樣；32 次抽樣只展示這個具體例子。",
}
result["installed_pytorch_contract"] = {
    "torch_load_weights_only": inspect.getdoc(torch.load).split("weights_only:", 1)[1].split("mmap:", 1)[0].strip(),
    "module_eval": inspect.getdoc(torch.nn.Module.eval),
    "torch_version": torch.__version__,
}
result["result"] = "completed"
(AREA / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "output": str(AREA / "verification.json")}), flush=True)
