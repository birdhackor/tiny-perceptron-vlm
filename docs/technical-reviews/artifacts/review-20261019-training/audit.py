"""保存針對裝置敘述、固定實驗及公開成品的補充核對。"""

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import torch
from tiny_perceptron.training import choose_device
from tiny_perceptron.data import ByteTokenizer

out = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-training/audit.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}, "runs": [], "code": {}, "experiments": {}, "public": [], "tool_trace": []}


def run(args):
    completed = subprocess.run(args, capture_output=True, text=True, timeout=45, env=os.environ)
    entry = {"command": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    out["runs"].append(entry)
    return entry


for filename in ("scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"):
    run([sys.executable, filename, "--device", "cpu"])
out["device_priority"] = []
for cuda, mps in ((True, True), (False, True), (False, False)):
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        out["device_priority"].append({"cuda_available": cuda, "mps_available": mps, "auto": str(choose_device())})

functions = {
    "tiny_perceptron/training.py": ["choose_device", "save_checkpoint", "load_checkpoint"],
    "scripts/quantize.py": ["main"],
    "scripts/prepare_data.py": ["main"],
    "scripts/course_experiments/run.py": ["main", "execute"],
    "scripts/course_experiments/common.py": ["Context"],
    "scripts/course_experiments/architecture.py": ["_benchmark", "run_modern", "run_moe", "run_flash_probe"],
    "scripts/course_experiments/compression.py": ["_dataset", "_fit_text", "run_distillation", "run_multimodal_distillation"],
    "scripts/course_experiments/modalities.py": ["_freeze", "run_joint"],
    "scripts/course_experiments/applications.py": ["_sample"],
}
for filename, identifiers in functions.items():
    raw = Path(filename).read_bytes()
    text = raw.decode()
    lines = text.splitlines()
    tree = ast.parse(text)
    out["code"][filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "identifiers": {node.name: "\n".join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in identifiers}}

times = {"simple_models": 4.3, "text_foundation": 16.0, "real_text": 23.9, "tokenizer": 16.2, "sft": 31.7, "sft_ablation": 23.5, "style": 29.5, "safety": 22.2, "lora": 24.5, "encoders": 10.2, "projector": 15.9, "vqa": 83.8, "dpo": 18.7, "posttraining": 2.1, "modern": 44.6, "moe": 65.7, "efficiency": 32.3, "precision": 15.0, "flash_probe": 4.5, "quantization": 7.8, "distillation": 90.1, "joint": 33.4, "multimodal_distillation": 53.0}
for name, expected in times.items():
    path = Path("docs/course-experiments/results") / f"{name}.json"
    data = json.loads(path.read_bytes())
    result = data["results"]
    entry = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "expected_seconds": expected, "elapsed_seconds": data["elapsed_seconds"], "rounded_seconds": round(data["elapsed_seconds"], 1), "matches": round(data["elapsed_seconds"], 1) == expected, "device": data["device"], "gpu": data.get("gpu"), "seed": data["seed"], "evidence_status": data["evidence_status"]}
    if name in ("modern", "moe"):
        entry["dataset"] = result["dataset"]
        entry["variants"] = {k: {"config": v["model"]["config"], "steps": v["training"]["steps"], "heldout": {s: {key: val for key, val in stats.items() if key in ("nll", "nll_sum", "effective_tokens", "examples")} for s, stats in v["heldout"].items()}} for k, v in result["variants"].items()}
    if name in ("distillation", "multimodal_distillation"):
        entry["tasks"] = {k: {"data": v["data"], "runs": {key: {"training": r.get("training", {}), "examples": {split: r.get(split, {}).get("examples") for split in ("validation", "test")}} for key, r in v["runs"].items()}} for k, v in result["tasks"].items()}
    if name == "flash_probe":
        entry["probe"] = {k: result[k] for k in ("status", "schedule_completed", "verification_passed", "supported_routes", "configuration")}
        entry["routes"] = {k: {key: val for key, val in v.items() if key in ("status", "backend_verification", "reason")} for k, v in result["routes"].items()}
    out["experiments"][name] = entry

raw = Path("docs/course-experiments/results/tools.json").read_bytes()
data = json.loads(raw)["results"]
out["tools_source_sha256"] = hashlib.sha256(raw).hexdigest()
tok = ByteTokenizer()


def visit(value, path="results"):
    if isinstance(value, dict):
        generation = value.get("generation")
        if isinstance(generation, dict) and "input_ids" in generation and "messages" in generation:
            messages = generation["messages"]
            ids = [tok.bos_id]
            for message in messages:
                ids += [{"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
            ids.append(tok.assistant_id)
            actual = generation["input_ids"]
            if actual and isinstance(actual[0], list):
                actual = actual[0]
            mismatch = ids != actual
            if mismatch and len(out["tool_trace"]) < 3:
                out["tool_trace"].append({"path": path, "input_ids": actual, "decoded_input": tok.decode(actual), "messages_length": len(messages), "messages_rendered_length": len(ids), "input_length": len(actual), "mismatch": mismatch})
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                visit(item, path + "." + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                visit(item, path + f"[{index}]")


visit(data)
for name, url in (
    ("models", "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/revision/2c6a1062f3e66a602cf83eda6c703456ec18bca0"),
    ("product_b", "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/b?recursive=true"),
    ("product_c", "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/c?recursive=true"),
    ("moe_v2", "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint?recursive=false"),
):
    entry = run(["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url])
    response = json.loads(entry["stdout"]) if entry["returncode"] == 0 else None
    out["public"].append({"name": name, "url": url, "response_sha256": hashlib.sha256(entry["stdout"].encode()).hexdigest(), "returncode": entry["returncode"], "summary": {k: response.get(k) for k in ("id", "sha", "private")} if isinstance(response, dict) else response})
    entry["stdout"] = "已解析回應；其摘要、檔案列表與全文 SHA-256 保存於 public。"

out["scope_note"] = "未讀其他審閱報告或處置內容；git status 與檔案清單曾顯示其他報告路徑名稱。首次 uv run 版本查詢自動重裝專案套件，之後一律直接執行 .venv/bin/python。"
out["initial_version_command_receipt"] = {"command": "PYTHONDONTWRITEBYTECODE=1 uv run python -c 'import sys,torch,platform; print(sys.version); print(torch.__version__); print(platform.platform()); print(\"cuda\",torch.cuda.is_available(),\"mps\",torch.backends.mps.is_available())'", "stderr": "Building tiny-perceptron-vlm @ file:///Users/yclin/project/tiny-perceptron-vlm\nBuilt tiny-perceptron-vlm @ file:///Users/yclin/project/tiny-perceptron-vlm\nUninstalled 1 package in 1ms\nInstalled 1 package in 1ms", "stdout": "3.13.16 (main, Oct 2 2026, 08:32:25) [Clang 21.0.0 (clang-2100.1.1.101)]\n2.14.1\nmacOS-26.4.1-arm64-arm-64bit-Mach-O\ncuda False mps False"}
(BASE / "audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"all_times_match": all(e["matches"] for e in out["experiments"].values()), "public_status": {e["name"]: e["returncode"] for e in out["public"]}, "tool_input_mismatch_examples": len(out["tool_trace"]), "device_flag_returncodes": [e["returncode"] for e in out["runs"][:3]]}, ensure_ascii=False))
