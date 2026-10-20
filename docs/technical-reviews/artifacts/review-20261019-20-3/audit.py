"""核對參數儲存算例、固定來源及保存的資源量測；不載入大模型。"""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


ROOT = Path.cwd()
DEST = Path(__file__).resolve().parent
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T063126809300Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    path = DEST / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_record(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": sha(raw), "record": json.loads(raw)}


def excerpt(text, start, end=None, limit=1800):
    position = text.index(start)
    stop = text.index(end, position + len(start)) if end else position + limit
    return text[position:stop]


def method(text, class_name, method_name):
    tree = ast.parse(text)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    function = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == method_name)
    return ast.get_source_segment(text, function)


UPSTREAM = {
    "qwen_metadata": "https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct/revision/89644892e4d85e24eaac8bacfd4f463576704203",
    "whisper_metadata": "https://huggingface.co/api/models/openai/whisper-large-v3-turbo/revision/41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "qwen_config": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json",
    "qwen_model": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
    "whisper_card": "https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md",
    "lora": "https://raw.githubusercontent.com/microsoft/LoRA/main/loralib/layers.py",
    "checkpoint": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/utils/checkpoint.py",
    "cuda_memory": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/cuda/memory.py",
    "adam": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adam.py",
    "adamw": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adamw.py",
    "autograd": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/autograd.rst",
    "python_functions": "https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/library/functions.rst",
    "python_lexical": "https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/reference/lexical_analysis.rst",
    "binary_units": "https://physics.nist.gov/cuu/Units/binary.html",
}


def fetch(item):
    name, url = item
    command = ["curl", "-fsSL", "--max-time", "40", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return name, {
        "url": url,
        "command": command,
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_of_full_response": sha(result.stdout),
        "response_bytes": len(result.stdout),
        "exit_code": result.returncode,
        "full_text": result.stdout.decode("utf-8"),
    }


with ThreadPoolExecutor(max_workers=6) as pool:
    fetched = dict(pool.map(fetch, UPSTREAM.items()))

texts = {name: item.pop("full_text") for name, item in fetched.items()}
for name in ("qwen_metadata", "whisper_metadata"):
    parsed = json.loads(texts[name])
    fetched[name]["selected_fields"] = {key: parsed[key] for key in ("id", "sha", "safetensors")}
config = json.loads(texts["qwen_config"])
fetched["qwen_config"]["selected_fields"] = {
    "architectures": config["architectures"],
    "model_type": config["model_type"],
    "text_config": {key: config["text_config"][key] for key in (
        "hidden_size", "head_dim", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "tie_word_embeddings",
    )},
}
fetched["qwen_model"]["excerpts"] = {
    "dense_mlp": method(texts["qwen_model"], "Qwen3VLTextMLP", "forward"),
    "attention_projections": method(texts["qwen_model"], "Qwen3VLTextAttention", "__init__"),
    "decoder_layer": method(texts["qwen_model"], "Qwen3VLTextDecoderLayer", "__init__"),
    "cache_update": excerpt(texts["qwen_model"], "if past_key_values is not None:", limit=490),
}
fetched["whisper_card"]["excerpts"] = {
    "purpose": excerpt(texts["whisper_card"], "Whisper is a", limit=1250),
}
fetched["lora"]["excerpts"] = {
    "initialization": method(texts["lora"], "Linear", "__init__"),
    "forward": method(texts["lora"], "Linear", "forward"),
}
fetched["checkpoint"]["excerpts"] = {
    "tradeoff": excerpt(texts["checkpoint"], "Activation checkpointing is a technique", limit=990),
}
fetched["cuda_memory"]["excerpts"] = {
    "allocated": excerpt(texts["cuda_memory"], "def memory_allocated(", "def max_memory_allocated("),
    "peak": excerpt(texts["cuda_memory"], "def max_memory_allocated(", "def memory_reserved("),
}
fetched["adam"]["excerpts"] = {
    "states": excerpt(texts["adam"], 'state["exp_avg"] =', limit=850),
    "moments": excerpt(texts["adam"], "exp_avg.lerp_", limit=850),
}
fetched["adamw"]["excerpts"] = {
    "default_configuration": method(texts["adamw"], "AdamW", "__init__"),
}
fetched["autograd"]["excerpts"] = {
    "reverse_mode": excerpt(texts["autograd"], "Autograd is a reverse", limit=1750),
}
fetched["python_functions"]["excerpts"] = {
    "round": excerpt(texts["python_functions"], ".. function:: round(", ".. _func-set:"),
}
fetched["python_lexical"]["excerpts"] = {
    "underscores": excerpt(texts["python_lexical"], "Underscores are ignored", limit=450),
}
fetched["binary_units"]["excerpts"] = {
    "gibibyte": excerpt(texts["binary_units"], "gibibyte", limit=390),
}
write("primary-sources.json", {
    "scope": "本次以 curl 讀取原始來源；只保留定位片段與完整回應 SHA-256，未保存第三方全文。",
    "sources": fetched,
})

bundle = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/20.md#20.3' --execute",
    "origin": RUN.relative_to(ROOT).as_posix(),
    "files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (RUN / name).read_bytes()
    bundle["files"][name] = {"sha256": sha(raw), "text": raw.decode("utf-8")}
write("section-execution.json", bundle)

paths = {
    "prepare": "docs/natural-assistant/evidence/v4-runtime/prepare-37212856741/model-prepare-summary.json",
    "training": "docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json",
    "training_execution": "docs/natural-assistant/evidence/v4-runtime/train-37217452291/verify-execution.json",
    "ui": "docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui/report.json",
    "disk": "docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui/disk-observation.json",
    "release": "docs/natural-assistant/v4/public-release.json",
}
records = {name: read_record(path) for name, path in paths.items()}
prepare = records["prepare"]["record"]
train = records["training"]["record"]
ui = records["ui"]["record"]
disk = records["disk"]["record"]
release = records["release"]["record"]
qwen = fetched["qwen_metadata"]["selected_fields"]["safetensors"]["total"]
whisper = fetched["whisper_metadata"]["selected_fields"]["safetensors"]["total"]
assert qwen == 2_127_532_032 and whisper == 808_878_080
snapshot_bytes = sum(file["bytes"] for snapshot in prepare["snapshots"].values() for file in snapshot["files"])
assert snapshot_bytes == prepare["total_downloaded_snapshot_bytes"] == disk["official_cache"]["unique_regular_file_bytes"] == 5_889_111_977
assert ui["command"][ui["command"].index("--dtype") + 1] == "float32"
assert ui["server_cpu_resources"]["maximum_resident_set_kib"] == 13_496_104
assert ui["model_load_counts"] == {"load_core": 1, "load_asr": 1}
assert train["resources"]["peak_cuda_memory_allocated_bytes"] == 5_719_892_480
assert records["training_execution"]["record"]["returncode"] == 0
assert release["base_model"]["revision"] == fetched["qwen_metadata"]["selected_fields"]["sha"]
assert release["asr_model"]["revision"] == fetched["whisper_metadata"]["selected_fields"]["sha"]

text_config = config["text_config"]
rank = 8
hidden = text_config["hidden_size"]
qout = text_config["num_attention_heads"] * text_config["head_dim"]
vout = text_config["num_key_value_heads"] * text_config["head_dim"]
lora_parameters = text_config["num_hidden_layers"] * rank * (hidden + qout + hidden + vout)
assert lora_parameters == train["independent_audit"]["trainable_lora_parameters"] == 1_605_632

torch.manual_seed(42)
base = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
left = torch.nn.Parameter(torch.randn(2, 4))
right = torch.nn.Parameter(torch.randn(4, 2))
optimizer = torch.optim.AdamW([left, right], lr=3e-5)
calls = {"plain": 0, "checkpointed": 0}


def block(inputs, name):
    calls[name] += 1
    return torch.tanh(inputs @ base.T + (inputs @ left.T) @ right.T)


x = torch.randn(3, 4)
block(block(x, "plain"), "plain").square().sum().backward()
plain = [parameter.grad.clone() for parameter in (left, right)]
optimizer.zero_grad(set_to_none=True)
y = checkpoint(lambda inputs: block(inputs, "checkpointed"), x, use_reentrant=False)
checkpoint(lambda inputs: block(inputs, "checkpointed"), y, use_reentrant=False).square().sum().backward()
checkpoint_matches = all(torch.allclose(parameter.grad, expected) for parameter, expected in zip((left, right), plain))
assert checkpoint_matches and base.grad is None and all(parameter.grad is not None for parameter in (left, right))
assert calls["checkpointed"] > calls["plain"]
optimizer.step()
states = {str(index): {
    key: {"shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}
    for key, value in optimizer.state[parameter].items()
} for index, parameter in enumerate((left, right))}
assert all(set(state) == {"step", "exp_avg", "exp_avg_sq"} for state in states.values())

repository_paths = [
    "tiny_perceptron/natural_assistant.py", "docs/natural-assistant/v4/TRAINING.md",
    "docs/natural-assistant/evidence/v4-runtime/train-37217452291/verify-training-lower-lr.py.source.txt",
]
repository = {}
for path in repository_paths:
    raw = (ROOT / path).read_bytes()
    text = raw.decode("utf-8")
    item = {"path": path, "sha256": sha(raw)}
    if path.endswith("natural_assistant.py"):
        item["load_core"] = excerpt(text, "def load_core(", "def synchronize(")
        item["optimizer"] = excerpt(text, "parameters = [parameter for parameter", "record = dict(")
        item["peak_start"] = excerpt(text, "if str(options.device).startswith(\"cuda\"):\n        torch.cuda.reset_peak_memory_stats", limit=130)
        item["peak_end"] = excerpt(text, 'record["peak_cuda_memory_allocated_bytes"] =', limit=185)
        item["asr"] = excerpt(text, "def load_asr(", "@torch.inference_mode()")
        item["lora_targets"] = next(line for line in text.splitlines() if line.startswith("LORA_TARGETS ="))
    elif path.endswith(".source.txt"):
        item["measured_configuration_assertion"] = next(line for line in text.splitlines() if 'assert report["device"]' in line)
        item["training_configuration_assertion"] = next(line for line in text.splitlines() if 'assert report["learning_rate"]' in line)
    else:
        item["resource_recipe"] = excerpt(text, "本章這份完整訓練使用", "## 2.")
    repository[path] = item

rounding = {
    "input_exact_binary": 0.125,
    "input_ratio": list((0.125).as_integer_ratio()),
    "round_to_two": round(0.125, 2),
    "decimal_half_up_expected": "0.13",
    "input_other_tie": 0.375,
    "round_other_tie": round(0.375, 2),
}
assert rounding["round_to_two"] == 0.12
write("verification.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-3/audit.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_hash": sha((ROOT / "docs/course-review-20261019/freeze-01/sources/20.3.md").read_bytes()),
    "freeze_hash": sha((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes()),
    "arithmetic": {
        "qwen_parameters": qwen,
        "qwen_float32_bytes": qwen * 4,
        "qwen_float32_gib": qwen * 4 / 1024**3,
        "qwen_float16_bytes": qwen * 2,
        "qwen_float16_gib": qwen * 2 / 1024**3,
        "whisper_parameters": whisper,
        "whisper_float32_gib": whisper * 4 / 1024**3,
        "whisper_float16_gib": whisper * 2 / 1024**3,
        "combined_parameters": qwen + whisper,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_bytes_per_parameter": snapshot_bytes / (qwen + whisper),
        "cpu_rss_gib": ui["server_cpu_resources"]["maximum_resident_set_kib"] / 1024**2,
        "lora_formula": f"28 * 8 * ((2048 + {qout}) + (2048 + {vout}))",
        "lora_parameters": lora_parameters,
        "gradient_m_v_float32_bytes": lora_parameters * 4 * 3,
        "gradient_m_v_float32_gib": lora_parameters * 4 * 3 / 1024**3,
        "peak_cuda_allocated_gib": train["resources"]["peak_cuda_memory_allocated_bytes"] / 1024**3,
        "torch_element_sizes": {str(dtype): torch.empty((), dtype=dtype).element_size() for dtype in (torch.float32, torch.float16, torch.bfloat16)},
        "literal_underscores_ignored": 2_127_532_032 == 2127532032,
        "unpacking": [[label, value] for label, value in [("32-bit", 4), ("16-bit", 2)]],
        "gib_unit": 1024**3,
    },
    "rounding": rounding,
    "toy_autograd_and_adamw": {
        "calls": calls, "checkpoint_gradients_match": checkpoint_matches,
        "frozen_base_gradient": None if base.grad is None else "present",
        "adapter_gradients_present": [parameter.grad is not None for parameter in (left, right)],
        "optimizer_states": states,
        "scope": "只核對小型 CPU 運算圖中的凍結權重、可訓修正、重算與更新器狀態；不是大型模型記憶體或品質實測。",
    },
    "records": {name: {"path": item["path"], "sha256": item["sha256"]} for name, item in records.items()},
    "selected_record_fields": {
        "prepare": {"snapshots": prepare["snapshots"], "total_downloaded_snapshot_bytes": prepare["total_downloaded_snapshot_bytes"]},
        "training": {"independent_audit": {key: train["independent_audit"][key] for key in (
            "source_revision", "train_run_id", "completed_steps", "trained_rows", "supervised_tokens", "trainable_lora_parameters", "peak_cuda_memory_allocated_bytes", "report_sha256"
        )}, "resources": train["resources"]},
        "training_execution": records["training_execution"]["record"],
        "ui": {key: ui[key] for key in ("command", "server_cpu_resources", "model_load_counts", "generation_stop_conditions", "status")},
        "disk": disk["official_cache"],
        "release": {key: release[key] for key in ("base_model", "asr_model", "dependency_versions", "runtime")},
    },
    "repository_excerpts": repository,
    "result": "全部算式與紀錄一致；round(0.125, 2) 的結果為 0.12，推翻一律五入的 API 描述。",
    "limits": "未重跑 Linux CPU 大模型服務或 L4 訓練；其峰值是核對 repo 保存的量測、配置與成功驗證收據。未讀 outputs 中的歷史訓練原檔，也未重取私有權重。",
})
print(json.dumps({"result": "completed", "qwen_parameters": qwen, "whisper_parameters": whisper, "rounding": rounding, "toy_calls": calls}, ensure_ascii=False))
