"""直接核對固定公開權重；下載內容只留在記憶體，輸出小份核對紀錄。"""

import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "55", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    requests.append({"command_argv": command, "url": url, "bytes": len(raw), "sha256": digest(raw)})
    return raw


torch.set_num_threads(1)
torch.manual_seed(20261020)
requests = []
results = json.loads((ROOT / "docs/selftrained/results/v2-final-public-results.json").read_bytes())
revision = results["hf_revision"]
base = f"https://huggingface.co/{results['hf_repo']}/resolve/{revision}/selftrained/v2"
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-10/verify_public.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "safetensors": safetensors.__version__, "device": "cpu"},
    "revision": revision,
    "requests": requests,
    "models": {},
}
tree_url = f"https://huggingface.co/api/models/{results['hf_repo']}/tree/{revision}/selftrained/v2?recursive=true&expand=false"
tree = json.loads(fetch(tree_url))
record["public_release_tree"] = [{key: entry[key] for key in ("type", "path", "size") if key in entry} for entry in tree]
for architecture, count in [("moe", 5_447_107), ("dense", 2_288_067)]:
    prefix = f"{base}/{architecture}-joint"
    config_raw = fetch(f"{prefix}/model-config.json")
    manifest_raw = fetch(f"{prefix}/inference-manifest.json")
    raw = fetch(f"{prefix}/model.safetensors")
    manifest = json.loads(manifest_raw)
    assert digest(raw) == manifest["files"]["model.safetensors"]
    assert digest(config_raw) == manifest["files"]["model-config.json"]
    assert digest(raw) == results["architectures"][architecture]["safe_weights"]["sha256"]
    config = SelftrainedConfig(**json.loads(config_raw))
    model = LimitedAssistant(config).eval()
    state = load(raw)
    assert set(state) == set(model.state_dict())
    assert {tensor.dtype for tensor in state.values()} == {torch.float32}
    model.load_state_dict(state, strict=True)
    unique_count = sum(p.numel() for p in model.parameters())
    assert unique_count == count
    assert model.lm.output.weight is model.lm.embedding.weight
    assert torch.equal(state["lm.embedding.weight"], state["lm.output.weight"])
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_length])
    tensor_bytes = sum(v.numel() * v.element_size() for v in state.values())
    unique_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert len(raw) == 8 + header_length + tensor_bytes
    assert tensor_bytes - unique_bytes == 550 * 256 * 4
    detail = {
        "config": json.loads(config_raw),
        "manifest": {key: manifest[key] for key in ("schema", "files", "stage", "selected_step", "origin", "selection", "tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight") if key in manifest},
        "parameters": unique_count,
        "parameters_millions": unique_count / 1_000_000,
        "components": {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "bytes_per_parameter": sorted({p.element_size() for p in model.parameters()}),
        "unique_fp32_bytes": unique_bytes,
        "unique_fp32_mib": unique_bytes / 1024**2,
        "ideal_two_byte_bytes": unique_count * 2,
        "ideal_two_byte_mib": unique_count * 2 / 1024**2,
        "file_bytes": len(raw),
        "file_sha256": digest(raw),
        "file_mib": len(raw) / 1024**2,
        "header_json_bytes": header_length,
        "header_prefix_bytes": 8,
        "stored_tensor_count": len(state),
        "stored_tensor_bytes": tensor_bytes,
        "duplicated_embedding_bytes": tensor_bytes - unique_bytes,
        "duplicated_embedding_mib": (tensor_bytes - unique_bytes) / 1024**2,
        "embedding": header["lm.embedding.weight"],
        "output": header["lm.output.weight"],
        "duplicate_values_equal": True,
        "runtime_weights_tied": True,
    }
    with torch.no_grad():
        output = model(torch.tensor([[1, 10, 11]], dtype=torch.long))
        continuation = model(torch.tensor([[12]], dtype=torch.long), cache=output["cache"])
    detail["cache"] = {"prefix_shapes": [list(t.shape) for t in output["cache"][0]], "continuation_shapes": [list(t.shape) for t in continuation["cache"][0]], "dtype": str(output["cache"][0][0].dtype)}
    if architecture == "moe":
        ffn = model.lm.blocks[0].ffn
        pointers = [p.data_ptr() for expert in ffn.experts for p in expert.parameters()]
        with torch.no_grad():
            _, _, routing = ffn(torch.randn(1, 8, config.width))
        detail["routing"] = {"experts_allocated": len(ffn.experts), "top_k": ffn.top_k, "chosen": routing["chosen"].tolist(), "expert_storage_unchanged": pointers == [p.data_ptr() for expert in ffn.experts for p in expert.parameters()]}
        assert detail["routing"]["expert_storage_unchanged"]
    record["models"][architecture] = detail

record["milestones"] = {}
for stage in ("moe-pretrain", "moe-sft"):
    raw = fetch(f"{base}/{stage}/model.safetensors")
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_length])
    dtypes = sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})
    assert dtypes == ["F32"]
    record["milestones"][stage] = {"file_bytes": len(raw), "file_sha256": digest(raw), "dtypes": dtypes}

selection = json.loads((ROOT / "docs/selftrained/results/v2-final-training-source-selection.json").read_bytes())
record["training_selection"] = {
    "moe_loss_weights": selection["moe_joint"]["language_loss_weights"],
    "moe_origin_kind": selection["moe_joint"]["origin"]["kind"],
    "moe_selected_step": selection["moe_joint"]["selected_step"],
    "dense_selected_step": selection["dense_joint"]["selected_step"],
    "moe_completed_steps": selection["moe_joint"]["completed_requested_steps"],
    "dense_completed_steps": selection["dense_joint"]["completed_steps"],
}
record["repository_inputs"] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in [ROOT / "docs/selftrained/results/v2-final-public-results.json", ROOT / "docs/selftrained/results/v2-final-training-source-selection.json", ROOT / "tiny_perceptron/selftrained/model.py", ROOT / "tiny_perceptron/selftrained/inference.py", ROOT / "tiny_perceptron/model.py", ROOT / "tiny_perceptron/modern.py", ROOT / "scripts/selftrained/train.py"]}
record["limitations"] = "只核對固定公開權重、實作與 CPU 合成輸入；沒有測量峰值 RAM、速度、任務品質，也沒有重新訓練。"
print(json.dumps(record, ensure_ascii=False, indent=2))
