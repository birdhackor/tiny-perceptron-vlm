from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import platform
import struct
import torch
import transformers
import peft
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, WhisperConfig, WhisperForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

root = Path(__file__).resolve().parents[3]
sources = root / "docs/technical-reviews/artifacts/natural-20.2-sources"
qconfig = Qwen3VLConfig.from_dict(json.loads((sources / "qwen-config.json").read_text()))
wconfig = WhisperConfig.from_dict(json.loads((sources / "whisper-config.json").read_text()))
with torch.device("meta"):
    qwen = Qwen3VLForConditionalGeneration(qconfig)
    whisper = WhisperForConditionalGeneration(wconfig)
qwen.tie_weights()
whisper.tie_weights()
def count(model):
    return sum(p.numel() for p in model.parameters())
qcount, wcount = count(qwen), count(whisper)
assert qcount == 2127532032, qcount
assert wcount == 241734912, wcount
qparts = {"vision": count(qwen.model.visual), "language": count(qwen.model.language_model)}
assert sum(qparts.values()) == qcount
qwen.requires_grad_(False)
with torch.device("meta"):
    adapted = get_peft_model(qwen, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)", task_type=TaskType.CAUSAL_LM))
trainable = {name: list(p.shape) for name, p in adapted.named_parameters() if p.requires_grad}
lora_count = sum(math.prod(shape) for shape in trainable.values())
assert lora_count == 1605632
assert len(trainable) == 112
assert all("lora_" in name for name in trainable)
assert count(adapted) == qcount + lora_count
headers = []
for model_name, revision in [("Qwen--Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
    ("openai--whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")]:
    path = root / "outputs/natural-extension/student-base-cache/hf" / ("models--" + model_name) / "snapshots" / revision / "model.safetensors"
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        raw = stream.read(size)
    header = json.loads(raw)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    count_header = sum(math.prod(v["shape"]) for v in tensors.values())
    expected = qcount if model_name.startswith("Qwen") else wcount
    assert count_header == expected, (model_name, count_header)
    headers.append({"model": model_name, "revision": revision, "path": str(path.relative_to(root)),
        "file_bytes": path.stat().st_size, "header_bytes": size, "header_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(tensors), "unique_serialized_elements": count_header,
        "dtype_counts": dict(Counter(v["dtype"] for v in tensors.values())),
        "shared_tensors_metadata": header.get("__metadata__", {})})
    (sources / ("qwen-header.json" if model_name.startswith("Qwen") else "whisper-header.json")).write_bytes(raw)
adapter_path = root / "outputs/natural-extension/runs/train-37190116187/natural-natural-v3-train-37190116187-1/review/adapter/adapter_model.safetensors"
with adapter_path.open("rb") as stream:
    size = struct.unpack("<Q", stream.read(8))[0]
    raw = stream.read(size)
adapter_header = json.loads(raw)
adapter_tensors = {name: value for name, value in adapter_header.items() if name != "__metadata__"}
adapter_count = sum(math.prod(v["shape"]) for v in adapter_tensors.values())
assert adapter_count == lora_count
assert len(adapter_tensors) == 112
assert all(".lora_" in name for name in adapter_tensors)
(sources / "adapter-header.json").write_bytes(raw)
result = {
    "reviewer_task": "/root/natural_factual_20_2",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "device": "meta; CPU process, no model data loaded"},
    "qwen_unique_parameters": qcount, "qwen_parts": qparts,
    "dtype_element_bytes": {str(dtype): torch.empty(0, dtype=dtype).element_size() for dtype in (torch.float32, torch.float16, torch.bfloat16)},
    "qwen_embedding_head_tied": adapted.base_model.model.get_input_embeddings().weight is adapted.base_model.model.get_output_embeddings().weight,
    "whisper_unique_parameters": wcount,
    "whisper_embedding_head_tied": whisper.get_input_embeddings().weight is whisper.get_output_embeddings().weight,
    "qwen_contains_moe_modules": any("expert" in name or "router" in name for name, _ in adapted.named_modules()),
    "lora_unique_parameters": lora_count, "lora_tensor_count": len(trainable), "lora_shapes": dict(Counter(tuple(x) for x in trainable.values()).items()),
    "lora_formula": "28 * [8*(2048+2048) + 8*(2048+1024)] = 1605632",
    "serialized_headers": headers,
    "adapter_header": {"parameters": adapter_count, "tensors": len(adapter_tensors), "sha256_weights": hashlib.sha256(adapter_path.read_bytes()).hexdigest(), "file_bytes": adapter_path.stat().st_size},
    "scope": "Independent fixed-config meta construction and safetensors header inspection; no forward, backward, generation or quality claim",
}
result["lora_shapes"] = {str(k): v for k,v in result["lora_shapes"].items()}
print(json.dumps(result, ensure_ascii=False, indent=2))
