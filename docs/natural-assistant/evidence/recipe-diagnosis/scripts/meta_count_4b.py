"""No weights downloaded/loaded and no forward/generate: CPU metadata construction only."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

OUT = Path(__file__).resolve().parent
config = AutoConfig.from_pretrained(OUT, local_files_only=True)
with init_empty_weights():
    model = Qwen3VLForConditionalGeneration(config)
model.tie_weights()
base_count = sum(p.numel() for p in model.parameters())
categories = {}
for name, parameter in model.named_parameters():
    category = "vision" if name.startswith("model.visual.") else "text"
    categories[category] = categories.get(category, 0) + parameter.numel()
qv = [{"name": name, "shape": list(module.weight.shape),
       "rank8_parameters": 8 * sum(module.weight.shape)}
      for name, module in model.named_modules()
      if name.startswith("model.language_model.layers.") and name.endswith((".self_attn.q_proj", ".self_attn.v_proj"))]
target_regex = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
model.requires_grad_(False)
with init_empty_weights():
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=target_regex, task_type=TaskType.CAUSAL_LM))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable == sum(item["rank8_parameters"] for item in qv)
assert all(p.device.type == "meta" for p in model.parameters())
index = json.loads((OUT / "model.safetensors.index.json").read_text())
report = {
    "scope": "CPU metadata only: init_empty_weights; no checkpoint weights loaded; no forward/generate; no CUDA operations",
    "official_repo": "Qwen/Qwen3-VL-4B-Instruct",
    "official_revision": json.loads((OUT / "official-source-receipts.json").read_text())["revision"],
    "architecture": config.architectures,
    "config_model_type": config.model_type,
    "dense_evidence": "qwen3_vl_text config and Qwen3VLForConditionalGeneration class; text MLP has dense projections and no expert/router configuration",
    "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "accelerate"]},
    "meta_construction_success": True,
    "all_parameter_devices": sorted({p.device.type for p in model.parameters()}),
    "base_unique_parameters": base_count,
    "base_parameter_categories": categories,
    "tie_word_embeddings": config.tie_word_embeddings,
    "qv_target_module_count": len(qv),
    "qv_target_shapes_counts": {str(shape): sum(item["shape"] == list(shape) for item in qv)
                              for shape in sorted({tuple(item["shape"]) for item in qv})},
    "qv_lora_rank": 8,
    "qv_lora_alpha": 16,
    "qv_lora_trainable_parameters": trainable,
    "base_plus_lora_parameters": sum(p.numel() for p in model.parameters()),
    "safetensors_index_total_size_bytes": index["metadata"]["total_size"],
    "bf16_unique_base_weight_bytes": base_count * 2,
    "bf16_unique_base_weight_gib": base_count * 2 / 2**30,
    "bf16_lora_weights_gib": trainable * 2 / 2**30,
    "fp32_lora_weights_gradient_two_adam_states_gib": trainable * 16 / 2**30,
    "text_configuration": config.text_config.to_dict(),
    "vision_configuration": config.vision_config.to_dict(),
    "qv_modules": qv,
    "inference_kv_cache_bf16_per_token_bytes": config.text_config.num_hidden_layers * 2
        * config.text_config.num_key_value_heads * config.text_config.head_dim * 2,
    "kv_scope": "batch=1 text decoder KV only, two K/V tensors, BF16; excludes vision features, activations, allocator/workspaces, model and LoRA weights",
    "support_limit": "Confirms installed transformers 4.57.6 can construct official 4B configuration and PEFT 0.18.1 q/v adapter metadata. Does not prove actual weight loading, processor/inference, numerical stability, GPU fit, quality, or throughput.",
}
assert report["bf16_unique_base_weight_bytes"] == index["metadata"]["total_size"], "Stored bytes differ from BF16 parameter count"
(OUT / "meta-4b-analysis.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ["text_configuration", "vision_configuration", "qv_modules"]}, indent=2))
