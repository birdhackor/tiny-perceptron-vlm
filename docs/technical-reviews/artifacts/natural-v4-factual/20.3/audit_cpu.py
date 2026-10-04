"""Independent bounded CPU arithmetic and record audit for lesson 20.3.

No pretrained tensor payload, ASR inference, GPU work, or training replication.
Run from the repository root with .venv/bin/python.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import unicodedata

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import CapstoneModel, default_config
from tiny_perceptron.natural_concepts import thin_stroke_report

torch.set_num_threads(1)
RAW = ROOT / "outputs/natural-v4/factual-research/20.3"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def distance(a, b):
    # Fresh complete 2D dynamic program; independent of repository metric code.
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        table[i][0] = i
    for j in range(len(b) + 1):
        table[0][j] = j
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = min(table[i-1][j] + 1, table[i][j-1] + 1,
                              table[i-1][j-1] + (x != y))
    return table[-1][-1]

def norm(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())

result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                          "torch_git": torch.version.git_version, "device": "cpu",
                          "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()},
          "limits": "CPU mechanism and original-record audit only; no original ASR/GPU computation repeated."}
assert torch.__version__ == "2.14.1+cpu" and not torch.cuda.is_available()

# Execute the exact lesson block, preserving its actual stdout separately in JSON.
s = (RAW / "20.3.source.md").read_text()
block = re.search(r"```python\n(.*?)```", s, re.S).group(1)
import contextlib, io
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(block, "course/chapters/20.md#20.3", "exec"), {})
result["exact_lesson_stdout"] = stream.getvalue()
result["storage"] = {str(n): {str(b): {"bytes": n*b, "GiB": n*b/1024**3,
                                                "rounded_GiB": round(n*b/1024**3, 2)}
                               for b in (4, 2)} for n in (2_127_532_032, 328_128)}
result["element_size_bytes"] = {str(t): torch.empty(1, dtype=t).element_size()
                                 for t in (torch.float32, torch.float16, torch.bfloat16)}
assert result["storage"]["2127532032"]["4"]["bytes"] == 8_510_128_128
assert result["storage"]["2127532032"]["2"]["bytes"] == 4_255_064_064
assert result["storage"]["328128"]["4"]["rounded_GiB"] == 0.0
assert result["storage"]["328128"]["2"]["rounded_GiB"] == 0.0

# Independent config/source count, with tied embeddings counted once.
config = load(RAW / "qwen-config.json")
t, v = config["text_config"], config["vision_config"]
h, d, q, kv, f = t["hidden_size"], t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"], t["intermediate_size"]
text_parts = {"tied_embedding_and_head": t["vocab_size"]*h,
              "all_text_attention_weights": t["num_hidden_layers"]*(2*h*q*d + 2*h*kv*d),
              "all_text_qk_norms": t["num_hidden_layers"]*2*d,
              "all_text_MLP_weights": t["num_hidden_layers"]*3*h*f,
              "all_text_layer_RMSnorms": t["num_hidden_layers"]*2*h,
              "final_RMSnorm": h}
vh, vf, merge = v["hidden_size"], v["intermediate_size"], v["hidden_size"]*v["spatial_merge_size"]**2
merger_linear = merge*merge + merge + merge*v["out_hidden_size"] + v["out_hidden_size"]
vision_parts = {"patch_conv": v["in_channels"]*v["temporal_patch_size"]*v["patch_size"]**2*vh+vh,
                "position_embedding": v["num_position_embeddings"]*vh,
                "all_vision_attention": v["depth"]*(4*vh*vh+4*vh),
                "all_vision_MLP": v["depth"]*(2*vh*vf+vf+vh),
                "all_vision_LayerNorms": v["depth"]*4*vh,
                "main_merger": merger_linear + 2*vh,
                "three_deepstack_mergers": len(v["deepstack_visual_indexes"])*(merger_linear+2*merge)}
header = load(RAW / "qwen-safetensors-header.json")
tensors = {k: x for k, x in header.items() if k != "__metadata__"}
header_count = sum(math.prod(x["shape"]) for x in tensors.values())
header_visual = sum(math.prod(x["shape"]) for k, x in tensors.items() if ".visual." in k)
assert config["tie_word_embeddings"] is True and header_count == 2_127_532_032
assert sum(text_parts.values()) + sum(vision_parts.values()) == header_count
result["qwen_count"] = {"config_sha256": sha(RAW/"qwen-config.json"),
                        "header_sha256": sha(RAW/"qwen-safetensors-header.json"),
                        "tensor_count": len(tensors), "BF16_only": {x["dtype"] for x in tensors.values()} == {"BF16"},
                        "text_parts": text_parts, "text_parameters": sum(text_parts.values()),
                        "vision_parts": vision_parts, "vision_parameters": sum(vision_parts.values()),
                        "header_visual_count": header_visual, "total": header_count,
                        "payload_values_downloaded": 0}

# Current tiny model's parameter list verifies the exercise's 328,128 input.
torch.manual_seed(3)
model = CapstoneModel(default_config())
description = model.description()
assert description["parameters"] == 328_128
assert description["logical_active_parameters"] == 195_776
assert sum(p.numel()*p.element_size() for p in model.parameters()) == 1_312_512
result["capstone"] = description

# Exact prerequisite area-pooling probe and explicit non-injectivity witness.
stroke = thin_stroke_report()
assert stroke == {"original_shape": [32, 32], "small_shape": [4, 4], "original_brightest": 1.0,
                  "small_brightest": 0.125, "pixels_at_least_half_bright_before": 32,
                  "pixels_at_least_half_bright_after": 0}
x1 = torch.zeros(1, 1, 32, 32); x1[:, :, :, 9] = 1
x2 = torch.zeros_like(x1); x2[:, :, :, 15] = 1
same = torch.equal(torch.nn.functional.interpolate(x1, (4,4), mode="area"),
                   torch.nn.functional.interpolate(x2, (4,4), mode="area"))
assert same and not torch.equal(x1, x2)
result["stroke"] = {"exact_report": stroke, "distinct_strokes_same_reduced_input": same,
                    "scope": "One-pixel toy information loss, not OCR accuracy."}

# Bounded low-rank update: frozen base values still participate in forward/backward.
torch.manual_seed(11)
base = torch.nn.Parameter(torch.randn(6, 4), requires_grad=False)
A = torch.nn.Parameter(torch.randn(2, 4)); B = torch.nn.Parameter(torch.randn(6, 2))
before = base.detach().clone(); x = torch.randn(3, 4, requires_grad=True)
saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda z: (saved.append(list(z.shape)) or z), lambda z: z):
    output = x @ base.T + (x @ A.T) @ B.T
    loss = output.square().mean()
opt = torch.optim.AdamW([A, B], lr=0.01)
loss.backward(); opt.step()
assert base.grad is None and torch.equal(base, before) and x.grad is not None
result["lora_mechanism"] = {"base_values": base.numel(), "adapter_values": A.numel()+B.numel(),
                            "base_grad_is_none": base.grad is None, "base_unchanged": torch.equal(base,before),
                            "input_gradient_exists": x.grad is not None, "saved_tensor_shapes": saved,
                            "AdamW_state": [{k: list(z.shape) for k,z in opt.state[p].items()} for p in (A,B)],
                            "scope": "6x4 frozen CPU base, rank 2, one AdamW update; no quality or GPU memory conclusion."}

# Independently recount all 32 ASR result strings, with unchanged source references.
asr = {}
records = {}
source_files = [ROOT/"docs/natural-assistant/v4/data/voice-sources.json",
                ROOT/"docs/natural-assistant/v4/data/voice-question-sources.json"]
source_rows = {r["id"]:r for p in source_files for r in load(p)["audio_rows"] if r["split"] == "validation"}
assert len(source_rows) == 16
for variant in ("small", "turbo"):
    path = ROOT/f"docs/natural-assistant/evidence/v4-research/asr-cpu-validation/{variant}-records.json"
    rows = load(path); records[variant] = rows
    totals = {"raw_errors": 0, "raw_reference_characters": 0, "errors": 0, "reference_characters": 0}
    for row in rows:
        ref, pred = row["reference_transcript"], row["transcript"]
        assert row["split"] == "validation"
        source_row = source_rows[row["id"]]
        assert ref == source_row["user"] and row["source_audio_sha256"] == source_row["sha256"]
        nr, np = norm(ref), norm(pred)
        actual = {"raw_errors": distance(ref,pred), "raw_reference_characters": len(ref),
                  "errors": distance(nr,np), "reference_characters": len(nr)}
        assert nr == row["normalized_reference"] and np == row["normalized_prediction"]
        for k, number in actual.items():
            assert number == row[k]; totals[k] += number
    asr[variant] = {"source_sha256": sha(path), "recordings": len(rows), **totals,
                    "raw_micro_CER": totals["raw_errors"]/totals["raw_reference_characters"],
                    "normalized_micro_CER": totals["errors"]/totals["reference_characters"],
                    "model": rows[0]["asr_repo"], "revision": rows[0]["asr_revision"]}
assert len(records["small"]) == len(records["turbo"]) == 16
for a,b in zip(records["small"],records["turbo"],strict=True):
    for k in ("id", "reference_transcript", "audio_sha256", "source_audio_sha256", "split"):
        assert a[k] == b[k]
assert asr["small"]["errors"] == 117 and asr["turbo"]["errors"] == 50
assert asr["small"]["raw_errors"] == 138 and asr["turbo"]["raw_errors"] == 68
assert asr["turbo"]["normalized_micro_CER"] < asr["small"]["normalized_micro_CER"]
assert asr["turbo"]["raw_micro_CER"] < asr["small"]["raw_micro_CER"]
selection_path=ROOT/"docs/natural-assistant/v4/asr-selection.json"
selection=load(selection_path)
assert selection["selected_variant"] == "turbo"
result["ASR_selection"] = {"variants": asr, "same_16_validation_inputs": True,
                            "source_manifest_sha256": {str(p.relative_to(ROOT)):sha(p) for p in source_files},
                            "selection_sha256": sha(selection_path), "selected": selection["model"],
                            "scope": "Recomputed existing outputs; no ASR inference/audio decoding or spontaneous-speech generalization."}

# Audit the fixed GPU record, not GPU allocation/training replication.
directory=ROOT/"outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1"
path=directory/"review/result.json"; report=load(path); receipt=load(directory/"result.json")
assert receipt["experiment"] == report
assert sha(path) == "e84da438dc2931570a238f8eb3b6b526b4c3a5e09e40d0f43d92b9e88ce865c7"
assert sha(directory/"result.json") == "e0e63ead465ad7d1472c903ba71318f4f9551a43e697d4ee693e104cb857dcfe"
history=report["history"]
assert [r["step"] for r in history] == list(range(1,2078))
assert all(len(r["row_ids"]) == 2 for r in history)
assert sum(len(r["row_ids"]) for r in history) == report["trained_rows"] == 4154
tokens=sum(r["supervised_tokens"] for r in history)
assert tokens == 78872
from collections import Counter
manifest_path=ROOT/"docs/natural-assistant/v4/manifest.json"
manifest=load(manifest_path)
assert sha(manifest_path) == report["manifest_sha256"]
train_ids={r["id"] for r in manifest["rows"] if r["split"] == "train"}
used=Counter(n for r in history for n in r["row_ids"])
assert len(train_ids) == 2077 and set(used) == train_ids and set(used.values()) == {2}
assert report["total_parameters"]-report["trainable_parameters"] == header_count
assert all(re.fullmatch(r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_[AB]\..*", n)
           for n in report["optimizer_parameter_names"])
assert report["optimizer_parameter_names"] == report["trainable_parameter_names"]
assert len(report["optimizer_parameter_names"]) == 112
assert 28*(8*(2048+2048) + 8*(2048+1024)) == report["trainable_parameters"] == 1605632
result["training_record"] = {"input_path": str(path.relative_to(ROOT)), "input_sha256": sha(path),
                              "wrapper_sha256": sha(directory/"result.json"),
                              "run_id": receipt["run_id"], "revision": receipt["revision"],
                              **{k: report[k] for k in ["model","model_revision","asr_model","asr_revision","versions",
                                 "device","dtype","gpu_name","min_pixels","max_pixels","max_tokens","seed",
                                 "learning_rate","gradient_accumulation","total_parameters","trainable_parameters",
                                 "completed_steps","trained_rows","elapsed_seconds","peak_cuda_memory_allocated_bytes"]},
                              "summed_supervised_tokens": tokens,
                              "manifest_sha256": sha(manifest_path), "distinct_train_rows":len(train_ids),
                              "each_train_row_occurrences":2,
                              "base_count_subtraction": report["total_parameters"]-report["trainable_parameters"],
                              "core_minutes": report["elapsed_seconds"]/60,
                              "outer_seconds": report["execution"]["seconds"],
                              "outer_minutes": report["execution"]["seconds"]/60,
                              "peak_allocated_GiB": report["peak_cuda_memory_allocated_bytes"]/1024**3,
                              "optimizer_names": len(report["optimizer_parameter_names"]),
                              "memory_scope": "PyTorch allocated tensors after post-load reset; not reserved memory/context/whole-card/minimum.",
                              "own_replication": False}
execution=load(directory/"review/execution.json")
assert execution["resource_spec"] == {"cpu":4,"memory_gib":32,"seconds":3600,"gpu":True}
result["training_record"]["resource_spec"]=execution["resource_spec"]

print(json.dumps(result,indent=2,ensure_ascii=False))
