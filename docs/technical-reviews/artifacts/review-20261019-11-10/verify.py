"""核對切塊算例、矩陣維度與既有實驗紀錄；不進行訓練。"""

import hashlib
import html
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, patchify
from scripts.course_experiments.modalities import _hash, _media, _vision_records
from types import SimpleNamespace

ARTIFACT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fetch(url):
    command = ["curl", "-fsSL", url]
    raw = subprocess.check_output(command)
    return raw, {"command_argv": command, "exit_code": 0, "response_sha256": sha(raw)}


def plain(raw):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


torch.set_num_threads(1)
torch.manual_seed(42)
environment = {
    "python": sys.version,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "platform": platform.platform(),
    "torch_threads": str(torch.get_num_threads()),
}

run = ROOT / "outputs/reviewer-tools/runs/20261020T054645599611Z"
files = {}
for name in ("stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json", "fence-1.py"):
    raw = (run / name).read_bytes()
    files[name] = {"sha256": sha(raw), "content": raw.decode("utf-8")}
assert json.loads(files["execution.json"]["content"])["exit_code"] == 0
assert json.loads(files["extraction.json"]["content"])["source_sha256"] == "76281f5bd14e293adc389731250e302903d4dce8441b4db95b5abac9aa351ecc"
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/11.md#11.10" --execute',
    "result": "exit_code=0；原始標準輸出、錯誤輸出、環境與執行收據逐份複製於 files。",
    "environment": json.loads(files["environment.json"]["content"]),
    "files": files,
})

snapshots = []
originals = [
    {
        "id": "flexivit",
        "url": "https://arxiv.org/html/2212.08013v2",
        "version": "arXiv:2212.08013v2",
        "locator": "Vision Transformers recap；Using pre-trained FlexiViT models；Flexifying the training of ViT models",
        "excerpts": [
            "smaller patch sizes correspond to longer input sequences and slower, more expressive models.",
            "the optimal patch size is not necessarily the smallest.",
        ],
        "inspection_note": "讀取 patchification 的序列長度公式及 patch embedding／position embedding 對 patch 大小的依賴；亦讀取 PI-resize 與彈性訓練例外、下游偵測結果。較細切法可以保留更多空間區分，但不保證每項任務較佳。專案固定輸入矩陣不具有 FlexiViT 的重新取樣實作。",
    },
    {
        "id": "attention",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7",
        "locator": "Scaled Dot-Product Attention，equation (1)；Applications of Attention in our Model；Why Self-Attention，Table 1",
        "excerpts": ["The keys and values are also packed together into matrices"],
        "inspection_note": "核對 QK 轉置得到 query×key 分數矩陣、softmax 前套因果遮罩、self-attention 每層複雜度 O(n²d)。完整矩陣格數與可見格數不同；格數也不直接等於秒數、總參數或所有後端的記憶體配置。",
    },
    {
        "id": "linear",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/linear.py",
        "version": "PyTorch 2.14.1；commit 5c4886908584029761b579af026dcfb627c84070",
        "locator": "Linear：Shape、weight、__init__、forward",
        "excerpts": ["Applies an affine linear transformation to the incoming data"],
        "inspection_note": "讀取 Linear 的輸入最後維度必須等於 in_features、weight 的 out_features×in_features 形狀、bias 及 F.linear 呼叫。版本對上當次安裝 wheel 的 torch.version.git_version。",
    },
    {
        "id": "python",
        "url": "https://docs.python.org/3.13/reference/expressions.html",
        "version": "Python 3.13.16 documentation",
        "locator": "Binary arithmetic operations：floor division",
        "excerpts": ["floor division of integers results in an integer"],
        "inspection_note": "讀取 // 為 floor division 的完整定義及型別規則；本頁影像高度與 patch 大小都是正整數，所以結果等於除法商的非負整數部分。不把這個說法延伸成負數向零截斷。",
    },
]
for source in originals:
    raw, receipt = fetch(source["url"])
    text = plain(raw.decode("utf-8"))
    for excerpt in source["excerpts"]:
        assert excerpt in text
    snapshots.append({**source, "accessed_on": "2026-10-20", "checked_original": True, **receipt})
save("original-sources.json", {"sources": snapshots, "storage": "第三方全文只經記憶體讀取；保存取用收據、全文雜湊及支持主張的小段摘錄。"})

geometry = []
for height in (16, 32):
    for patch in (8, 4, 2):
        visual = (height // patch) ** 2
        patches = patchify(torch.zeros(1, 3, height, height), patch)
        assert list(patches.shape) == [1, visual, 3 * patch**2]
        geometry.append({"height": height, "patch_size": patch, "per_side": height // patch,
                         "visual_tokens": visual, "total": 20 + visual, "attention_cells": (20 + visual) ** 2,
                         "patchified_shape": list(patches.shape)})
assert [r["visual_tokens"] for r in geometry] == [4, 16, 64, 16, 64, 256]
assert [r["attention_cells"] for r in geometry] == [576, 1296, 7056, 1296, 7056, 76176]

attentions = []
for length in (24, 36, 84):
    q = torch.zeros(1, 1, length, 4)
    allowed = attention_mask(torch.arange(length), torch.arange(length))
    _, weights = manual_attention(q, q, q, allowed)
    assert weights.numel() == length**2
    assert int(allowed.sum()) == length * (length + 1) // 2
    attentions.append({"length": length, "weights_shape": list(weights.shape),
                       "full_cells": weights.numel(), "causally_visible_cells": int(allowed.sum())})

encoder = VisionEncoder(patch_size=4)
encoder.patch_size = 2
try:
    encoder(torch.zeros(1, 3, 16, 16))
except RuntimeError as error:
    dimension_error = str(error)
else:
    raise AssertionError("更換 patch_size 後應出現輸入矩陣維度不符")
assert "12" in dimension_error and "48" in dimension_error

model = TinyLM(ModelConfig(max_length=64))
try:
    model(embeddings=torch.zeros(1, 84, model.config.width))
except ValueError as error:
    capacity_error = str(error)
else:
    raise AssertionError("超過上下文上限應被拒絕")
inputs = torch.ones(84, dtype=torch.long)
labels = torch.full((84,), IGNORE)
labels[70:] = 2
assert int((labels[:64] != IGNORE).sum()) == 0
try:
    pad_batch([(inputs, labels)], max_length=64)
except ValueError as error:
    truncation_error = str(error)
else:
    raise AssertionError("截斷後沒有有效答案應被拒絕")

result_path = ROOT / "docs/course-experiments/results/vision_ablation.json"
record = json.loads(result_path.read_bytes())
variants = record["results"]["patch_variants"]
assert set(variants) == {"4", "8"}
tok = ByteTokenizer()
variant_checks = {}
for patch, variant in variants.items():
    config = ModelConfig(**variant["training"]["config"])
    modal = variant["training"]["modal_config"]
    candidate = MultiModalLM(TinyLM(config), modal["vision_width"], modal["audio_width"])
    candidate.vision = VisionEncoder(modal["vision_width"], modal["image_size"], int(patch))
    parameters = sum(p.numel() for p in candidate.parameters())
    assert parameters == variant["training"]["parameters"]
    samples = variant["test"]["samples"]
    rescored = []
    for sample in samples:
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        exact = raw == tok.encode(sample["target"])
        assert exact == sample["exact_match"]
        rescored.append(exact)
    assert sum(rescored) == variant["test"]["correct"] == 9
    assert len(samples) == variant["test"]["examples"] == 12
    assert sum(rescored) / len(samples) == variant["test"]["exact_match"] == 0.75
    variant_checks[patch] = {
        "parameters": parameters,
        "projection_shape": list(candidate.vision.projection.weight.shape),
        "projection_parameters": sum(p.numel() for p in candidate.vision.projection.parameters()),
        "vision_position_parameters": candidate.vision.position.numel(),
        "encoder_parameters": sum(p.numel() for p in candidate.vision.parameters()),
        "sample_exact_match": rescored,
        "correct": sum(rescored), "examples": len(samples),
        "steps": variant["training"]["steps"],
        "effective_targets": variant["training"]["effective_targets"],
    }
assert variant_checks["8"]["parameters"] - variant_checks["4"]["parameters"] == 2112

splits = _vision_records(("shape?", "color?"))
split_checks = {}
for split, rows in splits.items():
    saved = record["results"]["data"]["splits"][split]
    assert rows == saved["records"]
    assert _hash(rows) == saved["sha256"]
    assert all("digits" not in row and "image" not in row for row in rows)
    split_checks[split] = {"examples": len(rows), "families": len({r["family"] for r in rows}),
                           "questions": sorted({r["question"] for r in rows}), "sha256": _hash(rows)}
image, _ = _media(splits["test"][0], SimpleNamespace(device="cpu"))
assert list(image.shape) == [3, 16, 16]

code_paths = ["tiny_perceptron/multimodal.py", "tiny_perceptron/attention.py", "tiny_perceptron/model.py",
              "tiny_perceptron/data.py", "scripts/course_experiments/modalities.py"]
code_checks = {}
for path in code_paths:
    current = sha((ROOT / path).read_bytes())
    recorded = record["code_sha256"][path]
    code_checks[path] = {"current_sha256": current, "experiment_sha256": recorded,
                          "matches_experiment": current == recorded}

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-10/verify.py",
    "result": "所有決定性核對與逐題重算通過；沒有重新訓練或載入權重重做生成。",
    "environment": environment,
    "geometry": geometry, "attention_tables": attentions,
    "dimension_error": dimension_error, "capacity_error": capacity_error,
    "truncation": {"original_targets": int((labels != IGNORE).sum()), "remaining_targets": 0,
                   "error": truncation_error},
    "experiment_source": {"path": result_path.relative_to(ROOT).as_posix(),
                          "sha256": sha(result_path.read_bytes()), "revision": record["revision"],
                          "seed": record["seed"], "device": record["device"], "gpu": record["gpu"],
                          "python": record["python_version"], "torch": record["torch_version"]},
    "variants": variant_checks, "splits": split_checks, "experiment_code": code_checks,
}, ensure_ascii=False, indent=2))
