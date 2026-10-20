"""讀取固定上游版本，只保存定位、指紋與支持主張的小段原文。"""

import concurrent.futures
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = []


def add(group, repo, revision, path, patterns, radius=3, limit=100):
    JOBS.append({"id": group, "url": f"https://raw.githubusercontent.com/{repo}/{revision}/{path}",
                 "version": revision, "path": path, "patterns": patterns,
                 "radius": radius, "limit": limit})


nano = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
mv = "1862b633fc082a723e78dbead9777545f09d960c"
mini = "f659b55761b754d306bd140573493a6543cafd7f"
smol = "f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541"
ao = "a701b6a6058720c21f95908b7ae4a24bf0cae1b6"
add("nano-readme", "karpathy/nanochat", nano, "README.md", ["depth", "tokenizer", "SFT", "CPU", "evaluation"], 2, 80)
add("nano-cpu", "karpathy/nanochat", nano, "runs/runcpu.sh", ["python", "depth", "device"], 1, 45)
add("nano-pipeline", "karpathy/nanochat", nano, "runs/speedrun.sh", ["scripts.tok", "scripts.base", "scripts.chat", "depth"], 2, 55)
add("nano-tokenizer-eval", "karpathy/nanochat", nano, "scripts/tok_eval.py", ["ratio", "decode", "bytes"], 2, 55)
add("nano-bpb", "karpathy/nanochat", nano, "nanochat/loss_eval.py", ["def evaluate", "loss", "total_bytes", "log(2"], 2, 65)
add("nano-tokenizer", "karpathy/nanochat", nano, "nanochat/tokenizer.py", ["mask.extend", "assistant", "render_conversation"], 2, 60)
add("nano-gpt", "karpathy/nanochat", nano, "nanochat/gpt.py", ["class MLP", "relu(x).square", '"wte":', "self.lm_head =", "self.mlp ="], 3, 35)
add("nano-engine", "karpathy/nanochat", nano, "nanochat/engine.py", ["class KVCache", "prefill", "def generate"], 2, 35)
add("nano-flash", "karpathy/nanochat", nano, "nanochat/flash_attention.py", ["FA3", "scaled_dot_product_attention", "import flash"], 2, 40)
for path, patterns in [
    ("README.md", ["SigLIP", "SFT", "纯文本", "预对齐", "首尾", "首末", "冻结", "预训练"]),
    ("model/model_vlm.py", ["siglip", "vision_proj", "image_ids", "image_pad", "tie_word", "hidden_act", "SiLU", "requires_grad"]),
    ("dataset/lm_dataset.py", ["assistant", "labels", "-100", "loss_mask"]),
    ("trainer/trainer_utils.py", ["requires_grad", "layers[0]", "layers[-1]", "vision_proj", "freeze"]),
    ("trainer/train_pretrain_vlm.py", ["freeze", "pretrain", "vlm_model", "train_ds"]),
    ("trainer/train_sft_vlm.py", ["freeze", "sft", "train_ds", "requires_grad"]),
    ("scripts/web_demo_vlm.py", ["gradio", "generate", "ChatInterface"]),
]:
    add("mv-" + path.replace("/", "-"), "jingyaogong/minimind-v", mv, path, patterns, 3, 90)
for path, patterns in [
    ("README.md", ["LoRA", "黑盒", "白盒", "蒸馏"]),
    ("trainer/train_dpo.py", ["def dpo_loss", "reference", "ref_model", "logsigmoid"]),
    ("model/model_minimind.py", ["hidden_act", "tie_word_embeddings", "class MOEFeedForward", "top_k", "aux_loss", "gate_proj", "up_proj"]),
    ("trainer/train_distillation.py", ["temperature", "kl_div", "cross_entropy", "loss_mask", "teacher_model"]),
]:
    add("mini-" + path.replace("/", "-"), "jingyaogong/minimind", mini, path, patterns, 3, 90)
for path, patterns in [
    ("units/en/unit1/1.md", ["instruction", "supervised", "pre-train"]),
    ("units/en/unit1/3.md", ["Supervised", "behavior", "dataset"]),
    ("units/en/unit2/1.md", ["preference", "DPO", "SFT"]),
    ("units/en/unit2/2.md", ["reference", "preference", "reward"]),
]:
    add("smol-" + path.replace("/", "-"), "huggingface/smol-course", smol, path, patterns, 2, 30)
add("safe-rlhf", "PKU-Alignment/safe-rlhf", "e8cca16665ef2340ac92c6514f05519310251581", "README.md", ["helpfulness", "cost", "reward", "safety"], 2, 50)
for path, patterns in [
    ("README.md", ["quantization", "kernel", "qat"]),
    ("docs/source/contributing/quantization_overview.rst", ["packing", "kernel", "dtype", "quantization"]),
    ("docs/source/workflows/qat.md", ["fake", "prepare", "convert"]),
]:
    add("ao-" + path.replace("/", "-"), "pytorch/ao", ao, path, patterns, 2, 55)
add("cs336-eval", "stanford-cs336/lectures", "de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e", "lecture_12.py", ["validity", "evaluation"], 2, 20)
add("cs336-data", "stanford-cs336/lectures", "de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e", "lecture_14.py", ["deduplication", "mixing"], 2, 20)
add("cs336-multimodal", "stanford-cs336/lectures", "de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e", "lecture_17.py", ["CLIP", "SigLIP", "resolution"], 2, 30)
add("harvard-2025", "Harvard-IACS/2025-AC215", "a0d3ed05aa749aa46c95c70a2f519db0eef2bb30", "_modules/week-06.md", ["RAG", "Agent"], 2, 30)
add("harvard-2026", "Harvard-IACS/2026-AC215", "f00978fe6d008935ab87c7fda589b461bec6d236", "README.md", ["2026", "Retrieval", "Agent", "RAG"], 2, 35)
add("mv-default-llm", "jingyaogong/minimind-v", mv, "model/model_minimind.py", ["hidden_act", "tie_word_embeddings", "gate_proj", "class MOEFeedForward", "self.mlp ="], 3, 65)
add("mini-lora", "jingyaogong/minimind", mini, "model/model_lora.py", ["class LoRA", "requires_grad", "lora_A", "lora_B", "rank"], 3, 45)
add("mit-2025", "MITDeepLearning/introtodeeplearning.com", "c3841a160a77d336391107cc71fd3b80555bfa22", "2025/index.html", ["Post-Training", "2025"], 1, 20)
add("mit-2026", "MITDeepLearning/introtodeeplearning.com", "c3841a160a77d336391107cc71fd3b80555bfa22", "2026/index.html", ["Computer Vision", "Sequence", "2026"], 1, 20)
add("mit-style", "MITDeepLearning/introtodeeplearning", "535e0862f52c81288258fae0516c40968d07f603", "lab3/LLM_Finetuning.ipynb", ["control", "style", "LoRA"], 1, 35)
add("mit-cv", "MITDeepLearning/introtodeeplearning", "535e0862f52c81288258fae0516c40968d07f603", "lab2/PT_Part2_Debiasing.ipynb", ["bias", "skin", "accuracy", "gender"], 1, 25)
add("ntu-mirror-topics", "MLNLP-World/MachineLearning2025Spring--Notes", "3ec1979138e82b2087a1018c7b91f2db5d8f98b3", "README.md", ["遗忘", "推理", "Agent", "2025"], 1, 30)
for repo, number in [("jingyaogong/minimind",476), ("jingyaogong/minimind",633), ("stanford-cs336/assignment1-basics",21), ("karpathy/nanochat",590), ("jingyaogong/minimind",732), ("jingyaogong/minimind-v",36)]:
    JOBS.append({"id": f"issue-{repo}-{number}", "url": f"https://api.github.com/repos/{repo}/issues/{number}", "version": "GitHub issue API 回應；含 created_at/updated_at", "patterns": [], "radius": 0, "limit": 0})
JOBS.append({"id": "hf-v2-tree", "url": "https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2?recursive=true&expand=false", "version": "979cdfacc588ad0536f1c64fff96f264571cf054", "patterns": [], "radius": 0, "limit": 0})
JOBS.append({"id": "hf-license", "url": "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/README.md", "version": "979cdfacc588ad0536f1c64fff96f264571cf054", "patterns": ["license:", "selftrained/v2", "MIT"], "radius": 1, "limit": 20})
for stage in ("moe-pretrain", "moe-sft", "moe-joint", "dense-joint"):
    JOBS.append({"id": "hf-manifest-"+stage, "url": f"https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/{stage}/inference-manifest.json", "version": "979cdfacc588ad0536f1c64fff96f264571cf054", "patterns": [], "radius": 0, "limit": 0})
for name, module in [("torch-zero", "torch/optim/optimizer.py"), ("torch-backward", "torch/_tensor.py"), ("torch-attention", "torch/nn/functional.py")]:
    add(name, "pytorch/pytorch", "v2.14.1", module,
        ["def zero_grad", "def backward", "Accumulates the gradient", "is_causal", "If a BoolTensor is provided"], 3, 45)


def fetch(job):
    command = ["curl", "-fsSL", "--max-time", "40", job["url"]]
    result = subprocess.run(command, capture_output=True)
    record = {k: v for k, v in job.items() if k not in {"patterns", "radius", "limit"}}
    record.update(command=command, exit_code=result.returncode,
                  accessed_on=datetime.date.today().isoformat(), bytes=len(result.stdout),
                  sha256=hashlib.sha256(result.stdout).hexdigest())
    if result.returncode:
        record["error"] = result.stderr.decode(errors="replace")
        return record
    if job["id"] == "hf-v2-tree":
        record["entries"] = json.loads(result.stdout)
        return record
    if job["id"].startswith("hf-manifest-"):
        response = json.loads(result.stdout)
        record["manifest_excerpt"] = {k:response[k] for k in ("schema", "selected_checkpoint_sha256", "files", "stage", "selected_step", "origin", "selection") if k in response}
        record["manifest_top_level_keys"] = list(response)
        record["manifest_data_counts"] = {k:len(v) for k,v in response.items() if k.endswith("sha256") and isinstance(v,dict)}
        return record
    if job["id"].startswith("issue-"):
        response = json.loads(result.stdout)
        record["issue_excerpt"] = {k: response[k] for k in ("html_url", "title", "created_at", "updated_at")}
        record["issue_excerpt"]["body_excerpt"] = (response.get("body") or "")[:2000]
        return record
    lines = result.stdout.decode(errors="replace").splitlines()
    indexes = set()
    for pattern in job["patterns"]:
        matches = [i for i, line in enumerate(lines) if pattern.lower() in line.lower()]
        for i in matches[:5]:
            indexes.update(range(max(0, i-job["radius"]), min(len(lines), i+job["radius"]+1)))
    record["total_lines"] = len(lines)
    record["excerpts"] = [{"line": i+1, "text": lines[i]} for i in sorted(indexes)[:job["limit"]]]
    return record


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fetch, JOBS))
    output = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-curriculum/inspect_upstream.py", "environment": {"python": sys.version, "platform": platform.platform(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}, "records": records}
    target = HERE / "upstream-evidence.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"path": target.relative_to(Path.cwd()).as_posix(), "records": len(records), "failures": [r["id"] for r in records if r["exit_code"]]}, ensure_ascii=False))
