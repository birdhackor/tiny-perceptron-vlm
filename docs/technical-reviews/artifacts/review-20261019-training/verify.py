"""以短程 CPU 檢查核對訓練入口；所有工作檔留在本檔旁。"""

import ast
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
WORK = BASE / "work"
WORK.mkdir(exist_ok=False)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

import torch

torch.set_num_threads(2)
from scripts import evaluate, prepare_data, train
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.multimodal import MultiModalLM, scene, tone
from tiny_perceptron.training import choose_device, load_checkpoint, save_checkpoint, seed_everything

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-training/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()},
    "runs": [],
    "observations": {},
}


def run(arguments, expected=0):
    command = [sys.executable, *map(str, arguments)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=50, env=os.environ)
    entry = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    result["runs"].append(entry)
    if completed.returncode != expected:
        raise RuntimeError(json.dumps(entry, ensure_ascii=False))
    return completed.stdout


def last_json(output):
    return json.loads(output.strip().splitlines()[-1])


try:
    source = ROOT / "docs/course-review-20261019/freeze-01/sources/training.md"
    result["observations"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["observations"]["figure_references"] = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source.read_text())
    result["observations"]["split_documents"] = {k: len(v) for k, v in split_documents(toy_documents(), 42).items()}
    for kind in ("toy-text", "attributes-sft", "style", "safety", "preference"):
        run(["scripts/prepare_data.py", "--kind", kind, "--seed", "42", "--output", WORK / "data"])
    result["observations"]["families"] = {}
    for kind in ("toy-text", "attributes-sft", "style", "safety", "preference"):
        parts = {key: [json.loads(line) for line in (WORK / "data" / kind / f"{key}.jsonl").read_text().splitlines()] for key in ("train", "validation", "test")}
        families = {key: {row["family"] for row in rows} for key, rows in parts.items()}
        result["observations"]["families"][kind] = {"counts": {k: len(v) for k, v in parts.items()}, "intersections": {f"{a}/{b}": sorted(families[a] & families[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}}
    rows = prepare_data.generate_records("attributes-sft")
    result["observations"]["attributes_examples"] = [row for row in rows if row["family"] in ("red:square:low", "blue:circle:high")]
    tok = ByteTokenizer()
    example = [{"role": "user", "content": "color=red;shape=circle;pitch=low;shape?"}, {"role": "assistant", "content": "circle"}]
    x, y = render_chat(example)
    result["observations"]["tokenizer"] = {"A": tok.encode("A"), "eos": tok.eos_id, "common_chinese_character": tok.encode("圓"), "supervised_circle": y[y != -100].tolist(), "expected_circle": tok.encode("circle") + [tok.eos_id]}
    result["observations"]["byte_example"] = __import__("tiny_perceptron.tokenization", fromlist=["generation_report"]).generation_report(tok, [73, 2])
    for task in ("text", "sft", "vision"):
        path = WORK / f"dry-{task}.pt"
        report = last_json(run(["scripts/train.py", "--task", task, "--device", "cpu", "--output", path]))
        entry = report["history"][0]
        assert report["mode"] == "dry-run-no-weight-update" and not path.exists()
        assert math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"]) and entry["grad_norm"] > 0
        assert entry["effective_tokens"] is None if task == "vision" else entry["effective_tokens"] > 0
    for model in ("bigram", "mlp"):
        before = last_json(run(["scripts/train_simple.py", "--model", model, "--context", "3", "--width", "16", "--seed", "42", "--device", "cpu", "--output", WORK / "unused.pt"]))
        after = last_json(run(["scripts/train_simple.py", "--model", model, "--context", "3", "--width", "16", "--seed", "42", "--device", "cpu", "--train", "--steps", "200", "--output", WORK / f"{model}.pt"]))
        result["observations"][f"simple_{model}"] = {"before": before, "after": after}
    run(["scripts/infer.py", WORK / "bigram.pt", "--device", "cpu"], expected=1)
    seed_everything(42)
    model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
    save_checkpoint(WORK / "start.pt", model, step=0, metadata={"seed": 42, "note": "untrained baseline"})
    payload = torch.load(WORK / "start.pt", weights_only=True)
    result["observations"]["start"] = {"config": payload["config"], "optimizer": payload["optimizer"], "step": payload["step"], "train_defaults": vars(train.parser().parse_args([]))}
    for task, kind, count in (("text", "toy-text", 32), ("sft", "attributes-sft", 24)):
        path = WORK / f"{task}.pt"
        run(["scripts/evaluate.py", WORK / "start.pt", "--data", WORK / "data" / kind / "validation.jsonl", "--mode", task, "--tokens", str(count), "--device", "cpu", "--limit", "all", "--output", WORK / f"{task}-before.json"])
        run(["scripts/train.py", "--task", task, "--data", WORK / "data" / kind / "train.jsonl", "--checkpoint", WORK / "start.pt", "--seed", "42", "--device", "cpu", "--train", "--steps", "2", "--save-every", "100", "--output", path])
        run(["scripts/evaluate.py", path, "--data", WORK / "data" / kind / "validation.jsonl", "--mode", task, "--tokens", str(count), "--device", "cpu", "--limit", "all", "--output", WORK / f"{task}-after.json"])
        result["observations"][f"{task}_evaluations"] = {key: json.loads((WORK / f"{task}-{key}.json").read_text()) for key in ("before", "after")}
    for bits in (4, 8):
        run(["scripts/quantize.py", WORK / "sft.pt", "--bits", str(bits), "--output", WORK / f"int{bits}.pt"])
        run(["scripts/infer.py", WORK / f"int{bits}.pt", "--chat", "--prompt", "color=blue;shape=circle;pitch=low;color?", "--tokens", "24", "--device", "cpu", "--json"])
        for split in ("validation", "test"):
            run(["scripts/evaluate.py", WORK / f"int{bits}.pt", "--data", WORK / "data/attributes-sft" / f"{split}.jsonl", "--split-label", split, "--mode", "sft", "--tokens", "24", "--limit", "all", "--device", "cpu", "--output", WORK / f"int{bits}-{split}.json"])
    result["observations"]["size_check"] = []
    for name in ("sft.pt", "int4.pt", "int8.pt"):
        path = WORK / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model"]
        result["observations"]["size_check"].append({"path": name, "tensor_bytes": sum(t.numel() * t.element_size() for t in state.values()), "file_bytes": path.stat().st_size, "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype)} for k, t in state.items()}})
    for modality in ("vision", "audio"):
        run(["scripts/pretrain_encoders.py", "--modality", modality, "--train", "--steps", "1", "--device", "cpu", "--output", WORK / f"{modality}-encoder.pt"])
    for task, freeze in (("vision", "projector"), ("audio", "projector"), ("joint", "partial"), ("joint", "none")):
        path = WORK / f"modal-{task}-{freeze}.pt"
        args = ["scripts/train.py", "--task", task, "--checkpoint", WORK / "sft.pt", "--freeze", freeze, "--device", "cpu", "--train", "--steps", "2", "--stop-after", "1", "--output", path]
        if task != "audio":
            args += ["--vision-encoder", WORK / "vision-encoder.pt"]
        if task != "vision":
            args += ["--audio-encoder", WORK / "audio-encoder.pt"]
        run(args)
        payload = torch.load(path, weights_only=True)
        result["observations"][f"modal_{task}_{freeze}"] = {"format_version": payload["format_version"], "step": payload["step"], "optimizer_present": payload["optimizer"] is not None, "rng_fields": [k for k in payload if k.endswith("_rng")], "trainable_parameters": payload["trainable_parameters"], "modal_config": payload["modal_config"]}
        run(["scripts/infer_modal.py", path, "--color", "red", "--shape", "square", "--frequency", "220", "--tokens", "2", "--device", "cpu"])
        resume = ["scripts/train.py", "--task", task, "--checkpoint", path, "--resume", "--freeze", freeze, "--device", "cpu", "--train", "--steps", "2", "--output", path]
        run(resume)
    run(["scripts/evaluate.py", WORK / "modal-joint-partial.pt", "--mode", "text", "--device", "cpu", "--output", WORK / "not-produced.json"], expected=1)
    run(["scripts/train.py", "--task", "dpo", "--checkpoint", WORK / "sft.pt", "--data", WORK / "data/preference/train.jsonl", "--train", "--steps", "1", "--device", "cpu", "--output", WORK / "dpo.pt"])
    dpo = torch.load(WORK / "dpo.pt", weights_only=True)
    original = torch.load(WORK / "sft.pt", weights_only=True)
    result["observations"]["dpo_reference_unchanged"] = all(torch.equal(v, dpo["training_state"]["dpo_reference"]["model"][k]) for k, v in original["model"].items())
    for name in ("rms", "moe"):
        run(["scripts/train.py", "--task", "text", "--data", WORK / "data/toy-text/train.jsonl", "--train", "--steps", "1", "--seed", "42", "--device", "cpu", "--output", WORK / f"{name}.pt", *(["--norm", "rms"] if name == "rms" else ["--experts", "4", "--top-k", "2"])])
    run(["scripts/fetch_training_assets.py", "--list"])
    run(["scripts/fetch_course_models.py", "--list"])
    run(["-m", "scripts.course_experiments.run", "--experiment", "sft", "--device", "cuda", "--output", WORK / "experiments"], expected=1)
    run(["-m", "scripts.course_experiments.run", "--experiment", "sft", "--device", "mps", "--output", WORK / "experiments"], expected=2)
    for filename in ("scripts/fetch_training_assets.py", "scripts/fetch_course_models.py", "scripts/prepare_data.py", "scripts/train.py", "scripts/train_simple.py", "scripts/pretrain_encoders.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/infer_modal.py", "scripts/evaluate_modal.py", "scripts/prepare_ocr.py", "scripts/quantize.py"):
        run([filename, "--help"])
    result["observations"]["hand_examples"] = {"two_plus_three": 2 + 3, "half_of_two": 0.5 * 2, "loss_delta": 0.6 - 0.8, "size_change": 60000 - 80000, "correct_change": 3 - 4}
    result["status"] = "completed"
finally:
    (BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    shutil.rmtree(WORK)

print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "environment": result["environment"]}, ensure_ascii=False))
