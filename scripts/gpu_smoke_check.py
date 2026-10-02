"""用既有 CLI 驗證實際更新、checkpoint 與續訓；不下載外部訓練資料。"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training(arguments, intermediate=None):
    """第 41 步輸出時，第 40 步已完成存檔；保留它供中斷後續訓驗證。"""
    command = [sys.executable, "-u", "scripts/train.py", *arguments]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8") as process:
        for line in process.stdout:
            record = json.loads(line)
            if intermediate is not None and record.get("step") == 41:
                shutil.copyfile(intermediate[0], intermediate[1])
        if process.wait() != 0:
            raise RuntimeError("train.py 執行失敗")


def evaluate(model, records, device):
    import torch

    from tiny_perceptron.data import ByteTokenizer, load_jsonl, pad_batch, shifted
    from tiny_perceptron.model import masked_loss

    tokenizer = ByteTokenizer()
    examples = [
        shifted([tokenizer.bos_id] + tokenizer.encode(row["text"]) + [tokenizer.eos_id]) for row in load_jsonl(records)
    ]
    inputs, labels, valid = (value.to(device) for value in pad_batch(examples, max_length=64))
    model.eval()
    with torch.no_grad():
        return float(masked_loss(model(inputs, valid=valid)["logits"], labels))


def train_and_measure(directory, device="cuda", revision="local"):
    import torch

    from tiny_perceptron.data import ByteTokenizer
    from tiny_perceptron.model import ModelConfig, TinyLM
    from tiny_perceptron.training import load_checkpoint, seed_everything

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU 測試必須使用 CUDA，不能自動退回 CPU")
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--kind", "toy-text", "--output", str(directory / "data")],
        check=True,
        capture_output=True,
        text=True,
    )
    data = directory / "data/toy-text"
    seed_everything(42)
    initial = TinyLM(ModelConfig(max_length=64)).to(device)
    before = {split: evaluate(initial, data / f"{split}.jsonl", device) for split in ("train", "validation")}
    checkpoint = directory / "baseline-80.pt"
    intermediate = directory / "checkpoint-40.pt"
    run_training(
        [
            "--task",
            "text",
            "--train",
            "--device",
            device,
            "--steps",
            "80",
            "--save-every",
            "40",
            "--batch-size",
            "4",
            "--max-length",
            "64",
            "--lr",
            "0.003",
            "--data",
            str(data / "train.jsonl"),
            "--output",
            str(checkpoint),
        ],
        intermediate=(checkpoint, intermediate),
    )
    trained, payload = load_checkpoint(checkpoint, device)
    after = {split: evaluate(trained, data / f"{split}.jsonl", device) for split in ("train", "validation")}
    intermediate_payload = load_checkpoint(intermediate)[1]
    report = json.loads(checkpoint.with_suffix(".json").read_text())
    changed = any(
        not torch.equal(first, last) for first, last in zip(initial.parameters(), trained.parameters(), strict=True)
    )
    checks = {
        "weights_changed": changed,
        "loss_decreased": after["train"] < before["train"] * 0.75,
        "finite_gradients": all(math.isfinite(row["grad_norm"]) for row in report["history"]),
        "nonzero_gradients": any(row["grad_norm"] > 0 for row in report["history"]),
        "finished_80_steps": payload["step"] == 80,
        "intermediate_40_steps": intermediate_payload["step"] == 40,
        "optimizer_saved": bool(intermediate_payload["optimizer"]["state"]),
        "cuda_rng_saved": device != "cuda" or bool(intermediate_payload["cuda_rng"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"訓練驗證未通過：{checks}")
    shutil.copyfile(data / "manifest.json", directory / "data-manifest.json")
    write_json(directory / "config.json", report["config"])
    write_json(directory / "tokenizer.json", ByteTokenizer().state())
    result = {
        "revision": revision,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": torch.__version__,
        "parameters": report["parameters"],
        "training_seconds": report["seconds"],
        "loss_before": before,
        "loss_after": after,
        "checkpoint_sha256": sha256(intermediate),
        "checks": checks,
        "scope": "課程合成資料的訓練流程驗證；不是自然語言能力評測",
    }
    write_json(directory / "training-check.json", result)
    return result


def resume_and_compare(directory, checkpoint, device="cuda"):
    import torch

    from tiny_perceptron.training import load_checkpoint

    directory = Path(directory)
    target = directory / "resumed-80.pt"
    run_training(
        [
            "--task",
            "text",
            "--train",
            "--device",
            device,
            "--checkpoint",
            str(checkpoint),
            "--resume",
            "--steps",
            "80",
            "--save-every",
            "40",
            "--batch-size",
            "4",
            "--lr",
            "0.003",
            "--data",
            str(directory / "data/toy-text/train.jsonl"),
            "--output",
            str(target),
        ]
    )
    baseline = load_checkpoint(directory / "baseline-80.pt")[1]
    resumed = load_checkpoint(target)[1]
    difference = max(float((baseline["model"][key] - value).abs().max()) for key, value in resumed["model"].items())
    matched = all(
        torch.allclose(baseline["model"][key], value, atol=1e-6, rtol=1e-5) for key, value in resumed["model"].items()
    )
    history = json.loads(target.with_suffix(".json").read_text())["history"]
    if resumed["step"] != 80 or history[0]["step"] != 41 or not matched:
        raise RuntimeError(f"續訓與不中斷訓練不一致：最大權重差={difference}")
    return {"resumed_from": 40, "resumed_to": 80, "matches_uninterrupted": matched, "max_weight_difference": difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu-smoke-local"))
    args = parser.parse_args()
    result = train_and_measure(args.output, args.device)
    result["resume"] = resume_and_compare(args.output, args.output / "checkpoint-40.pt", args.device)
    write_json(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
