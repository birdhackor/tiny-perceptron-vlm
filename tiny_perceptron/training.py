"""訓練基本件；Notebook 預設只檢查一個 batch 的 forward/backward。"""

import random
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer=None, step=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "metadata": metadata or {},
        "tokenizer": {
            "type": "byte" if model.config.vocab_size == 264 else "unspecified",
            "vocab_size": model.config.vocab_size,
        },
    }
    torch.save(payload, path)


def load_checkpoint(path, device="cpu", restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (1, "quantized-v1"):
        raise ValueError("不支援的 checkpoint 格式")
    model = TinyLM(ModelConfig(**payload["config"]))
    if payload["format_version"] == "quantized-v1":
        from tiny_perceptron.quantization import replace_linear_layers

        if restore_rng:
            raise ValueError("量化 checkpoint 是參考推論格式，不能 resume 訓練")
        replace_linear_layers(model, payload["bits"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    if restore_rng:
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return model, payload


def learning_rate(step, total, peak=0.001, warmup=10):
    import math

    if total < 1 or peak <= 0:
        raise ValueError("total/peak 必須為正")
    warmup = min(warmup, max(1, total // 4))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
