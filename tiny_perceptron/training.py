"""訓練基本件；Notebook 預設只檢查一個 batch 的 forward/backward。"""

import random
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder


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


def save_checkpoint(path, model, optimizer=None, step=0, metadata=None, training_state=None):
    """保存完整訓練狀態；多模態版本也包含 encoder 與 projector。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    multimodal = isinstance(model, MultiModalLM)
    config = model.language.config if multimodal else model.config
    payload = {
        "format_version": "multimodal-v1" if multimodal else 1,
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "metadata": metadata or {},
        "training_state": training_state or {},
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "tokenizer": {
            "type": "byte" if config.vocab_size == 264 else "unspecified",
            "vocab_size": config.vocab_size,
        },
    }
    if multimodal:
        payload["modal_config"] = {
            "vision_width": model.vision.projection.out_features,
            "audio_width": model.audio.projection.out_features,
            "image_size": model.vision.image_size,
            "patch_size": model.vision.patch_size,
            "bands": model.audio.bands,
        }
        payload["task"] = payload["metadata"].get("task")
    # 寫完才換掉舊檔，避免中斷時只剩半個 checkpoint。
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint_rng(payload):
    """模型／teacher／optimizer 重建完成後才呼叫，避免建構吃掉 RNG 狀態。"""
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if torch.backends.mps.is_available() and payload.get("mps_rng") is not None:
        torch.mps.set_rng_state(payload["mps_rng"])


def load_checkpoint(path, device="cpu", restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (1, "quantized-v1", "multimodal-v1"):
        raise ValueError("不支援的 checkpoint 格式")
    model = TinyLM(ModelConfig(**payload["config"]))
    if payload["format_version"] == "multimodal-v1":
        c = payload["modal_config"]
        model = MultiModalLM(model, c["vision_width"], c["audio_width"])
        model.vision = VisionEncoder(c["vision_width"], c["image_size"], c["patch_size"])
        model.audio = AudioEncoder(c["bands"], c["audio_width"])
    if payload["format_version"] == "quantized-v1":
        from tiny_perceptron.quantization import replace_linear_layers

        if restore_rng:
            raise ValueError("量化 checkpoint 是參考推論格式，不能 resume 訓練")
        replace_linear_layers(model, payload["bits"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    if "trainable_parameters" in payload:
        names = set(payload["trainable_parameters"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in names)
    if restore_rng:
        restore_checkpoint_rng(payload)
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
