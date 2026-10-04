"""把課程 LoRA adapter 配回同一份基底；先完整核對，再植入明示的 Linear。"""

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.model import TinyLM


def base_state_sha256(state):
    """與 behavior._state_digest 相同：名字、shape、原始 FP32 bytes，沒有檔案封裝。"""
    if not isinstance(state, dict) or not state:
        raise ValueError("LoRA 基底 state 必須是非空的 FP32 tensor dict")
    if not all(isinstance(name, str) for name in state):
        raise ValueError("LoRA 基底 tensor 名字必須是文字")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided:
            raise ValueError(f"LoRA 基底 {name} 必須保留訓練時的原始 FP32 tensor")
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _config_matches(actual, expected):
    return (
        isinstance(actual, dict)
        and actual == expected
        and all(type(actual[key]) is type(value) for key, value in expected.items())
    )


def _matrix(value, shape, name, base):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"LoRA {name} 必須是 shape={shape} 的浮點矩陣")
    if not torch.isfinite(value).all():
        raise ValueError(f"LoRA {name} 含有非有限數值")
    converted = value.detach().to(device=base.weight.device, dtype=base.weight.dtype)
    if not torch.isfinite(converted).all():
        raise ValueError(f"LoRA {name} 轉成基底 dtype={base.weight.dtype} 後溢位")
    return converted


def load_lora_adapter(model, path, *, base_state=None):
    """就地植入 lora-v1，回傳可加入 CLI JSON 的來源與縮放資訊。

    基底 hash 必須使用原始 FP32 state。若 caller 已轉換模型 dtype，請傳入
    native checkpoint 的 model state；實際模型還會逐張量核對，不能拿別份
    checkpoint 的 state 代替目前基底。基底權重保持原值，僅新增 A/B 分支。
    """
    if not isinstance(model, TinyLM):
        raise ValueError("文字 LoRA adapter 需要 native TinyLM 基底")
    if any(isinstance(layer, LoRALinear) for layer in model.modules()):
        raise ValueError("基底已含 LoRA；切換 adapter 請重新載入同一份 native checkpoint")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != "lora-v1":
        raise ValueError("adapter 必須是正式的 lora-v1 權重檔")
    if not _config_matches(payload.get("config"), asdict(model.config)):
        raise ValueError("LoRA adapter config 與基底 config 必須完全匹配")
    if payload.get("scaling") != "alpha/rank":
        raise ValueError("LoRA adapter scaling 必須明示 alpha/rank")
    state = model.state_dict() if base_state is None else base_state
    digest = base_state_sha256(state)
    if payload.get("base_sha256") != digest:
        raise ValueError("LoRA base_sha256 不匹配；請使用訓練 adapter 時的同一份基底")
    current = model.state_dict()
    if current.keys() != state.keys():
        raise ValueError("實際 LoRA 基底 tensor 名字與提供的 FP32 state 不匹配")
    for name, value in current.items():
        original = state[name]
        if value.shape != original.shape or not torch.equal(
            value.detach(), original.to(device=value.device, dtype=value.dtype)
        ):
            raise ValueError(f"實際 LoRA 基底 {name} 與已核對的 FP32 state 不匹配")
    adapter = payload.get("adapter")
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError("LoRA adapter 必須明示至少一個 module path 與 A/B 權重")
    prepared = []
    for name, entry in adapter.items():
        if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError(f"LoRA module path 不合法：{name!r}")
        try:
            base = model.get_submodule(name)
        except (AttributeError, TypeError) as error:
            raise ValueError(f"未知的 LoRA module path：{name}") from error
        if not isinstance(base, nn.Linear):
            raise ValueError(f"LoRA module {name} 必須是 Linear，實際為 {type(base).__name__}")
        if not isinstance(entry, dict) or set(entry) != {"a", "b", "rank", "alpha"}:
            raise ValueError(f"LoRA {name} 必須明示 a、b、rank、alpha")
        rank, alpha = entry["rank"], entry["alpha"]
        if type(rank) is not int or rank < 1:
            raise ValueError(f"LoRA {name} rank 必須是正整數")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"LoRA {name} alpha 必須是有限的正數")
        for field, expected in (("rank", rank), ("alpha", alpha)):
            if field in payload and (payload[field] != expected or type(payload[field]) is not type(expected)):
                raise ValueError(f"LoRA {name} {field} 與頂層設定不匹配")
        a = _matrix(entry["a"], (rank, base.in_features), f"{name}.a", base)
        b = _matrix(entry["b"], (base.out_features, rank), f"{name}.b", base)
        prepared.append((name, base, rank, alpha, a, b))
    # 所有 path／矩陣先通過檢查，錯誤 adapter 不會留下植入一半的模型。
    information = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for name, base, rank, alpha, a, b in prepared:
            layer = LoRALinear(base, rank=rank, alpha=alpha).to(device=base.weight.device, dtype=base.weight.dtype)
            layer.a.copy_(a)
            layer.b.copy_(b)
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, layer)
            information.append({"path": name, "rank": rank, "alpha": alpha, "scale": alpha / rank})
    return {
        "path": str(path),
        "format_version": "lora-v1",
        "base_sha256": digest,
        "base_sha256_verified": True,
        "scaling": "alpha/rank",
        "modules": information,
    }
