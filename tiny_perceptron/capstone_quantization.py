"""Packed capstone PTQ storage; reconstruction uses ordinary float32 operations.

Only Linear weights are quantized. Embeddings, norms, biases and every MoE router
remain float32. This is a file-size experiment, not an integer-kernel speed claim.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.capstone import DATA_VERSION, STAGES, TOK, CapstoneModel, digest, load_capstone
from tiny_perceptron.model import ModelConfig
from tiny_perceptron.quantization import pack_int4, quantize_symmetric, unpack_int4

FORMAT = "capstone-ptq-v1"


def quantizable_weights(model):
    return sorted(
        f"{name}.weight"
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.split(".")[-1] != "router"
    )


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return 0


def quantize_capstone(source, destination, bits=4):
    if bits not in (4, 8):
        raise ValueError("Capstone PTQ supports 4 or 8 bits")
    model, saved = load_capstone(source)
    if model.config.tied:
        raise ValueError("Capstone PTQ requires untied embedding and output weights")
    names = quantizable_weights(model)
    state = {name: value.detach().cpu().float().clone() for name, value in model.state_dict().items()}
    float_bytes = tensor_bytes(state)
    packed = {}
    for name in names:
        integers, scale = quantize_symmetric(state.pop(name), bits, per_channel=True)
        packed[name] = {
            "values": pack_int4(integers) if bits == 4 else integers,
            "scale": scale.float(),
            "shape": list(integers.shape),
        }
    metadata = {"source_checkpoint_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    if saved.get("inference_only") is True:
        metadata["public_source_sha256"] = metadata["source_checkpoint_sha256"]
    if isinstance(saved.get("metadata", {}).get("data_manifest"), dict):
        metadata["dataset_manifest_sha256"] = digest(saved["metadata"]["data_manifest"])
    elif "dataset_manifest_sha256" in saved.get("metadata", {}):
        metadata["dataset_manifest_sha256"] = saved["metadata"]["dataset_manifest_sha256"]
    branch = saved.get("metadata", {}).get("student_branch")
    if branch is not None:
        teacher = saved["metadata"].get("teacher_checkpoint_sha256")
        if (
            branch not in ("student-ce", "student-kd")
            or saved["stage"] != "joint"
            or model.config.experts != 0
            or not isinstance(teacher, str)
            or len(teacher) != 64
            or any(character not in "0123456789abcdef" for character in teacher)
        ):
            raise ValueError("Student PTQ requires explicit Dense branch and valid teacher provenance")
        metadata.update(student_branch=branch, teacher_checkpoint_sha256=teacher)
    payload = {
        "format_version": FORMAT,
        "inference_only": True,
        "config": asdict(model.config),
        "model": state,
        "quantized": packed,
        "quantization": {
            "bits": bits,
            "scheme": "symmetric-per-output-channel",
            "linear_weights": names,
            "router_dtype": "float32",
            "compute": "dequantize-to-float32",
        },
        "tokenizer": TOK.state(),
        "stage": saved["stage"],
        "step": saved["step"],
        "data_version": DATA_VERSION,
        "metadata": metadata,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    # Reload the serialized bytes, not the model kept in memory.
    loaded, _ = load_quantized_capstone(destination)
    with torch.no_grad():
        logits = loaded(torch.tensor([[TOK.bos_id, TOK.user_id, 8]]))["logits"]
    if not torch.isfinite(logits).all():
        raise ValueError("Reconstructed PTQ model has non-finite logits")
    return {
        "checkpoint": str(destination),
        "bits": bits,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": payload["metadata"]["source_checkpoint_sha256"],
        "file_bytes": destination.stat().st_size,
        "source_file_bytes": Path(source).stat().st_size,
        "tensor_bytes": tensor_bytes(state) + tensor_bytes(packed),
        "float_tensor_bytes": float_bytes,
        "quantized_linear_weights": names,
        "router_dtype": "float32",
        "compute": "dequantize-to-float32",
    }


def restore_quantized_payload(payload):
    if (
        payload.get("format_version") != FORMAT
        or payload.get("inference_only") is not True
        or payload.get("tokenizer") != TOK.state()
        or payload.get("data_version") != DATA_VERSION
        or payload.get("stage") not in STAGES
        or type(payload.get("step")) is not int
        or payload["step"] < 0
    ):
        raise ValueError("Unsupported capstone PTQ checkpoint/tokenizer/stage")
    if payload.get("config", {}).get("tied") is not False:
        raise ValueError("Capstone PTQ requires untied embedding and output weights")
    model = CapstoneModel(ModelConfig(**payload["config"]))
    names = quantizable_weights(model)
    specification = payload.get("quantization", {})
    bits = specification.get("bits")
    if bits not in (4, 8) or specification != {
        "bits": bits,
        "scheme": "symmetric-per-output-channel",
        "linear_weights": names,
        "router_dtype": "float32",
        "compute": "dequantize-to-float32",
    }:
        raise ValueError("Invalid PTQ scheme or quantized layers; routers must stay float32")
    original = model.state_dict()
    floats, packed = payload.get("model", {}), payload.get("quantized", {})
    if set(packed) != set(names) or set(floats) != set(original) - set(names):
        raise ValueError("PTQ tensor names do not match the architecture")
    state = {}
    for name, expected in original.items():
        if name in floats:
            tensor = floats[name]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.shape != expected.shape:
                raise ValueError("Unquantized PTQ tensors must match the float32 architecture")
            state[name] = tensor
        else:
            record = packed[name]
            if set(record) != {"values", "scale", "shape"} or record["shape"] != list(expected.shape):
                raise ValueError("Invalid packed tensor fields or shape")
            values, scale = record["values"], record["scale"]
            if (
                not isinstance(values, torch.Tensor)
                or not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float32
                or scale.shape != (expected.shape[0], 1)
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError("Packed scales must be positive finite per-output-channel float32 values")
            if bits == 4:
                if values.dtype != torch.uint8 or values.ndim != 1 or values.numel() != (expected.numel() + 1) // 2:
                    raise ValueError("int4 storage must contain exactly two nibbles per byte")
                if expected.numel() % 2 and int(values[-1] >> 4) != 8:
                    raise ValueError("int4 padding nibble must be zero")
                integers = unpack_int4(values, expected.shape)
            else:
                if values.dtype != torch.int8 or values.shape != expected.shape:
                    raise ValueError("int8 storage has the wrong dtype or shape")
                integers = values
            maximum = 2 ** (bits - 1) - 1
            if (integers.to(torch.int16).abs() > maximum).any():
                raise ValueError("Quantized values exceed the symmetric range")
            state[name] = integers.float() * scale
        if not torch.isfinite(state[name]).all():
            raise ValueError("PTQ weights must be finite")
    model.load_state_dict(state, strict=True)
    return model


def load_quantized_capstone(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return restore_quantized_payload(payload).to(device), payload
