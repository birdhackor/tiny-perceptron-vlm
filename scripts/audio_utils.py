"""讀取單聲道音訊與明確重採樣；不把取樣率標籤當成實際轉換。"""

import math

import numpy as np
import soundfile as sf
import torch


def resample_waveform(values, source_rate, target_rate=16000, radius=16):
    """加窗 sinc 插值；降採樣時先在 kernel 中縮小通帶，分塊限制記憶體。"""
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("重採樣需要非空且有限的單聲道波形")
    if source_rate < 1 or target_rate < 1 or radius < 1:
        raise ValueError("source_rate、target_rate、radius 必須為正")
    if source_rate == target_rate:
        return values.astype("float32", copy=True)
    count = round(len(values) * target_rate / source_rate)
    if count < 1:
        raise ValueError("音訊太短，目標取樣率無法保留一個測量點")
    cutoff = min(1.0, target_rate / source_rate)
    half_width = math.ceil(radius / cutoff)
    offsets = torch.arange(-half_width + 1, half_width + 1)
    source = torch.from_numpy(values.astype("float64"))
    chunks = []
    for start in range(0, count, 4096):
        times = torch.arange(start, min(start + 4096, count), dtype=torch.float64) * source_rate / target_rate
        indices = times.floor().long()[:, None] + offsets[None]
        distance = (times[:, None] - indices) * cutoff
        valid = (indices >= 0) & (indices < len(values))
        window = torch.where(distance.abs() < radius, 0.5 + 0.5 * torch.cos(math.pi * distance / radius), 0)
        kernel = torch.sinc(distance) * window * valid
        samples = source[indices.clamp(0, len(values) - 1)]
        chunks.append((kernel * samples).sum(-1) / kernel.sum(-1).clamp(min=1e-12))
    result = torch.cat(chunks).numpy().astype("float32")
    if not np.isfinite(result).all():
        raise ValueError("重採樣結果含非有限振幅")
    return result


def load_mono_audio(path, resample=False, target_rate=16000):
    """整数 PCM 由 libsndfile 解成 [-1,1] 浮點；不自動混音、截波或重採樣。"""
    info = sf.info(path)
    values, source_rate = sf.read(path, dtype="float32")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("音訊必須是非空單聲道；請先將來源明確轉成 mono")
    if not np.isfinite(values).all():
        raise ValueError("音訊含 NaN 或 Infinity；請先修正來源成有限振幅")
    if (np.abs(values) > 1).any():
        raise ValueError("來源音訊振幅需位於 [-1,1]；請明確正規化，程式不會偷偷截波")
    if source_rate != target_rate and not resample:
        raise ValueError(
            f"這個課堂 encoder 需真正16kHz mono，來源是 {source_rate} Hz；"
            "請加 --resample-audio 實際重採樣，不只改取樣率標籤"
        )
    source_count = len(values)
    changed = source_rate != target_rate
    if changed:
        values = resample_waveform(values, source_rate, target_rate)
    metadata = {
        "type": "file",
        "path": str(path),
        "subtype": info.subtype,
        "sample_rate": target_rate,
        "source_sample_rate": source_rate,
        "source_samples": source_count,
        "effective_samples": len(values),
        "source_duration_seconds": source_count / source_rate,
        "duration_seconds": len(values) / target_rate,
        "resampled": changed,
        "resampling_method": "windowed-sinc, radius 16, cutoff min(1,target/source)" if changed else None,
        "amplitude_min": float(values.min()),
        "amplitude_max": float(values.max()),
        "normalization": "integer PCM decoded to normalized floating point; no extra gain adjustment or clipping",
    }
    return values, metadata
