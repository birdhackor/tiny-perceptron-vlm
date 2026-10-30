"""從零初始化的局部影音分支；舊 A 不在此模組內更新。"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AVConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    representation: str = "time-rope"
    local_samples: int = 3200
    n_fft: int = 256
    hop: int = 80
    mel_bins: int = 24

    def validate(self):
        if self.representation not in {"additive", "time-rope"} or self.width % (2 * self.heads):
            raise ValueError("時間表示／head 維度不合法")
        if self.local_samples != 3200 or (self.n_fft, self.hop, self.mel_bins) != (256, 80, 24):
            raise ValueError("v1 固定獨立 200ms、center=False 的局部前處理")
        return self


def mel_bank(n_fft=256, bins=24, sample_rate=16000):
    def mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    points = torch.linspace(mel(0), mel(sample_rate / 2), bins + 2)
    hz = 700 * (10 ** (points / 2595) - 1)
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    up = (frequencies[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    down = (hz[2:, None] - frequencies[None]) / (hz[2:, None] - hz[1:-1, None])
    return torch.minimum(up, down).clamp_min(0)


def local_audio(waveform, config=AVConfig()):
    """每塊 STFT 只讀該塊；不使用全錄音統計，也不跨塊卷積。"""
    config.validate()
    if waveform.ndim != 1 or not len(waveform) or not torch.isfinite(waveform).all():
        raise ValueError("waveform 必須是一維有限且非空")
    count = math.ceil(len(waveform) / config.local_samples)
    padded = F.pad(waveform, (0, count * config.local_samples - len(waveform)))
    blocks = padded.reshape(count, config.local_samples)
    spectrum = torch.stft(
        blocks,
        n_fft=config.n_fft,
        hop_length=config.hop,
        window=torch.hann_window(config.n_fft, device=waveform.device),
        center=False,
        return_complex=True,
    )
    power = spectrum.abs().square() / config.n_fft
    mel = mel_bank(config.n_fft, config.mel_bins).to(waveform.device) @ power
    # 固定物理振幅尺度；padding 和窗外樣本不參與全局正規化。
    features = torch.log1p(mel).unsqueeze(1)
    starts = torch.arange(count, device=waveform.device) * 200.0
    ends = torch.minimum(starts + 200, torch.tensor(len(waveform) / 16, device=waveform.device))
    return features, starts, ends, (starts + ends) / 2


def collate_av(examples, device="cpu", config=AVConfig()):
    items = []
    for item, label in examples:
        item.validate()
        features, starts, ends, anchors = local_audio(item.waveform, config)
        items.append((item, label, features, starts, ends, anchors))
    batch, frames, blocks = len(items), max(len(x[0].frames) for x in items), max(len(x[2]) for x in items)
    audio_shape = items[0][2].shape[1:]
    image = torch.zeros(batch, frames, 1, 32, 32)
    audio = torch.zeros(batch, blocks, *audio_shape)
    times = torch.zeros(batch, frames + blocks)
    mask = torch.zeros(batch, frames + blocks, dtype=torch.bool)
    present = torch.zeros(batch, 2)
    support_start = torch.zeros(batch, blocks)
    support_end = torch.zeros(batch, blocks)
    labels = []
    for i, (item, label, features, starts, ends, anchors) in enumerate(items):
        ni, na = len(item.frames), len(features)
        image[i, :ni] = item.frames
        audio[i, :na] = features
        times[i, :ni] = item.frame_ms
        times[i, frames : frames + na] = anchors + item.audio_start_ms
        support_start[i, :na] = starts + item.audio_start_ms
        support_end[i, :na] = ends + item.audio_start_ms
        mask[i, :ni] = item.image_present
        mask[i, frames : frames + na] = item.audio_present
        present[i] = torch.tensor([item.image_present, item.audio_present])
        labels.append(label)
    return {
        "frames": image.to(device),
        "audio": audio.to(device),
        "time_s": times.to(device) / 1000,
        "mask": mask.to(device),
        "presence": present.to(device),
        "labels": torch.tensor(labels, device=device),
        "audio_support_start_ms": support_start.to(device),
        "audio_support_end_ms": support_end.to(device),
    }


class TimeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.width, config.width * 3)
        self.out = nn.Linear(config.width, config.width)
        frequencies = 2 ** torch.linspace(-1, 6, config.width // config.heads // 2)
        self.register_buffer("frequencies", frequencies)

    def rotate(self, x, times):
        angle = times[:, None, :, None] * self.frequencies[None, None, None, :]
        left, right = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (left * angle.cos() - right * angle.sin(), left * angle.sin() + right * angle.cos()), dim=-1
        ).flatten(-2)

    def forward(self, tokens, times, valid):
        b, length, width = tokens.shape
        q, k, v = (
            self.qkv(tokens)
            .reshape(b, length, 3, self.config.heads, width // self.config.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.config.representation == "time-rope":
            q, k = self.rotate(q, times), self.rotate(k, times)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.out(attended.transpose(1, 2).reshape(b, length, width))


class AVModel(nn.Module):
    def __init__(self, config=AVConfig()):
        super().__init__()
        self.config = config.validate()
        width = config.width
        self.image = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.audio = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.modality = nn.Embedding(3, width)
        self.readout = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                (
                    nn.LayerNorm(width),
                    TimeAttention(config),
                    nn.LayerNorm(width),
                    nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)),
                )
            )
            for _ in range(config.layers)
        )
        self.head = nn.Sequential(nn.LayerNorm(width + 2), nn.Linear(width + 2, 3))

    def forward(self, batch):
        frames, audio = batch["frames"], batch["audio"]
        b, ni, na = len(frames), frames.shape[1], audio.shape[1]
        image = self.image(frames.flatten(0, 1)).reshape(b, ni, -1) + self.modality.weight[0]
        sound = self.audio(audio.flatten(0, 1)).reshape(b, na, -1) + self.modality.weight[1]
        tokens = torch.cat((self.readout.expand(b, -1, -1) + self.modality.weight[2], image, sound), dim=1)
        times = F.pad(batch["time_s"], (1, 0))
        valid = F.pad(batch["mask"], (1, 0), value=True)
        if self.config.representation == "additive":
            # 兩種表示使用同一個 per-head 頻率集合；加法版只跨 heads 重複，
            # 不額外給它 width/2 個獨特頻率當作另一個變因。
            frequencies = self.blocks[0][1].frequencies.repeat(self.config.heads)
            angle = times[..., None] * frequencies
            tokens = tokens + torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)
        for norm1, attention, norm2, ffn in self.blocks:
            tokens = tokens + attention(norm1(tokens), times, valid)
            tokens = tokens + ffn(norm2(tokens))
        return self.head(torch.cat((tokens[:, 0], batch["presence"]), dim=-1))

    def config_sha256(self):
        return hashlib.sha256(json.dumps(self.config.__dict__, sort_keys=True).encode()).hexdigest()
