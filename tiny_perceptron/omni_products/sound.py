"""隨機初始化 VQ codec 與有限聲音程式 Talker；不是通用 TTS。"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .data import validate_program


@dataclass(frozen=True)
class CodecConfig:
    hop: int = 80
    width: int = 32
    codes: int = 128
    hidden: int = 128
    beta: float = 0.25
    sample_rate: int = 8000

    def validate(self):
        if self.sample_rate != 8000 or self.hop not in (40, 80) or self.codes not in (32, 64, 128, 256):
            raise ValueError("codec profile 不在事先公開的候選集合")
        if self.width not in (16, 32, 64) or self.beta != 0.25:
            raise ValueError("latent width／commitment beta 不合法")
        return self


class SoundCodec(nn.Module):
    """每個 block 零 lookahead；最後 padding 不進 loss／碼本統計。"""

    def __init__(self, config=CodecConfig()):
        super().__init__()
        self.config = config.validate()
        self.encoder = nn.Sequential(
            nn.Linear(config.hop, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.width)
        )
        self.codebook = nn.Embedding(config.codes, config.width)
        nn.init.normal_(self.codebook.weight, std=0.15)
        self.decoder = nn.Sequential(
            nn.Linear(config.width, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.hop), nn.Tanh()
        )

    def quantize(self, encoded):
        weights = self.codebook.weight
        distances = encoded.square().sum(-1, keepdim=True) + weights.square().sum(-1) - 2 * encoded @ weights.T
        codes = distances.argmin(-1)
        return self.codebook(codes), codes

    def forward(self, blocks, valid=None):
        if blocks.shape[-1] != self.config.hop:
            raise ValueError("codec blocks 的 hop 不符")
        if valid is None:
            valid = torch.ones_like(blocks, dtype=torch.bool)
        if valid.shape != blocks.shape or valid.dtype != torch.bool:
            raise ValueError("sample padding mask 不符")
        encoded = self.encoder(torch.where(valid, blocks, 0))
        quantized, codes = self.quantize(encoded)
        straight_through = encoded + (quantized - encoded).detach()
        reconstructed = self.decoder(straight_through)
        nonpadding = valid.any(-1)
        if not nonpadding.any() or not valid.any():
            raise ValueError("全 padding 不應進 codec loss")
        reconstruction = (reconstructed[valid] - blocks[valid]).square().mean()
        codebook_loss = (quantized - encoded.detach()).square()[nonpadding].mean()
        commitment = (encoded - quantized.detach()).square()[nonpadding].mean()
        return {
            "waveform": reconstructed,
            "codes": torch.where(nonpadding, codes, -1),
            "loss": reconstruction + codebook_loss + self.config.beta * commitment,
            "reconstruction_loss": reconstruction,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment,
            "encoded": encoded[nonpadding],
        }

    @torch.no_grad()
    def encode_wave(self, waveform):
        length, hop = len(waveform), self.config.hop
        if waveform.ndim != 1 or not length or not torch.isfinite(waveform).all():
            raise ValueError("waveform 非法")
        padded = F.pad(waveform, (0, (-length) % hop))
        _, codes = self.quantize(self.encoder(padded.reshape(-1, hop)))
        return codes, length

    def decode_codes(self, codes):
        if codes.ndim != 1 or not len(codes) or codes.min() < 0 or codes.max() >= self.config.codes:
            raise ValueError("codec code 超限或為空")
        # 逐塊與整段走完全相同的矩陣形狀／次序，CPU 與 GPU 皆可逐樣本對照。
        return torch.cat([self.decoder(self.codebook(code)).reshape(-1) for code in codes])

    def chunks(self, codes):
        for code in codes:
            yield self.decoder(self.codebook(code)).reshape(-1)

    @torch.no_grad()
    def reset_unused_training_codes(self, usage, encoded, generator=None):
        if not self.training:
            raise ValueError("eval／final 禁止重啟碼本")
        unused = (usage == 0).nonzero().flatten()
        if len(unused):
            flat = encoded.detach().reshape(-1, self.config.width)
            positions = torch.randint(len(flat), (len(unused),), generator=generator, device=flat.device)
            self.codebook.weight[unused] = flat[positions]
        return int(len(unused))


def program_schedule(program, hop=80, max_blocks=None):
    """只由公開輸入程式推導的時程，沒有參考波形、codec 或參考 codes。

    每列是 requested frequency／segment 內 block 位置／事件起點。
    最後 END 列是輸入所要求的片長特徵；Talker 仍須自己預測 EOS。
    """
    validate_program(program)
    if hop not in (40, 80):
        raise ValueError("Talker 公開時程僅支援已定義的 hop40/80")
    max_blocks = 16000 // hop if max_blocks is None else max_blocks
    rows = []
    frequencies = {0: 0, 220: 1, 440: 2, 880: 3}
    for event in program:
        for position in range(event.duration_ms * 8 // hop):
            rows.append((frequencies[event.frequency], position, int(position == 0)))
        for position in range(event.rest_ms * 8 // hop):
            rows.append((0, position, int(position == 0)))
    if len(rows) > max_blocks:
        raise ValueError("聲音程式超過指定 blocks 上限")
    rows.append((4, 0, 1))
    return torch.tensor(rows, dtype=torch.long)


class ProgramTalker(nn.Module):
    def __init__(self, codes=128, width=96, hop=80):
        super().__init__()
        if hop not in (40, 80):
            raise ValueError("Talker hop 不合法")
        self.codes, self.width, self.hop = codes, width, hop
        self.max_blocks = 16000 // hop
        self.bos, self.eos, self.pad = codes, codes + 1, codes + 2
        self.code_embedding = nn.Embedding(codes + 3, width)
        self.frequency_embedding = nn.Embedding(5, width)
        self.position_embedding = nn.Embedding(2400 // hop, width)
        self.onset_embedding = nn.Embedding(2, width)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, codes + 2)

    def inputs(self, previous, schedule):
        return (
            self.code_embedding(previous)
            + self.frequency_embedding(schedule[..., 0])
            + self.position_embedding(schedule[..., 1])
            + self.onset_embedding(schedule[..., 2])
        )

    def forward(self, previous, schedule):
        hidden, _ = self.recurrent(self.inputs(previous, schedule))
        return self.output(hidden)

    @torch.no_grad()
    def generate(self, program, max_blocks=None):
        """僅 BOS 起始，之後只讀自己的預測；不收 reference codes／length。"""
        device = self.code_embedding.weight.device
        max_blocks = self.max_blocks if max_blocks is None else max_blocks
        if type(max_blocks) is not int or not 1 <= max_blocks <= self.max_blocks:
            raise ValueError("生成 blocks 上限非法")
        schedule = program_schedule(program, self.hop).to(device)
        previous = torch.tensor([[self.bos]], device=device)
        hidden, emitted = None, []
        for step in range(max_blocks + 1):
            # 超過公開意圖片長仍讀 END；不直接用它終止，不補／截成 gold 長度。
            row = schedule[min(step, len(schedule) - 1)].reshape(1, 1, 3)
            value, hidden = self.recurrent(self.inputs(previous, row), hidden)
            predicted = self.output(value[:, -1]).argmax(-1).item()
            if predicted == self.eos:
                return {"codes": torch.tensor(emitted, device=device, dtype=torch.long), "eos": True, "capped": False}
            if predicted == self.bos or len(emitted) == max_blocks:
                return {"codes": torch.tensor(emitted, device=device, dtype=torch.long), "eos": False, "capped": True}
            emitted.append(predicted)
            previous = torch.tensor([[predicted]], device=device)
        raise RuntimeError("unreachable")

    def config(self):
        return {
            "codes": self.codes,
            "width": self.width,
            "hop": self.hop,
            "sample_rate": 8000,
            "max_blocks": self.max_blocks,
            "conditioning": "public-input-derived frequency/elapsed/onset/END schedule; no reference codes or forced stopping length",
        }


def codec_spec(codec):
    return {
        **asdict(codec.config),
        "lookahead_samples": 0,
        "state": "每塊無跨塊 decoder state",
        "flush": "只裁最後原始有效 samples；生成無參考長度裁切",
        "bitrate_bps": codec.config.sample_rate / codec.config.hop * (codec.config.codes.bit_length() - 1),
    }
