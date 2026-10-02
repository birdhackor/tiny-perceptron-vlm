"""不用外部編碼器的圖片／聲音入口；第10–12章逐步拆開讀。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE


def patchify(images, patch_size=4):
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError("圖片高寬必須能整除 patch_size")
    p = patch_size
    return images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * p * p)


def unpatchify(patches, channels, height, width, patch_size=4):
    b, _, _ = patches.shape
    p = patch_size
    return (
        patches.reshape(b, height // p, width // p, channels, p, p)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, channels, height, width)
    )


class VisionEncoder(nn.Module):
    def __init__(self, width=16, image_size=16, patch_size=4):
        super().__init__()
        self.patch_size, self.image_size = patch_size, image_size
        self.projection = nn.Linear(3 * patch_size**2, width)
        self.position = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, width) * 0.02)
        self.feature = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())

    def forward(self, images):
        if images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("先把圖片轉成這個 encoder 配置的 RGB 高寬")
        return self.feature(self.projection(patchify(images, self.patch_size)) + self.position)


def tone(frequency=440.0, seconds=0.1, sample_rate=16000):
    times = torch.arange(round(seconds * sample_rate)) / sample_rate
    return 0.5 * torch.sin(2 * math.pi * frequency * times)


def mel_filter_bank(sample_rate=16000, n_fft=400, bands=16, device=None):
    """三角濾波器：先在 mel 軸等距，再換回 Hz 頻率軸。"""
    if bands < 1 or sample_rate < 1:
        raise ValueError("bands/sample_rate 必須為正")
    max_mel = 2595 * math.log10(1 + (sample_rate / 2) / 700)
    mels = torch.linspace(0, max_mel, bands + 2, device=device)
    hz = 700 * (10 ** (mels / 2595) - 1)
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device)
    left = (frequencies[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    right = (hz[2:, None] - frequencies[None]) / (hz[2:] - hz[1:-1])[:, None]
    return torch.minimum(left, right).clamp(min=0)


def log_mel(waveform, sample_rate=16000, n_fft=400, hop_length=160, bands=16):
    if waveform.shape[-1] <= n_fft // 2:
        waveform = F.pad(waveform, (0, n_fft // 2 + 1 - waveform.shape[-1]))
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    power = spectrum.abs().square()
    bank = mel_filter_bank(sample_rate, n_fft, bands, waveform.device).to(power.dtype)
    return (bank @ power).clamp(min=1e-8).log()


class AudioEncoder(nn.Module):
    def __init__(self, bands=16, width=16):
        super().__init__()
        self.bands = bands
        self.projection = nn.Linear(bands, width)
        self.feature = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())

    def forward(self, waveform):
        features = log_mel(waveform, bands=self.bands).transpose(-1, -2)
        return self.feature(self.projection(features))


def expand_modalities(ids, labels, embedding, features, marker_ids):
    """一筆無 padding 資料：marker 展開成多向量，新增位置不直接算 loss。

    ids 與 labels 在此都尚未 shift。展開後才 shift，第一個答案的對齊才正確。
    """
    pieces, targets = [], []
    for token, label in zip(ids.tolist(), labels.tolist(), strict=True):
        if token in marker_ids:
            modal = features[token]
            pieces.append(modal)
            targets += [IGNORE] * len(modal)
        else:
            pieces.append(embedding(torch.tensor([token], device=embedding.weight.device)))
            targets.append(label)
    expanded = torch.cat(pieces)
    target = torch.tensor(targets, device=expanded.device, dtype=torch.long)
    return expanded[:-1].unsqueeze(0), target[1:].unsqueeze(0)


class MultiModalLM(nn.Module):
    def __init__(self, language_model, vision_width=16, audio_width=16):
        super().__init__()
        self.language = language_model
        self.vision = VisionEncoder(vision_width)
        self.audio = AudioEncoder(width=audio_width)
        self.image_projector = nn.Linear(vision_width, language_model.config.width)
        self.audio_projector = nn.Linear(audio_width, language_model.config.width)

    def forward(self, ids, labels=None, *, image=None, waveform=None):
        from tiny_perceptron.data import ByteTokenizer

        tok = ByteTokenizer()
        if image is None and waveform is None:
            if ((ids == tok.image_id) | (ids == tok.audio_id)).any():
                raise ValueError("placeholder 缺少配對圖片／聲音")
            return self.language(ids.unsqueeze(0))
        features = {}
        if image is not None:
            features[tok.image_id] = self.image_projector(self.vision(image.unsqueeze(0)))[0]
        if waveform is not None:
            features[tok.audio_id] = self.audio_projector(self.audio(waveform.unsqueeze(0)))[0]
        if labels is None:
            labels = torch.full_like(ids, IGNORE)
        # 未提供模態的 placeholder 要報錯，不能默默當普通文字。
        markers = {tok.image_id, tok.audio_id}
        for token in ids.tolist():
            if token in markers and token not in features:
                raise ValueError("placeholder 缺少配對圖片／聲音")
        x, y = expand_modalities(ids, labels, self.language.embedding, features, markers)
        result = self.language(embeddings=x)
        result["labels"] = y
        return result


@torch.no_grad()
def generate_modal(model, prefix, image=None, waveform=None, max_new_tokens=16, temperature=0.0):
    """先用清楚的完整重算路徑；最後加dummy，讓forward保留整個prompt。"""
    from tiny_perceptron.data import ByteTokenizer

    tok = ByteTokenizer()
    was_training = model.training
    model.eval()
    ids = prefix.clone()
    try:
        for _ in range(max_new_tokens):
            expanded_length = len(ids)
            if image is not None:
                count = (ids == tok.image_id).sum().item()
                expanded_length += count * ((model.vision.image_size // model.vision.patch_size) ** 2 - 1)
            if waveform is not None:
                count = (ids == tok.audio_id).sum().item()
                expanded_length += count * (log_mel(waveform).shape[-1] - 1)
            if expanded_length > model.language.config.max_length:
                if len(ids) == len(prefix):
                    raise ValueError("模態展開後的 prompt 超過 max_length")
                break
            padded = torch.cat((ids, ids.new_tensor([tok.pad_id])))
            # 多模態forward約定未shift完整序列；dummy只為取得最後prompt的logits。
            output = model(padded, image=image, waveform=waveform)
            scores = output["logits"][0, -1]
            next_id = (
                scores.argmax() if temperature <= 0 else torch.multinomial((scores / temperature).softmax(-1), 1)[0]
            )
            ids = torch.cat((ids, next_id.reshape(1)))
            # 生成的 image/audio 是非法答案 token，不是新的輸入素材。
            # 保留它供評估記失敗，但不能在下一步把它重新展開成圖片／聲音。
            if next_id.item() in (tok.eos_id, tok.image_id, tok.audio_id):
                break
    finally:
        model.train(was_training)
    return ids


def scene(color="red", shape="square", size=16, offset=0):
    """固定規則的合成圖片；屬性 label 是生成規則，不是模型猜測。"""
    colors = {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0]}
    image = torch.zeros(3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    center = size // 2 + offset
    region = (xx - center).abs() <= size // 4
    region &= (yy - size // 2).abs() <= size // 4
    if shape == "circle":
        region = (xx - center).square() + (yy - size // 2).square() <= (size // 4) ** 2
    elif shape != "square":
        raise ValueError("合成形狀只接受 circle/square")
    image[:, region] = torch.tensor(colors[color])[:, None]
    return image
