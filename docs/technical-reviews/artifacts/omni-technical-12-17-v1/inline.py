import math
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
rate, duration_ms, chunk_samples = 16000, 4000.0, 3200  # 3200 點＝200 ms
axis = torch.arange(32) - 15.5
y, x = torch.meshgrid(axis, axis, indexing="ij")
circle = ((x.square() + y.square()) <= 64).float()[None]
square = ((x.abs() <= 8) & (y.abs() <= 8)).float()[None]
frame_ms = torch.arange(16, dtype=torch.float64) * 250
frames = torch.stack([circle if t < 2000 else square for t in frame_ms])
wave = torch.zeros(12800, dtype=torch.float64)  # 800 ms 錄音
t = torch.arange(3200, dtype=torch.float64) / rate
wave[3200:6400] = 0.5 * torch.sin(2 * math.pi * 440 * t)

# 16 條 mel 三角頻帶，對上 400 點 STFT 的 201 個頻率格。
mel_edges = torch.linspace(0, 2595 * math.log10(1 + 8000 / 700), 18,
                           dtype=torch.float64)
hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
hz = torch.arange(201, dtype=torch.float64) * rate / 400
left, middle, right = hz_edges[:-2, None], hz_edges[1:-1, None], hz_edges[2:, None]
bank = torch.minimum((hz - left) / (middle - left),
                     (right - hz) / (right - middle)).clamp(min=0)

def local_tokens(waveform, start_ms):
    if waveform.ndim != 1 or len(waveform) == 0 or not torch.isfinite(waveform).all():
        raise ValueError("需要非空、有限的一維波形")
    end_ms = start_ms + len(waveform) / rate * 1000
    if not math.isfinite(start_ms) or start_ms < 0 or end_ms > duration_ms:
        raise ValueError("完整錄音須在共同時間軸內")
    result = []
    for begin in range(0, len(waveform), chunk_samples):
        raw = waveform[begin:begin + chunk_samples]
        energy = raw.square().mean()
        normalized = raw / energy.sqrt().clamp(min=1e-6)  # 只讀這一塊
        padded = F.pad(normalized, (0, max(0, 400 - len(raw))))
        power = torch.stft(padded, n_fft=400, hop_length=160,
                           window=torch.hann_window(400, dtype=waveform.dtype),
                           center=False, return_complex=True).abs().square()
        feature = (bank @ power).clamp(min=1e-8).log().mean(dim=1)
        support_start = start_ms + begin / rate * 1000
        support_end = start_ms + (begin + len(raw)) / rate * 1000
        result.append({"content": torch.cat([feature, energy[None]]),
                       "start": support_start, "end": support_end,
                       "anchor": (support_start + support_end) / 2})
    return result

def answer(images, waveform, start_ms):
    if images is None or waveform is None:
        return "不可判定"  # 刪除一路，就沒有足夠資訊回答相對關係
    if images.shape != (16, 1, 32, 32) or not torch.isfinite(images).all():
        raise ValueError("需要本例 16 張有限灰階影格")
    active = [token for token in local_tokens(waveform, start_ms)
              if token["content"][-1] > 0.01]
    if len(active) != 1:
        return "不可判定"
    event = active[0]
    index = int(torch.searchsorted(frame_ms, torch.tensor(event["anchor"]),
                                   right=True)) - 1
    image = images[index]
    if torch.equal(image, circle):
        return "圓"
    if torch.equal(image, square):
        return "方"
    return "不可判定"

a, b = local_tokens(wave, 1000.0), local_tokens(wave, 2000.0)
# 記錄 (種類, 內容)；時間另外附上，不用時間重新排列內容。
def serialize(tokens):
    return ([("video_begin", None)] + [("frame", image) for image in frames]
            + [("video_end", None), ("audio_begin", None)]
            + [("audio", token["content"]) for token in tokens] + [("audio_end", None)])

sequence_a, sequence_b = serialize(a), serialize(b)
causal_index = list(range(len(sequence_a)))
same_content = all(ka == kb and (va is None and vb is None or
                   va is not None and vb is not None and torch.equal(va, vb))
                   for (ka, va), (kb, vb) in zip(sequence_a, sequence_b, strict=True))
print("影格／波形固定；局部 token 內容相同", same_content)
print("序列項數與發聲塊的 causal index", len(sequence_a), causal_index[20])
print("A 的聲音錨點", [token["anchor"] for token in a])
print("B 的聲音錨點", [token["anchor"] for token in b])
predictions = [answer(frames, wave, offset) for offset in [1000.0, 2000.0]]
print("移時兩題", predictions, "整對正確", predictions == ["圓", "方"])
print("刪圖／刪聲", answer(None, wave, 1000.0), answer(frames, None, 1000.0))

changed = wave.clone()
changed[9600:] = 0.3  # 只改第 3 塊，不改發聲的第 1 塊
print("窗外擾動不改第 1 塊", torch.equal(
    a[1]["content"], local_tokens(changed, 1000.0)[1]["content"]))
short = local_tokens(wave[:100], 1000.0)[0]
print("短塊補零後的支持／錨點", short["start"], short["end"], short["anchor"])
