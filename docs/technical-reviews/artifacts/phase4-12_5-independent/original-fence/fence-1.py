import torch
from tiny_perceptron.multimodal import tone

wave = tone(440.0, seconds=0.2)
spectrum = torch.stft(wave, 400, 160, window=torch.hann_window(400), return_complex=True)
power = spectrum.abs().square()
peak = power.mean(-1).argmax().item()
print("頻率格、時間框", tuple(power.shape))
print("最高格編號", peak, "對應Hz", peak * 16000 / 400)
