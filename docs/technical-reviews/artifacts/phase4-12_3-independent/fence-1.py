import torch
from tiny_perceptron.multimodal import tone

wave = tone()
frames = wave.unfold(0, 400, 160)
print("框形狀", tuple(frames.shape))
print("框秒數", 400 / 16000, "移動秒數", 160 / 16000)
print("重疊相同", torch.equal(frames[0, 160:], frames[1, :240]))
