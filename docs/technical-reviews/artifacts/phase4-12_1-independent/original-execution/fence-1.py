from tiny_perceptron.multimodal import tone

wave = tone(440.0)
print("樣本形狀", tuple(wave.shape))
print("前三個值", [round(v, 3) for v in wave[:3].tolist()])
print("最小最大", round(wave.min().item(), 2), round(wave.max().item(), 2))
