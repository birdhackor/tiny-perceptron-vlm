from tiny_perceptron.multimodal import scene

image = scene("red", "square")
print("形狀", tuple(image.shape))
print("中央RGB", image[:, 8, 8].tolist())
print("左上RGB", image[:, 0, 0].tolist())
print("加入批次軸", tuple(image.unsqueeze(0).shape))
