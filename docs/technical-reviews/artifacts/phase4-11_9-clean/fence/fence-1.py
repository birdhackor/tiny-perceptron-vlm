from tiny_perceptron.multimodal import scene

image = scene("red", "square", offset=7)
crop = image[:, 4:12, 4:12]
print("原圖形狀", tuple(image.shape), "裁切形狀", tuple(crop.shape))
print("原圖紅色格", int((image[0] > 0).sum()))
print("裁切紅色格", int((crop[0] > 0).sum()))
