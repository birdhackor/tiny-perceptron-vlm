import torch
from tiny_perceptron.multimodal import scene, patchify, unpatchify

image = scene("red", "square")[None]
patches = patchify(image, 4)
restored = unpatchify(patches, 3, 16, 16, 4)
print("小塊序列", tuple(patches.shape))
print("拼回相同", torch.equal(image, restored))
