import json
import sys
import torch
from tiny_perceptron.multimodal import scene

torch.set_num_threads(1)
print('environment', json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu'}))
image = scene('red', 'square')
print('形狀', tuple(image.shape))
print('中央RGB', image[:, 8, 8].tolist())
print('左上RGB', image[:, 0, 0].tolist())
print('加入批次軸', tuple(image.unsqueeze(0).shape))
other = scene('green', 'square')
print('variation green 中央RGB', other[:, 8, 8].tolist())
