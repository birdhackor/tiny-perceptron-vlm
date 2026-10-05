## 10.3 每塊 48 個像素值，如何變成 8 個特徵？

上一節每塊有 48 個像素值，接下來希望每個位置用 8 個數字表示。可以用同一張加權配方表，對每塊的 48 個值乘權重、加總，再加偏置，產生 8 個輸出。這一排數字叫向量；配方表則是可以訓練的線性層。

```python
import torch
from torch import nn
from tiny_perceptron.multimodal import scene, patchify

