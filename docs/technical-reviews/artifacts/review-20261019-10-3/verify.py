import hashlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.multimodal import VisionEncoder, patchify, scene


torch.set_num_threads(1)
torch.set_default_device("cpu")
root = Path(__file__).resolve().parents[4]
chapter = (root / "course/chapters/10.md").read_bytes()
match = re.search(rb"(?ms)^## 10\.3 .*?(?=^## |\Z)", chapter)
frozen = (root / "docs/course-review-20261019/freeze-01/sources/10.3.md").read_bytes()
assert match is not None and match[0] == frozen, "現行小節與凍結原文必須相同"

image = scene()
assert torch.equal(image, scene("red", "square")), "預設圖必須與明確指定的紅方塊相同"
assert image[:, 8, 8].tolist() == [1.0, 0.0, 0.0], "中央必須是紅色"
assert image[:, 0, 0].tolist() == [0.0, 0.0, 0.0], "角落必須是黑色"
patches = patchify(image[None], 4)
assert patches.shape == (1, 16, 48), "切塊形狀必須符合通道與塊大小"

torch.manual_seed(0)
layer = nn.Linear(48, 8)
features = layer(patches)
weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
assert features.shape == (1, 16, 8), "線性層必須只替換最後一軸"
assert layer.weight.shape == (8, 48) and layer.bias.shape == (8,), "每個輸出必須有自己的權重列與偏移"
manual = patches.double() @ weight.double().T + bias.double()
error = (features.double() - manual).abs().max().item()
assert error < 1e-7, "線性層必須符合逐列加權和與偏移"
per_position = torch.stack([layer(patches[:, index]) for index in range(16)], dim=1)
assert torch.allclose(features, per_position, atol=1e-7, rtol=1e-6), "每個位置必須共用同一個線性層"
reverse = torch.arange(15, -1, -1)
assert torch.allclose(layer(patches[:, reverse]), features[:, reverse], atol=1e-7, rtol=1e-6), "交換位置必須只交換對應的輸出"
loss = features.square().sum()
loss.backward()
assert layer.weight.requires_grad and layer.bias.requires_grad, "權重與偏移必須可求導"
assert layer.weight.grad.norm().item() > 0 and layer.bias.grad.norm().item() > 0, "此輸入與代價必須能把梯度傳到參數"

torch.manual_seed(0)
repeated = nn.Linear(48, 8)
assert torch.equal(weight, repeated.weight) and torch.equal(bias, repeated.bias), "相同環境重設種子後初始化必須相同"
assert torch.equal(features.detach(), repeated(patches)), "相同初始化的計算必須可重做"

encoder = VisionEncoder(width=8)
events = []
projection_inputs = []


def capture_projection(module, args):
    events.append("projection")
    projection_inputs.append(args[0].detach().clone())


handles = [
    encoder.projection.register_forward_pre_hook(capture_projection),
    encoder.feature.register_forward_pre_hook(lambda module, args: events.append("feature")),
]
encoded = encoder(image[None])
for handle in handles:
    handle.remove()
assert events == ["projection", "feature"], "圖片入口必須先算投影再算後續特徵"
assert torch.equal(projection_inputs[0], patches), "投影輸入必須是切塊結果"
assert encoder.projection.in_features == 48 and encoder.projection.out_features == 8, "投影設定必須符合切塊寬度"
assert torch.equal(encoded, encoder.feature(encoder.projection(patches) + encoder.position)), "圖片入口必須符合實作的運算順序"

table = nn.Embedding(4, 3)
ids = torch.tensor([[2, 1, 2]])
assert torch.equal(table(ids), table.weight[ids]), "文字 embedding 必須依索引查出權重表的列"

small = nn.Linear(2, 1)
with torch.no_grad():
    small.weight.copy_(torch.tensor([[0.5, 0.5]]))
    small.bias.zero_()
collision = small(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
assert torch.equal(collision, torch.tensor([[0.5], [0.5]])), "不同輸入在指定配方下必須同得零點五"

selection = nn.Linear(48, 8)
with torch.no_grad():
    selection.weight.zero_()
    selection.weight[:, :8].copy_(torch.eye(8))
    selection.bias.zero_()
different_patches = torch.zeros(2, 48)
different_patches[1, 8] = 1.0
selection_output = selection(different_patches)
assert not torch.equal(different_patches[0], different_patches[1]), "壓縮反例必須使用不同輸入"
assert torch.equal(selection_output[0], selection_output[1]), "壓縮反例必須抹掉一項差異"

exercise = nn.Linear(48, 16)
exercise_features = exercise(patches)
assert exercise_features.shape == (1, 16, 16), "練習特徵必須保留位置數並改成指定寬度"
assert exercise.weight.shape == (16, 48), "練習矩陣形狀必須符合輸出與輸入寬度"

local_apis = {}
for name, obj in [("Linear", nn.Linear), ("Embedding.forward", nn.Embedding.forward), ("manual_seed", torch.manual_seed)]:
    path = Path(inspect.getsourcefile(obj))
    local_apis[name] = {
        "套件檔案": str(path),
        "套件檔案SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "定義起始行": inspect.getsourcelines(obj)[1],
    }

print(json.dumps({
    "環境": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(patches.dtype), "platform": platform.platform()},
    "凍結核對": {"source_sha256": hashlib.sha256(frozen).hexdigest(), "現行小節位元組相同": True},
    "預設圖片": {"形狀": list(image.shape), "中央RGB": image[:, 8, 8].tolist(), "角落RGB": image[:, 0, 0].tolist(), "明確指定相同": True},
    "原文形狀": {"輸入": list(patches.shape), "輸出": list(features.shape), "矩陣": list(layer.weight.shape), "偏移": list(layer.bias.shape)},
    "公式核對": {"雙精度逐列公式最大絕對誤差": error, "容忍差": 1e-7, "逐位置共用參數": True, "位置反轉只反轉輸出": True},
    "參數可求導": {"weight_requires_grad": layer.weight.requires_grad, "bias_requires_grad": layer.bias.requires_grad, "權重梯度範數": layer.weight.grad.norm().item(), "偏移梯度範數": layer.bias.grad.norm().item()},
    "種子核對": {"seed": 0, "同環境權重與偏移逐值相同": True, "輸出逐值相同": True},
    "圖片入口": {"執行順序": events, "projection輸入等於切塊結果": True, "projection輸入寬度": encoder.projection.in_features, "projection輸出寬度": encoder.projection.out_features, "輸出形狀": list(encoded.shape)},
    "文字查表": {"ID": ids.tolist(), "等於直接索引權重表": True},
    "兩值算例": {"輸入": [[1, 0], [0, 1]], "權重": [0.5, 0.5], "偏移": 0, "輸出": collision.detach().tolist()},
    "降維反例": {"矩陣": "前八欄為單位矩陣，其餘為零", "輸入差異": "第二筆只有第九格為一，第一筆全零", "輸出": selection_output.detach().tolist()},
    "練習": {"特徵": list(exercise_features.shape), "矩陣": list(exercise.weight.shape), "位置數來源": "(16//4)**2", "特徵寬度來源": "out_features=16"},
    "本機API定位": local_apis,
    "結果": "所有核對通過；沒有執行參數更新或訓練品質量測。",
}, ensure_ascii=False, indent=2))
