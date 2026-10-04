import torch
torch.set_num_threads(1)
torch.set_default_device("cpu")
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

torch.manual_seed(0)
lm = TinyLM(ModelConfig(width=8)).eval()
multimodal = MultiModalLM(lm).eval()
ids = torch.tensor([1, 40, 50])
a = lm(ids[None])["logits"]
b = multimodal(ids)["logits"]
print("分數形狀", tuple(a.shape))
print("純文字入口相同", torch.equal(a, b))

try:
    multimodal(torch.tensor([1, 5, 4]))
except Exception as error:
    print("缺圖錯誤種類", type(error).__name__)
    print("缺圖錯誤訊息", str(error))
else:
    print("缺圖沒有報錯")
