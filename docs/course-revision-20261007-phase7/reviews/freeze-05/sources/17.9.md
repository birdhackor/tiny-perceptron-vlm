## 17.9 不更新權重，也能做量化嗎？

手上已有一份訓練權重，可以先固定它，選刻度、轉成低位元，再檢查同樣輸入。這叫**訓練後量化（Post-Training Quantization，PTQ）**。最基本的weight-only PTQ依權重本身的範圍選scale，不必先收集中間特徵。

要隔離轉換造成的差別，原版與量化版必須來自同一份數字。這個短程式先用隨機初始化模型，只核對轉換與比較方式；真的PTQ應先載入選定訓練檔，再用相同流程和任務題測品質。下面先深複製獨立副本，再轉換副本內的線性層。

```python
import copy
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.quantization import replace_linear_layers

torch.manual_seed(0)
original = TinyLM(ModelConfig(width=8, tied=False)).eval()
ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    before = original(ids)["logits"]
    quantized = replace_linear_layers(copy.deepcopy(original), bits=4)
    after = quantized(ids)["logits"]
    original_again = original(ids)["logits"]
print("兩版形狀", tuple(before.shape), tuple(after.shape))
print("原版是否被改", (before - original_again).abs().max().item())
print("量化分數MAE", (before - after).abs().mean().item())
```

短序列3個token、預設264候選，兩版分數形狀都`(1,3,264)`。`eval()`採評估行為，`no_grad()`不記錄求導；`deepcopy`複製獨立數字。函式會修改傳入副本，不能把原版直接傳入後仍假設它不變。

原版前後最大差應0，量化分數MAE通常非零。此函式不轉換嵌入與正規化，因此也不是所有數都4-bit。

本例`tied=False`，即輸入與輸出表不共用權重。若原本共用，替換輸出為量化buffer還涉及共享關係，須另外設計，不能默默當成只改精度。

若也量化中間特徵，還須選它們的範圍；可以當次估計，或用資料先校準，17.12會比較。這不等於QAT，PTQ本身的這一步沒有更新權重。

練習只改bits8，保持原版、種子與輸入；形狀和原版不變核對仍同，分數誤差通常降低。速度則是另一個測量。

<details>
<summary>補充：既有實驗、來源與完整測量範圍</summary>

正式實驗則載入[T.4的直接SFT模型](../training.md#T.4)，寬64、兩層、不共享輸入輸出表，再以原45筆訓練題更新120次，batch16、學習率0.003、seed42，讀到13,610個有效回答目標。這份更新後的FP32才是共同來源；4-bit與8-bit各從它直接轉換，沒有再訓練，也沒有先轉8-bit再降4-bit。原始家族切分仍為45／5／10筆，分別來自9／1／2個屬性家族，三側無交集。兩份packed檔重新載入後才評估，品質檢查與實報入口見[17.15](17.md#17.15)，完整重跑入口見[T.9](../training.md#T.9)。來源模型原來的5/10與這次FP32的6/10之間另有120次更新，不能把那一題進步歸功於量化。

</details>

