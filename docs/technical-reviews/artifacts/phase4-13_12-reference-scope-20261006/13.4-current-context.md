## 13.4 固定參考在比較什麼？

對同題兩篇答案，原模型的log分數是較佳−4、較差−3，差距−1；更新中的模型變成−3、−3，差距0。它相對原差距改善了`0−(−1)=1`，但兩候選現在只是同分，尚未保證較佳勝出。

正在更新、之後實際作答的模型叫**策略模型（policy）**；保留原起點、不跟著更新的副本叫**參考模型（reference）**。DPO直接偏好最佳化會用它作固定基準，關注相對偏好如何改變。

```python
import copy
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

policy = TinyLM(ModelConfig(width=8))
reference = copy.deepcopy(policy).eval().requires_grad_(False)
before = reference.embedding.weight.clone()
with torch.no_grad():
    policy.embedding.weight.add_(0.1)
scores = reference(torch.tensor([[1, 2]]))["logits"]
print("參考未跟著改", torch.equal(before, reference.embedding.weight))
print("參考不記梯度", scores.requires_grad)
```

這個隨機小模型只檢查副本與凍結。`deepcopy`複製獨立數字，`requires_grad_(False)`不追蹤參考參數的梯度，`eval()`採評估行為。人為給policy的嵌入加0.1後，第一行應為`True`：reference沒變；第二行為`False`：參考輸出不需梯度。沒有偏好訓練發生。

正式使用時通常保留完成示範微調的起點。參考未必每題都答對，它提供原行為的基準，並非真值裁判。策略與參考最初相同，因此相對差距最初為0，即使原模型更偏向錯答。參考全程固定，也不能放進策略更新器。

練習把`copy.deepcopy(policy)`改成`policy`。兩個名稱將指向同一對象，凍結也會連policy凍結；手動加0.1後，第一行變`False`。獨立複製與凍結各有作用，缺一項都不能替代固定參考。

<details>
<summary>補充：來源、完整設定與既有實測紀錄</summary>

正式比較使用[8.3的`content.pt`](08.md#8.3)作固定參考，它在七道留出加法原本零題答對，不是已通過的算術老師。訓練前複製策略與參考，兩種beta設定都從這份相同起點開始。beta是正的縮放係數：偏好代價會先把前面算出的相對差距乘上它，再決定要推動多少。它不是學習率；此處先知道它是另一項訓練設定即可，[13.6](13.md#13.6)會用相同差距逐項觀察它的作用。參考數字指紋在更新後保持不變。這只能證明固定尺沒有跟著動，不能替新策略的回答保證真值。

還要分清「相對進步」與「已經排在前面」。beta 0.1版本在驗證題`3+5=?`上，較佳`8`減較差`9`的log分數，從參考的約−10.89346變成−8.44789，相對提高2.44557。但差距仍為負，表示這兩個完整候選中，錯的9仍有較高機率。相對改善可以是從更偏向錯答，移到稍微沒那麼偏向錯答，並不等於正確候選已勝出。

</details>

