## 1.8 怎樣用正確答案評分這次猜法？

答案是「看」。兩份預測都將看排第一，但一份給它 0.5，另一份給 0.9。只算「第一名有沒有對」會把它們當一樣；訓練時我們希望第二份付出的代價較低，才能看出微調分數是否有幫助。

取正確字那一格的機率 p，算 `-log(p)`，得到**代價**，也叫 **loss**。`log` 是自然對數：p=0.5 時代價約 0.693，p=0.9 時約 0.105，p=0.1 時約 2.303。正確答案得到的機率越少，代價越大；代價不是猜錯題數。

把多題的這種代價取平均，這裡稱**交叉熵**。先用「貓、看、狗」分數 `[0,2,0]` 算一題：答案 ID 為 1，即第二個候選「看」。

```python
import torch
from torch.nn import functional as F

p = torch.tensor([0.1, 0.5, 0.9])
print("正確機率", p, "代價", -p.log())
logits = torch.tensor([[0.0, 2.0, 0.0]])
target = torch.tensor([1])
probabilities = logits.softmax(dim=-1)
manual = -probabilities[0, 1].log()
loss = F.cross_entropy(logits, target)
print(probabilities, manual.item(), loss.item())
assert torch.allclose(loss, manual)
```

第一段列出三種正確機率的代價。第二段的 `logits` 是原始候選分數，形狀 `[1,3]`；`target=[1]` 是一題的答案位置。softmax 後約為 `[0.1065,0.7870,0.1065]`，`probabilities[0,1]` 取看的機率，手算與 `F.cross_entropy` 都約 0.2395。

`F.cross_entropy` 接收原始分數和答案 ID，內部已完成穩定的比例化與對數計算。因此交給它的是 `logits`；若交入機率，機率又被當成分數轉換一次，會算成另一個代價。

練習把答案改成 `[0]`，手算索引也改成 `[0,0]`。分數完全相同，但答案換成貓，代價便升到約 2.2395。這讓你看見：判定好壞要同時有預測與答案，不能只看最高分。

<details>
<summary>補充與重做</summary>

`F` 是 `torch.nn.functional` 的短名字。自然對數見 [W.4](../first-steps.md#W.4)，本例代價單位是 nat，除以 `log(2)` 可換成 bit。比較平均代價需固定資料及文字拆分方式；第 6 章會說明不同拆分如何影響比較。

</details>

