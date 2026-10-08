## 7.6 補齊 batch 的位置該算 loss 嗎？

兩筆對話長短不同，為排成同一張表，短筆會補PAD。補的格子沒有真正答案，不能因為多補幾格就讓平均代價變好，或要求模型學著寫PAD。

先固定兩個位置的候選分數與答案，只添三格label=-100。這樣可以單獨看分子與分母是否只計真正兩個答案；PAD是輸入專用ID，-100則是答案忽略值。

```python
import torch
from tiny_perceptron.model import masked_loss

a = torch.tensor([[[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
labels = torch.tensor([[1, 2]])
b = torch.cat([a, torch.zeros(1, 3, 4)], dim=1)
extended = torch.tensor([[1, 2, -100, -100, -100]])
before = masked_loss(a, labels)
after = masked_loss(b, extended)
print(before.item(), after.item())
assert torch.allclose(before, after)
```

輸入a是一筆、兩位置、四候選，正確答案分別ID1與2。第一格高分2給ID1約0.711機率、代價約0.341，第二格全0給每個候選0.25、代價約1.386，所以平均約0.864。`torch.cat(...,dim=1)` 沿位置軸接三格，形成 `[1,5,4]`；三格label全-100，分子不納入它們，分母仍2。輸出兩個平均應一致。

這裡新增位置的logits也是全0，但即使換成任意分數，只要label忽略，它們也不參加直接答案代價。反之，若label誤設PAD ID0，交叉熵會認真把它當成「下一項應為0」的任務，產生新的更新方向。不能只把輸入值填0，就期待loss自動知道是補齊。

統計不變還不代表模型讀入PAD後預測不變。本段沒有重跑補齊序列的前向，只驗證loss；下一節另檢查讀取許可和位置。新增填充不能增加有效答案，整批有效目標0時也不能算平均。

練習只把extended第一個-100改成0，先預測它加入第三道題、平均代價約1.038，舊一致檢查失敗，再執行核對。恢復-100後兩值再次相同。你沒有改原兩個有效位置的預測，變化完全來自哪些位置被納入統計。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

