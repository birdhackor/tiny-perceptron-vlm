## 5.2 一次看多少例子？

兩篇短文一起練，第一篇有四個下一項答案，第二篇只有一個。這次更新要聽「兩篇各一票」，還是「五個答案各一票」？本課選後者：五項代價相加再除以五，排齊矩陣的空位不投票。

一次共同計算的幾筆叫batch。以下只給候選分數和答案來看平均規則：兩筆各排四位置，五個候選沿用句號、狗、看、貓、逗號；第二筆後三個label=-100，表示沒有要計分的答案。

```python
import torch
from tiny_perceptron.model import loss_sum

logits = torch.zeros(2, 4, 5, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [1, -100, -100, -100]])
total, count = loss_sum(logits, labels)
mean_loss = total / count
print("有效答案", count.item())
print("總代價", total.item(), "平均代價", mean_loss.item())
mean_loss.backward()
```

輸入logits形狀 `[2,4,5]`：兩筆、每筆四位置、五候選。這些是尚未轉成機率的分數；五個相同的0分經過同一轉換，會得到相同機率，而五個機率總和是1，所以各為1÷5＝0.2。按1.8的負對數代價，任何有效答案的代價都是 `-log(0.2)≈1.609`。labels中五格是真答案，三格-100被排除。`loss_sum` 分別回傳代價總和與有效數量，所以輸出count=5、總數約8.047、平均約1.609。

`requires_grad=True`讓PyTorch記錄以這份分數進行的計算，供稍後`backward()`求出它們對代價的梯度。count、total和mean_loss都各自是只含一個數的tensor；`.item()`取出普通Python數字，方便列印。最後`backward()`對平均代價求導，被忽略的位置沒有答案代價貢獻。

先各篇平均再平均兩篇，短篇唯一答案會佔一半、長篇每項只佔八分之一，和每答案五分之一不同。梯度也能按代價相加與平均；平均後由優化器更新一次，和每題各更新一步不是同一流程。

練習把第二筆第二格答案從-100改成2，先預測有效數變6、總代價變約9.657、平均仍1.609，再執行核對。接著改回-100，應回到五；你改的是哪些答案進入統計，不是模型的候選字表。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

