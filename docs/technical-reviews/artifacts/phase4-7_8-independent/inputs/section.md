## 7.8 補了 PAD 該取哪個輸出？

短題右補PAD時，下一項要從最後真實格預測，不能一律取矩陣最末格。第一筆[真,真,假,假]末格索引1；另一筆左補[假,真,真,真]末格索引3。

光用有效數減一，左補那筆會得到2而非3。應找True所在的最大物理索引，再為每筆取對應logits。以下用人工編號分數，讓取了哪格看得清楚。

```python
import torch

valid = torch.tensor([[True, True, False, False], [False, True, True, True]])
positions = torch.arange(4)[None].expand_as(valid)
last = positions.masked_fill(~valid, -1).amax(dim=-1)
logits = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
selected = logits[torch.arange(2), last]
print("最後有效索引", last)
print("取出的候選分數", selected)
assert last.tolist() == [1, 3]
```

輸入兩筆有效表和人為編號的候選分數，每位置五候選。`arange(4)` 產生0到3，`[None]` 加筆軸，`expand_as` 讓每筆共用這排物理位置編號。`~valid` 將真/假反過來，`masked_fill` 把無效格位置設-1，再沿各筆取最大 `amax`，得到 `[1,3]`。

logits用0到39排成 `[2,4,5]`，數字只為取值清楚。`logits[torch.arange(2),last]` 同時指定兩筆各自的末格：第0筆格1是 `[5,6,7,8,9]`，第1筆格3是 `[35,36,37,38,39]`。輸出形狀 `[2,5]`，每筆一組下一項候選，而不是再多取一個位置軸。

選取只找正確起點，不會修改分數或替你抽下一token。某筆全False會得到-1，Python又把它當末格，正式入口须先拒絕空題。前面的valid或模型位置若錯，即使取對末格也不能修好預測。

練習只把第一筆改成 `[False,False,True,True]`，先預測last從1變3，取值變 `[15,16,17,18,19]`，第二筆保持不動，再執行並改檢查核對。這次有效數仍兩格，卻末索引不同，直接證明數量與最後位置是不同資訊。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

