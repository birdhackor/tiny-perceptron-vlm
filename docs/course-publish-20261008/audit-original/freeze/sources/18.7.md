## 18.7 教師需要跟著更新嗎？

一次學生更新後，我們希望教師權重完全不變，學生自己的權重改變。若兩者一起更新，學生追的目標也會移動；本章選擇先保存固定教師版本，再讓學生學它提供的訊號。下面用同一個輸入 `[1,2]` 核對誰真的改了數字，同時分清評估模式、權重凍結與不記錄導數三個工具。

`eval()`切換模組行為，例如停用訓練時的dropout，不會自動停用導數記錄；`requires_grad_(False)`讓教師權重不求梯度；`no_grad()`則讓該區塊的教師計算不建立反向運算圖。優化器只接收學生參數，決定更新哪些數字。一起使用能清楚表達固定教師的角色，也避免保存不需要的教師中間計算。

下面用兩個很小的線性層，只檢驗固定與更新；這個教師隨機初始化，沒有已學會的知識。輸入 `[1,2]`，教師和學生都輸出三個特徵，用兩者平方差平均當學生誤差，方便檢查一次更新是否只改學生。

```python
import torch
from torch import nn

torch.manual_seed(0)
teacher = nn.Linear(2, 3).eval().requires_grad_(False)
student = nn.Linear(2, 3)
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
x = torch.tensor([[1.0, 2.0]])
teacher_before = teacher.weight.detach().clone()
student_before = student.weight.detach().clone()
with torch.no_grad():
    target = teacher(x)
output = student(x)
optimizer.zero_grad()
(output - target).square().mean().backward()
optimizer.step()
print("教師梯度", teacher.weight.grad)
print("教師最大改動", (teacher.weight - teacher_before).abs().max().item())
print("學生有改動", bool((student.weight - student_before).abs().max() > 0))
probe = x.clone().requires_grad_()
print("輸入可微時教師輸出記圖", teacher(probe).requires_grad)
```

應印出教師梯度None、最大改動0、學生有改動True。`SGD`是沿梯度反方向更新的優化器，lr=0.1表示步幅；它只拿student.parameters，所以沒有教師權重。快照的clone建立獨立數值副本，才能比較更新前後；若只保留引用，檢查就會同時看到新權重而失效。

最後一行應為True：即使教師權重凍結，probe輸入需要梯度，普通teacher(probe)仍會記錄如何從輸入影響輸出。這解釋為何no_grad與freeze不是完全相同的事。正式蒸餾通常不希望通過教師教輸入端，所以教師生成目標那段仍使用no_grad；若另設可微教師用途，應明確說明改變了哪條學習路徑。

凍結不等於教師計算免費，forward仍要讀權重、做矩陣運算；若預先保存目標可以重用，也會增加儲存與版本管理成本。教師的能力應由獨立任務驗證，不是由 `grad is None`判斷。

練習只把最後 `teacher(probe)`也包進no_grad，再預測requires_grad變False，執行核對。其他訓練更新那段不用改，教師仍不變、學生仍更新；你就能具體分辨模組模式、權重凍結與整段不記圖三種機制。

<details>
<summary>選讀：來源、量測條件與原始實報</summary>

正式實驗有三種教師：屬性教師回答顏色、形狀等小世界問題；風格教師按指定格式回答算術與其他題；MoE教師則做故事文字續寫。MoE是混合專家模型，每個文字位置會選部分子網路來處理，這些子網路叫專家，詳見[15.1](15.md#15.1)。這三種用途的來由可先讀[18.1](18.md#18.1)。快取保存教師在已知前文下，各個下一文字候選的分數，讓學生多次學習時重用；硬回答則是教師實際生成的一整段文字，例如形狀答案`circle`，不是整份候選分數表。

三個教師都採eval、凍結權重及no_grad預先生成訊號；所有學生完成後，逐張權重表的數值指紋仍與起點相同，報告`teacher_frozen_and_unchanged=true`。這比只看梯度None多核對了實際結果。教師訊號可以重用，但生成與保存仍有成本；各教師的版本、權重未變核對與成本記錄保留在[正式蒸餾報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/distillation.json)的`tasks`中。

</details>

