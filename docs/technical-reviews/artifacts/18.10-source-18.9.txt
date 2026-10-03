## 18.9 標準答案和教師偏好怎麼一起學？

教師的分布含候選關係，標準答案則提供明確真值，兩者有時也會矛盾。可以把[標準答案交叉熵](18.md#18.4)和[教師KL](18.md#18.8)一起作學生的學習目標，而不是選完一種就丟掉另一種。混合係數alpha介於0與1，決定教師訊號占多少份量；它不是教師可信度的自動測量。

設CE是標準答案誤差，K是已包含T²尺度的蒸餾KL，使用 `L=(1-alpha)×CE+alpha×K`。alpha=0只學真值，alpha=1只學教師，中間值保留兩條路。以學生每格各給50%、教師每格給 `[0.8,0.2]`、標準答案兩格分別0與1，CE為0.6931，T=1時K為0.1927。alpha=0.5總誤差約0.4429，就是兩個誤差各一半。

```python
import torch
from tiny_perceptron.alignment import distillation_loss, distillation_kl
from tiny_perceptron.model import masked_loss

student = torch.zeros(1, 2, 2, requires_grad=True)
p = [0.8, 0.2]
teacher = torch.tensor([[p, p]]).log()
labels = torch.tensor([[0, 1]])
ce = masked_loss(student, labels)
k = distillation_kl(student, teacher, labels, temperature=1.0)
print("CE/KL", round(ce.item(), 4), round(k.item(), 4))
for alpha in (0.0, 0.5, 1.0):
    loss = distillation_loss(student, teacher, labels, alpha=alpha, temperature=1.0)
    print("alpha", alpha, "總誤差", round(loss.item(), 4))
assert torch.allclose(distillation_loss(student, teacher, labels, alpha=0, temperature=1), ce)
```

輸入是一段文字、兩個位置，每個位置有兩個候選。`student`的形狀是`(1, 2, 2)`，零logits形成均勻分布。教師先用清單`p`保存兩候選的比例；`[[p, p]]`最外層只有一段，裡面放兩份相同的`p`，因此建成同樣的`(1, 2, 2)`形狀。`.log()`逐個比例取自然對數，讓後面的softmax得到指定比例。`masked_loss`按labels索引取標準答案的負log機率；`distillation_kl`則對整列候選比例比較。labels此時同時扮演真值與有效位置標記，沒有-100，所以兩格都算。程式沒有更新學生，只核對同一份分數的誤差組合。

輸出第一行0.6931與0.1927，後面依序0.6931、0.4429、0.1927。最後assert檢查alpha=0確實退回普通CE。這個端點是很有用的程式契約，避免配方看似混合卻仍偷偷加入教師項。alpha=1的數字較小不代表學生答得更好，因為各行的目標已經不同；真正質量要用相同獨立評估比較。

本例第二格標準答案是候選1，但教師仍偏好候選0。alpha越大，對這格真值的直接要求越弱，說明教師出錯時保留CE路徑的意義；具體哪個alpha有效，必須依驗證資料判斷，不是固定0.5永遠最佳。

正式CE+KL支線也是`0.5×CE + 0.5×T²×KL(teacher||student)`，T=2；CE仍讀獨立真值，沒有用教師錯答案代填。教師分布在標準回答前文上預先保存，學生每次更新才算自己的分布。原始logits可以重新用不同T轉成比例，但本輪沒有搜尋alpha或T，也沒有因測試不好臨時調整。不同目標的訓練loss不能互相排名，下一節只用共同真值評估結果。

如果T不為1，teacher與student都用同樣T形成KL分布。本節的`distillation_kl`輔助函式已在內部乘T²，外層混合時不可再乘，否則把教師項又放大一次。記錄alpha、T與是否包含T²，才能讓其他人知道你訓練的實際目標。

練習只在alpha列表增加0.25。先手算 `0.75×0.6931+0.25×0.1927≈0.5680`，再執行核對。接著比較這項的尺度，不要把降低總誤差當成新任務正確率；你改的是兩種指導之間的權重。

