## 5.1 訓練迴圈真的正常嗎？

「狗看貓，」是一條固定短句。我們把句號、狗、看、貓、逗號依序編成0、1、2、3、4；給模型「狗看貓」，要求三個位置分別接「看、貓、逗號」。先只練這一道題：如果連它的代價都降不下來，就值得檢查更新流程，再擴大資料。

下面用已組好的TinyLM，字表5項、每位置8個特徵、最多8位置。X是[1,2,3]，下一項答案Y是[2,3,4]。AdamW是依梯度調參數的工具，此處衰減設0，先觀察清梯度、算代價、求梯度、更新四步。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[2, 3, 4]])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
before = masked_loss(model(x)["logits"], y).item()
for step in range(40):
    optimizer.zero_grad()
    loss = masked_loss(model(x)["logits"], y)
    loss.backward()
    if step == 0:
        print("第一步embedding梯度大小", model.embedding.weight.grad.norm().item())
    optimizer.step()
after = masked_loss(model(x)["logits"], y).item()
print("固定例子代價", before, "→", after)
assert after < before
```

`model.parameters()` 交給優化器模型全部可調數字。每次先清舊梯度，再前向算新預測與loss、`backward` 求導、`step` 更新，四步順序與 [1.12](01.md#1.12)、[1.13](01.md#1.13) 對上。最後重新算更新後代價，應明顯低於開始；精確小數依PyTorch環境而異。

`model.embedding`是輸入編號查向量的工具，`embedding.weight`是它保存的可調數字表：本例有5列，每列8個數，詳見[2.1的查表](02.md#2.1)。`.grad`保存這張表各格收到的更新訊號；`.norm()`把訊號各格平方加總再開根，得到一個大小。第一步這個大小應為正，說明答案代價已對字向量表產生影響。

梯度為正只說這張表收到訊號，代價下降再說更新改善了固定題。若代價不動，依序看優化器是否拿到對的參數、step是否執行、梯度是否在更新前被清掉。固定題刻意允許記憶；新題改善需要另留資料。

練習把答案y的三格都改成-100。這個工具約定-100表示忽略答案，所以三格都忽略等於沒有學習目標。先預測 `masked_loss` 會直接報錯「所有labels都被忽略」，再執行核對；這次不應顯示成功下降的報告。流程拒絕空目標，才能避免把沒有訓練假裝成訓練完成。

