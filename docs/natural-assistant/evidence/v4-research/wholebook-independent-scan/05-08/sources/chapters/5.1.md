## 5.1 訓練迴圈真的正常嗎？

你把模型接好，loss也能算，接下來是否就該找更多資料？先用一道固定題檢查它有沒有真的學習。像修好一臺影印機先印一張已知測試紙，不急著處理整箱文件。我們讓模型反覆看同一個短序列，故意允許它記住答案；如果連這題都改善不了，就先檢查對齊、梯度與更新，而不是把問題歸因於資料太少。

[4.7](04.md#4.7) 已能算完整模型的代價與梯度，本節補上優化器更新。輸入 `[1,2,3]`、答案 `[2,3,4]`，只有三道固定下一字題。AdamW是管理參數更新的工具，其步幅與衰減細節會在 [5.4](#5.4)、[5.5](#5.5) 拆解；此處先把它當作按梯度調旋鈕的現成做法，衰減設0避免多一個影響。

`TinyLM`是專案中已組好的小型文字模型，`ModelConfig`把設定交給它：`vocab_size=5`提供編號0到4這五個候選，`width=8`表示每個位置的向量有8個數字，`max_length=8`限制一次最多處理8個位置；本例只用其中3個。優化器的`lr`是learning rate（學習率），設定更新步幅的尺度，此處用0.03；`weight_decay=0.0`則關閉衰減。

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

這次四十步、極小CPU模型只是診斷流程，不是正式語言訓練。loss變小表明它可以調整固定題的偏好；要知道新題表現，仍須 [1.2](01.md#1.2) 的獨立資料。如果開始與結束幾乎相同，先逐步確認optimizer負責了哪些參數、step有執行、梯度沒有被提前清掉。能背一道題不證明泛化，背不了則是值得追查的線索。

正式腳本操作可再看[訓練配方](../training.md)。要做同樣的單例診斷，資料必須真的只有那一道，而不是以為batch-size=1就永遠抽到同一題；batch大小和資料總量不同。

通過固定題之後，才把練習擴到幾篇短文。我們已用9篇小世界短文，讓一個兩層模型在L4 GPU上更新600次：用同一批訓練位置重新計算的平均代價，由5.79243降至0.06352。這一次確實有更新，而不只是呼叫`backward()`；但同一份模型在沒教過的那篇驗證短文，代價仍有0.96282。教過的題目與新題要分開看，正是[5.8](#5.8)接下來要處理的事。完整設定和重做入口在[T.4](../training.md#T.4)，不用把這份600步結果套到眼前width8、40步的診斷程式。

練習把答案y的三格都改成-100。這個工具約定-100表示忽略答案，所以三格都忽略等於沒有學習目標。先預測 `masked_loss` 會直接報錯「所有labels都被忽略」，再執行核對；這次不應顯示成功下降的報告。流程拒絕空目標，才能避免把沒有訓練假裝成訓練完成。

