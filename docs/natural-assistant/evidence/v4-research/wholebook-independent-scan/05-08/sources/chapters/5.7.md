## 5.7 訓練中斷怎麼接續？

遊戲存檔不僅要記角色在哪，還要記揹包和任務進度。訓練存檔也一樣：只存模型權重能做預測，但若想下一步接着同樣方式學，還需要優化器的歷史、已走步數、學習率計劃、文字編碼規則與隨機狀態。這樣的完整狀態快照叫checkpoint（檢查點），不是把一份文字改副檔名就能代替。

[5.3](#5.3)、[5.4](#5.4) 已看到相同梯度在不同歷史下會有不同更新，所以遺漏歷史可能讓接續路線改變。本例真的先走一小步，再保存與重載；輸入只有一筆三字ID，CPU很快就能完成。字表用metadata另存，因為只記字表大小不足以知道每個ID代表什麼。

這裡沿用[4.7的完整文字模型與答案對齊](04.md#4.7)。`TinyLM`是接字模型，`ModelConfig`記錄其設定：字表5項、每位置8特徵、最多8位置。`torch.tensor`把數字清單存成可計算的資料，軸背景見[W.3](../first-steps.md#W.3)。按本次字表，x的 `[1,2,3]`是「狗看貓」，y的 `[2,3,4]`是「看貓，」，逐位置要求看到狗猜看、看到看猜貓、看到貓猜逗號。`model(x)["logits"]`取得一筆、三位置、每位置五個候選的原始分數；`masked_loss`把這些分數與y的下一字答案比較，得到平均猜錯代價，本例沒有忽略位置。`.parameters()`列出模型的可調數字，交給更新工具處理。

本例的AdamW沿用Adam的梯度方向m、平方大小v與步數歷史，另外把[5.5的權重衰減](#5.5)獨立控制；所以恢復時仍須帶回這些更新記錄，不能只恢復模型數值。

```python
import tempfile
from pathlib import Path
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.training import save_checkpoint, load_checkpoint

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
masked_loss(model(x)["logits"], y).backward()
optimizer.step()
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "state.pt"
    save_checkpoint(path, model, optimizer, step=1, metadata={"chars": ["。", "狗", "看", "貓", "，"]})
    loaded, payload = load_checkpoint(path, restore_rng=True)
    restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert torch.equal(model(x)["logits"], loaded(x)["logits"])
    for candidate, opt in [(model, optimizer), (loaded, restored_optimizer)]:
        opt.zero_grad()
        masked_loss(candidate(x)["logits"], y).backward()
        opt.step()
    assert torch.equal(model(x)["logits"], loaded(x)["logits"])
    print("保存步數", payload["step"], "字表", payload["metadata"]["chars"])
```

`TemporaryDirectory` 建立測試資料夾，離開縮排區自動清除；`Path` 用途見 [W.7](../first-steps.md#W.7)。`save_checkpoint` 保存模型設定與權重、optimizer狀態、步數和隨機資訊，metadata是額外自訂資料。重載後先確認同輸入得到完全相同分數，再各自更新一步比較，兩次檢查都應通過；輸出步數1與五字對照表。

`load_checkpoint` 用檔內設定重建模型，並把參數加載進去；`restore_rng=True` 還恢復隨機數狀態。優化器則需另建並調用 `load_state_dict`，不能只建一個新AdamW就認為歷史也回來。字表metadata也不是自動產生文字工具，要按存下的約定重建編碼與解碼。正式訓練還要保存資料抽樣位置與排程設定，保證下一批題也相同。

這個例子沒有dropout，也沒有隨機抽題，適合逐數檢查下一步。dropout是在訓練中隨機將部分中間數值設成零的操作；若有這類隨機行為，還須控制隨機狀態才能比較，詳見[5.17](#5.17)。更大訓練也可能有非確定性運算，也就是相同設定下仍因運算順序等原因產生微小差異，不能保證所有儲存數字完全相同，應另外定義可接受的續訓驗證。此處只加載自己剛生成的存檔，正式使用同樣只加載可信來源。

我們也在GPU上另做了有隨機抽題的接續實驗：先定好總共80步的學習率計畫，一路跑完；另一條路在第40步保存權重、更新工具和隨機狀態，重載後再跑第41到80步。最後逐項比較兩份權重，最大差為0。這項結果支持的是這份小模型和這次排程的接續，不是所有GPU計算都必然逐位相同。尤其不能在重載後偷偷換成新的總步數計畫，卻仍要求它走出原來的路線；排程也是存檔的一部分。這個80步接續檢查和[T.4的600步文字訓練](../training.md#T.4)是分開的實驗。

練習只註釋掉 `restored_optimizer.load_state_dict(...)`，先預測第一次重載預測檢查仍通過、第二次更新後可能不相同，再執行核對。權重成功恢復與訓練狀態完整恢復是兩個層次，第二個檢查才會發現漏掉的歷史。

