## 5.7 訓練中斷怎麼接續？

沿用「狗看貓，」：先更新一步、存檔、重載。只恢復權重，當下預測可以相同；若希望兩邊下一步也相同，就還要恢復Adam的方向、平方大小與步數歷史。

X=[1,2,3]是狗、看、貓，Y=[2,3,4]是看、貓、逗號。以下比較剛重載時的分數，再讓兩邊各更新一步比較；字表另存每個ID的含義。

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

`TemporaryDirectory` 建立測試資料夾，離開縮排區自動清除。`Path(directory)` 表示那個資料夾的位置；後面的 `/ "state.pt"` 接上檔名，得到資料夾內 state.pt 的存檔路徑。路徑物件的其他基本操作見 [W.7](../first-steps.md#W.7)。`save_checkpoint` 保存模型設定與權重、optimizer狀態、步數和隨機資訊，metadata是額外自訂資料。重載後先確認同輸入得到完全相同分數，再各自更新一步比較，兩次檢查都應通過；輸出步數1與五字對照表。

`load_checkpoint` 用檔內設定重建模型，並把參數加載進去；`restore_rng=True` 還恢復隨機數狀態。優化器則需另建並調用 `load_state_dict`，不能只建一個新AdamW就認為歷史也回來。字表metadata也不是自動產生文字工具，要按存下的約定重建編碼與解碼。正式訓練還要保存資料抽樣位置與排程設定，保證下一批題也相同。

正式續訓還需保存資料抽樣位置與學習率計畫，讓下一批題和步幅也一致。若有dropout這類隨機省略中間數值的操作，需恢復相應亂數；若運算本身不能逐位重現，則另定可接受誤差。只載入可信來源存檔。

練習只註釋掉 `restored_optimizer.load_state_dict(...)`，先預測第一次重載預測檢查仍通過、第二次更新後可能不相同，再執行核對。權重成功恢復與訓練狀態完整恢復是兩個層次，第二個檢查才會發現漏掉的歷史。

<details>
<summary>補充：實作約定與原始紀錄</summary>

我們也在GPU上另做了有隨機抽題的接續實驗：先定好總共80步的學習率計畫，一路跑完；另一條路在第40步保存權重、更新工具和隨機狀態，重載後再跑第41到80步。最後逐項比較兩份權重，最大差為0。這項結果支持的是這份小模型和這次排程的接續，不是所有GPU計算都必然逐位相同。尤其不能在重載後偷偷換成新的總步數計畫，卻仍要求它走出原來的路線；排程也是存檔的一部分。這個80步接續檢查和[T.4的600步文字訓練](../training.md#T.4)是分開的實驗。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

