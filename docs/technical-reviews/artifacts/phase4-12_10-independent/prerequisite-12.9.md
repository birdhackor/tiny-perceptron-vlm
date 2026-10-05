## 12.9 聲音特徵怎麼接進文字模型的回答位置？

用 440 Hz 單音做一題：規則是頻率高於 300 Hz 回 `high`，否則回 `low`，所以本題希望答案 `high`。先檢查聲音入口能否接到文字模型，讓答案誤差沿這條路徑傳回。

文字序列中放一個 `<audio>` 位置，實際計算時以聲音的 11 條特徵取代它；接頭把聲音特徵寬度轉成文字核心的寬度。這個標記提供放入聲音的位置，裡面沒有預先寫 `high`。

```python
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, tone

torch.manual_seed(0)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
prefix = [tok.bos_id, tok.user_id, tok.audio_id, tok.eos_id, tok.assistant_id]
answer = tok.encode("high") + [tok.eos_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, waveform=tone(440))
loss = masked_loss(out["logits"], out["labels"])
loss.backward()
print("分數形狀", tuple(out["logits"].shape))
print("聲音接頭收到梯度", model.audio_projector.weight.grad.norm().item() > 0)
```

原序列有 5 個前綴位置、`high` 的 4 個 ASCII bytes 與 EOS，共 10 個位置。聲音標記 1 個位置擴成 11 個，成為 20；再為下一個 token 的預測移位，分數形狀是 `(1,19,264)`。264 是 tokenizer 詞表大小，不是聲音類別數。

前綴標籤為 −100，聲音插入位置也不計答案損失；只有四個字母與 EOS 是有效目標。計算訓練損失時放入正確前文，和真正生成時只能拿自己的前一個輸出不同。

`backward()` 後接頭梯度大於零，表示損失與聲音路徑相連；本程式沒有 optimizer step，沒有把這題教會模型。既有單音訓練另測 14 題，生成答案對 11/14，錯誤集中在邊界與時長變化；完整配方放補充，不與這段梯度示範混稱成果。

練習把頻率改成 200，答案與有效目標應一起改成 `low`。接通相同通路之後，還要用換聲音控制確認模型不是固定回 high，這是下一節的問題。

<details>
<summary>回顧與查證</summary>

可回顧：[12.8時間特徵](12.md#12.8)、[10.7展開與標籤錯位](10.md#10.7)、[11.3梯度檢查](11.md#11.3)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/audio.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>

