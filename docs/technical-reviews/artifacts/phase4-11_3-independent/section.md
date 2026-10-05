## 11.3 只調圖片接頭，梯度怎麼穿過固定文字核心？

給一張紅方塊，目標短描述是 `red square`。如果圖片特徵能分相關屬性，文字核心也會用這些詞，可以先只調接頭，讓視覺線索接到文字行為。本節先確認答案代價能穿過固定核心回到接頭。

```python
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene

torch.manual_seed(0)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, image=scene())
loss = masked_loss(out["logits"], out["labels"])
loss.backward()
print("接頭收到梯度", model.image_projector.weight.grad.norm().item() > 0)
print("語言權重未累積梯度", model.language.embedding.weight.grad is None)
```

答案 10 個 ASCII 位元組加 EOS，共 11 個有效目標。前綴放角色、圖片與助手位置，labels 把前綴設 −100；模型先展開圖片，再做下一位置對齊。masked_loss 只對回答與結束算交叉熵，圖片仍是上下文。

兩行預期 True：接頭收到非零梯度，凍結的文字 embedding 沒有自己的 grad。固定文字運算仍傳遞對接頭的影響；沒有優化器 step，所以這個短例是梯度通路，不是生成成功率。

既有接頭小實驗真正更新後，完整描述仍為 0/6，雖然標準答案上下文裡的預測代價降低。紅方塊只生成 `red`，缺了空白與 square；綠圓還會重複 e。這保留一個有用差別：訓練時計算下一字常用標準前文，叫 teacher forcing；生成時接自己的上一字，前面一錯，後面可能跟著偏離。

因此要分開核對兩端能力、接頭更新、完整回答與 EOS。這次弱底座只有部分文字題做對，不能拿失敗推出所有接頭方案都無效；接頭也無法補回圖片入口已丟掉的資訊。

練習將接頭也凍結，在 backward 前印可訓練總數，會是 0，接著 backward 報代價未連到可求梯度參數。這是關閉所有學習路徑的反例，應在啟動訓練前攔下。

<details>
<summary>回顧與查證</summary>

可回顧：[11.1尺寸與理解的區別](11.md#11.1)、[11.2凍結名單](11.md#11.2)、[10.7展開後才錯位](10.md#10.7)、[W.6梯度與更新](../first-steps.md#W.6)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/projector.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>

