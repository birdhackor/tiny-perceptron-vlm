## 7.5 不算 loss 的內容就看不見嗎？

問題Q不算直接答案代價，模型仍需要讀它回答A。因此問題格的候選分數可以梯度0，Q的輸入特徵卻從後面答案收到梯度。兩種梯度针對不同變量，不能混看。

在最短Q／A對話，Q位於X索引2、ID89。embedding.weight第89列是Q的字向量；答案代價通過中間運算追溯到它。下面分別查看第一格logits和Q特徵列。

```python
import torch
from tiny_perceptron.data import render_chat
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
model = TinyLM(ModelConfig(width=8))
logits = model(x[None])["logits"]
logits.retain_grad()
masked_loss(logits, y[None]).backward()
print("第一位置logits梯度", logits.grad[0, 0].norm().item())
print("Q字向量梯度", model.embedding.weight.grad[x[2]].norm().item())
```

`logits` 是中間結果，PyTorch一般不保留中間節點的 `.grad`，所以 `retain_grad()` 明確要求留下它，相關葉參數區別見 [1.11](01.md#1.11)。第一格標籤-100，輸出的logits梯度大小應0；Q字表列的梯度通常大於0，因為後面的答案預測用到它。`norm` 把各格平方加總再開根，這裡只看是否有影響，不把兩者的大小當同一單位比較。

第一項回答「這個位置的候選分數有沒有直接答案代價」，第二項回答「這份輸入表示有沒有影響後面的答案」。都叫梯度，卻對不同變量問不同問題。只檢查整個embedding層是不是None，無法知道Q那列是否參與；只看到第一格0，也不能推斷所有user內容都被凍結。

共享輸入表也會因user內容被後續讀取而調整；忽略user loss沒有凍結它。detach輸入或禁止回答讀Q，會切掉另一條影響路線，是新的操作。這段只求梯度，沒有step；它核對資料依賴，尚未調整出回答能力。

練習只把user的Q替換成Z，保持答案A與seed不動。先預測第一格logits梯度仍0、被讀取的Z字表列通常有梯度，再重新執行核對。因為X索引2仍是問題字，程式會自動查新的那一列，這次測試的是角色與位置關係，而不是Q這個字特別有魔法。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

