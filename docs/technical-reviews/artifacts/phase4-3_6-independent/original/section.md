## 3.6 整段一起算，怎樣禁止讀後面的答案？

「貓看狗。」會配成「貓→看、看→狗、狗→。」。訓練時已知完整文字，可以同時計算各位置；但貓位置若讀到右邊的看，就是把答案拿來猜答案。生成時只給貓，又沒有這個答案可讀。

所以每個查詢要有讀取許可：位置 i 只准讀位置 j≤i，即自己與更早的位置。這份真／假的許可表叫**因果遮罩**。目前字允許讀，因為要猜的是它右邊一字。

下面改成四張透明數字卡，q/k 全為 0，所有允許位置分數相同，就會平分讀取比例。value 為 `[1,10]`、`[2,20]`、`[3,30]`、`[4,40]`；希望前面查詢不受後面資料改動影響。

```python
import torch
from tiny_perceptron.attention import manual_attention

q = torch.zeros(1, 1, 4, 2)
k = torch.zeros_like(q)
v = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]])
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
out, weights = manual_attention(q, k, v, allowed)
changed = v.clone()
changed[:, :, 3] += 100
other, _ = manual_attention(q, k, changed, allowed)
print(weights[0, 0])
print(out[0, 0])
assert torch.allclose(out[:, :, :3], other[:, :, :3])
```

`tril()` 留下對角線與下方的 True。`[None,None]` 補入筆數與讀法兩軸，資料形狀為 `[1,1,4,2]`：一筆、一種讀法、四位置、每張兩格。

![每個查詢的可讀位置、讀取比例與實際混合結果](../figures/rewrite-03-causal-results.svg)

圖中每列四卡從左到右是位置 0、1、2、3，卡內是 V 的兩個特徵。第 0 查詢只讀第 0 卡，輸出 `[1,10]`；第 1 查詢平分前兩卡，得 `[1.5,15]`；第 2 得 `[2,20]`；最後可讀四卡，得 `[2.5,25]`。灰色未來卡的比例為 0，不參與該列混合。

工具在 softmax 前將禁止位置的分數設 `-inf`，即負無限；它的指數權重為 0，所以最後比例也為 0。若只是把分數設成普通 0，softmax 仍會給它機會，不能排除。

程式把第 3 卡加 100，再算一次。前面三個輸出不變，最後可以改變；斷言查的是這個實際影響，而不只是圖形是否三角形。

練習移除 `tril()`，讓每格都可讀。四張卡會平分，改最後卡會影響所有輸出，斷言應失敗。答案配對決定「猜什麼」，因果許可決定「能用什麼猜」；兩者都要正確。

<details>
<summary>補充與重做</summary>

前面的 shift 見 [1.3](01.md#1.3)。`manual_attention` 同時回傳混合結果與權重表，`clone()` 避免改動原 V。這段只測讀取範圍，沒有更新模型；低答案代價不能代替防偷看檢查。

</details>

