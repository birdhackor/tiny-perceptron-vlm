## 16.8 手寫 attention 怎麼換成最佳化介面？

手寫注意力讓每一步容易檢查，但矩陣分數、遮罩、softmax與加權取回分開執行，也可能多次配置與搬移中間資料。PyTorch提供把這段運算合在一起的介面，稱scaled dot-product attention（縮放點積注意力，SDPA）。更換時第一件事不是計時，而是確認輸入軸、縮放和遮罩語意相同；否則較快的結果可能已經算了另一個問題。

材料固定成一頭、四個位置、每個位置三項特徵，兩條路都要算相同的[匹配與取回](03.md#3.4)。Query表示目前要找什麼，Key表示各位置可以怎麼匹配，Value是匹配後要取回的內容。每個頭的Query、Key有D項特徵，原分數是 `QKᵀ/√D`，softmax後乘Value。本例一段輸入、一頭、四位置、每位置三項特徵，所以Q/K/V形狀都是 `[1,1,4,3]`，D=3。布林遮罩True在這個SDPA介面表示「允許讀」，不是「遮掉」。

程式的`@`取左邊矩陣的一列、右邊的一欄，對應項相乘再相加。Q與轉置後的K相乘，為每個Query算出四個候選Key分數，所以scores形狀是 `[1,1,4,4]`；`softmax(-1)`沿最後的四個候選分配比例，weights形狀相同。weights再乘V，把四份三項特徵按比例混合，得到每個Query的三項輸出，形狀回到 `[1,1,4,3]`。

圖的列是Query，欄是Key位置，對角線包含自己，左下角是過去，右上角是未來。綠色允許、灰色禁止；下方的最後位置可以讀四格，而第一位置只能讀自己。下面程式的 `tril`正是建立這個下三角規則。

![因果注意力只允許讀取自己與較早位置](../figures/rewrite-16-causal-mask.svg)

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
q, k, v = [torch.randn(1, 1, 4, 3) for _ in range(3)]
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
scores = q @ k.transpose(-2, -1) / (3**0.5)
weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
manual = weights @ v
optimized = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
print("輸出形狀", tuple(optimized.shape))
print("最大差異", (manual - optimized).abs().max().item())
assert torch.allclose(manual, optimized, atol=1e-6)
```

`transpose(-2,-1)`只交換Key的最後兩軸，保留batch和head；`[None,None]`替四乘四遮罩補上前兩軸，以便同一規則套到注意力分數。手寫路徑先縮放一次，把禁止格填成負無窮大，它們softmax後權重為零。SDPA內部已做相同縮放，傳入q時不要再先除√3，否則會縮放兩次。`dropout_p=0.0`停用隨機丟棄權重，讓兩路比較同一數學問題。[模型的eval評估模式](05.md#5.17)會停用模型層中的某些訓練行為，但這個SDPA函式仍照傳入的`dropout_p`執行；評估時也必須明確傳0.0，不能期待它隨模型模式自動清零。

輸出形狀應為 `(1,1,4,3)`，最大差異通常是零或約10⁻⁷的浮點尾數，assert確認接近。SDPA可能依裝置、精度和形狀選擇不同底層算法；[FlashAttention](16.md#16.9)是一種減少注意力中間資料讀寫的GPU實作，使用SDPA介面不保證選到它，更不保證在CPU小尺寸下加速。模型訓練還要核對梯度，這個例子只隔離向前數值與資料契約。

練習只將SDPA的 `attn_mask=allowed` 改成 `attn_mask=~allowed`，保留手寫路徑。先預測兩路不再相同，再執行觀察assert失敗；這不是最佳化誤差，而是可見位置完全改變。練習結束改回原遮罩，再把`torch.manual_seed(0)`的0改成1，從頭執行以產生另一組可重現的Q/K/V。先預測輸出形狀不變、兩路仍應接近且assert通過，再核對，確認對齊不是只發生在原來那組輸入。

<details>
<summary>選讀：來源、量測條件與原始實報</summary>

[PyTorch 2.14 SDPA文件](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)明列上述True與`dropout_p`語意，也提醒不同後端的浮點結果可能不同。先說明實測用的數字格式：dtype指儲存數字的型別；FP32是32位元浮點數，FP16與BF16是兩種16位元浮點格式。較少位元與不同格式可能產生不同捨入，因此比較容差需按格式事先設定，詳見[精度小例子](16.md#16.7)。

實際整模型核對用NVIDIA L4圖形處理器、四個Query頭各自保留一組Key與Value（四KV頭）、四題補齊為53位置，數字格式是FP32。logit是模型對每個候選token尚未轉成機率的原始分數；整模型logit最高差3.81470×10⁻⁶，權重梯度最高差1.60071×10⁻¹⁰。梯度表示各個可調數字對代價的敏感度，見[梯度暖身](../first-steps.md#W.6)。以向前絕對容差10⁻⁵、梯度10⁻⁸閱讀這次核對，兩者都在範圍內；不能套用單個CPU例子的10⁻⁶容差去要求所有大型輸出逐bit相同。

運算追蹤工具profiler實際記到`aten::_scaled_dot_product_efficient_attention`與`aten::_efficient_attention_forward`，所以本次是節省中間資料記憶體的memory-efficient實作，沒有觀察到CUDA Flash；CUDA是NVIDIA GPU的運算環境。暖機後向前中位數手寫2.990毫秒、SDPA2.346毫秒，量的是完整TinyLM、不是只有attention核心。兩個副本再各更新40次、2,328個有效目標，更新後權重最高差3.51369×10⁻⁵；手寫與SDPA每步14.070、12.849毫秒。這些結果支持此次形狀的接近與較低耗時，不能以API名字替其他dtype或GQA輸入指定後端。GQA讓多個Query頭共用較少組Key與Value，兩邊頭數因而可能不同；共享方式與形狀見[16.4](16.md#16.4)。

我們另做一份只核對注意力核心的[CUDA Flash補驗報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/flash_probe.json)。Q/K/V都固定為`[2,4,512,32]`：兩段輸入、每段四頭、512個位置、每頭32項特徵；把控制隨機數起點的seed固定為42建立，再分別轉成FP16與BF16。這次明確只開`SDPBackend.FLASH_ATTENTION`，不提供額外遮罩，以`is_causal=True`限制未來、`dropout_p=0.0`停用隨機丟棄。L4上的profiler在兩種精度都記到CUDA Flash前向與反傳運算子，以及真正的GPU運算程式（CUDA kernel），才確認使用了Flash。

先核對再計時。手寫對照的QK與PV乘法使用同一低精度，P指softmax後的權重；分數縮放、遮罩與softmax用FP32，它不是所有運算都用FP32的真值。反傳時需要知道後續代價對每格輸出的敏感度，這份由輸出端送回的數值表叫上游梯度。本次將同一份表送給兩路，再比較它們算出的Q/K/V敏感度，才是在相同輸入條件下核對反傳。兩路的輸出與Q/K/V梯度差如下：

| 精度 | 輸出最大絕對差 | Q/K/V梯度中的最大絕對差 |
| --- | ---: | ---: |
| FP16 | 0.001953125 | 0.000244140625 |
| BF16 | 0.015625 | 0.001953125 |

全部數值有限，且在執行前固定的容差內：FP16輸出`atol=rtol=0.01`、梯度0.03；BF16輸出`atol=0.06, rtol=0.04`、梯度均0.08。`atol`是接近零時仍允許的絕對差，`rtol`是隨對照數值大小調整的相對差；兩者共同判斷接近，並沒有要求逐bit相同。完整報告另列三份梯度各自的誤差，時間與記憶體在下一節閱讀。這份補驗沒有訓練模型，也沒有替上面的整模型更新核對增加一組訓練結果。

</details>

