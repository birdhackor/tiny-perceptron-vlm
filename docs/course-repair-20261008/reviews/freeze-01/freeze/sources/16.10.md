## 16.10 怎麼用重算換記憶體？

訓練時，向前計算不只是產生答案，也會留下中間特徵，讓向後計算知道每個權重怎麼影響誤差。如果層很深或輸入很長，這些特徵可能比權重本身更占記憶體。可以少存一些，等向後計算需要時，再從較早的輸入重算嗎？這叫activation checkpointing（中間特徵檢查點）：拿額外計算換較少保留的特徵，不是把模型存成權重檔的checkpoint。

我們沿用[MLP的特徵處理](14.md#14.4)，只改中間結果何時保存、何時重算。以「四項特徵擴成八項、套GELU、再收回四項」為例，普通訓練會保存導數計算需要的中間結果；checkpoint版本把這段當作可重算區域，保留必要邊界資料，向後時再執行區域中的運算。權重沒有刪掉，誤差目標也應保持相同。

```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
layer = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
base = torch.randn(2, 4)
x_plain = base.clone().requires_grad_()
x_recompute = base.clone().requires_grad_()
y_plain = layer(x_plain)
y_recompute = checkpoint(layer, x_recompute, use_reentrant=False)
y_plain.square().sum().backward()
y_recompute.square().sum().backward()
print("輸出形狀", tuple(y_recompute.shape))
print("輸出最大差", (y_plain - y_recompute).abs().max().item())
print("輸入梯度最大差", (x_plain.grad - x_recompute.grad).abs().max().item())
assert torch.allclose(x_plain.grad, x_recompute.grad, atol=1e-6)
```

輸入兩列、每列四項特徵，兩條路共用同一個layer和相同數值的輸入，但x各自獨立，以便比較梯度。`requires_grad_`使輸入也記錄梯度；`checkpoint`包住整段layer，`use_reentrant=False`選用會記錄前向求導關係、在反向時按需重算中間結果的方式，框架自己管理這些重算，不要求你另寫求導程序。兩個誤差都是輸出平方和，沒有更新權重，所以後算的那條路仍使用同一份權重。

輸出形狀應是 `(2,4)`，輸出最大差與輸入梯度最大差應為零或小於容忍範圍的尾數。共享layer的權重梯度會被兩次backward累加，這是本例刻意不拿來直接比較的部分；真正核對權重梯度時，應先清除前一條路的累積或用獨立副本。這裡只確認重算路徑能保持向前結果和輸入導數，沒有量到記憶體節省比例。

重算不是免費收益。選很小區域可能省不了多少，選很大區域則可能重做更多計算；要量實際訓練的最高記憶體與每步時間。dropout是在訓練時隨機把部分特徵設為零的運算，這次保留哪些格、清零哪些格的記錄稱為隨機遮罩。若重算區域含這類運算，就要重現原來的隨機狀態，使用同一個遮罩，才能求出當初那次前向計算的導數；框架通常會管理這件事，但自行改變裝置或在函數內修改外部狀態仍需特別核對。

一輪兩層TinyLM的L4對照，把完整模型層包成重算區域：普通／重算每次更新14.070／20.735毫秒，配置器峰值增量8.824／5.041 MiB；兩支各更新40次、2,328個助手回答及EOS目標，最後權重最大差為0。這是本次以更多時間換較少同時保留資料的例子。兩支原本駐留量不同，這些增量不能當作模型部署大小或固定節省比例。

練習只把中間寬度8改成16，兩個Linear的尺寸都要一起改。先預測輸出仍 `(2,4)`，兩路差異仍接近零，再重跑。它增加可重算的中間特徵，卻不改對齊要求；此小CPU例子不適合替大型GPU模型宣稱節省了多少記憶體。

<details>
<summary>選讀：來源、量測條件與原始實報</summary>

L4實報則用已訓練的兩層TinyLM。這裡的一個Transformer block是[4.5的完整模型層塊](04.md#4.5)：先整理特徵尺度、用注意力讀其他位置並加回原輸入，再整理尺度、處理各位置特徵並加回。實報把每個完整block包在`checkpoint(..., use_reentrant=False)`，所以重算區域比上面單獨的MLP大；輸入查表與最終候選分數的計算在這些區域之外。其重算與前後函式須一致的條件可對照[PyTorch 2.14.1原始程式說明](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/utils/checkpoint.py#L435)。

四題的整模型logit（每個候選token尚未轉成機率的原始分數）與權重梯度最大差都為0。普通與重算副本再各更新40次，共計2,328個有效目標；這是納入答案代價的token位置數，包含助手回答與結束符號，排除user提問、角色標記與PAD補齊位置，見[7.3的答案位置](07.md#7.3)。兩路不只更新次數相同，納入代價的答案工作量也相同。最後權重最大差仍0，驗證3/5、最後檢查5/10一致，分數是完整答案答對題數／總題數。這才是此模型、本輪訓練的數值與品質核對，不能由CPU小層替代。

普通每步14.070毫秒，重算20.735毫秒；重算支線開始前已配置69.336 MiB、訓練峰值74.377 MiB，新增5.041 MiB，普通為67.164、75.988、新增8.824 MiB。按[16.1](16.md#16.1)的配置器範圍，這輪以時間換到較小的峰值增量，沒有刪除權重或減少更新數。兩者駐留起點不同，不能直接用峰值比宣稱大型模型可省固定百分比；也沒有dropout或跨裝置函式的核驗。

</details>

