## 5.17 eval 就會停止計算梯度嗎？

Linear將[1,1]變成y=w1×1+w2×1+b，w1、w2是權重，b是偏置。切到eval後，還能對w1、w2、b求梯度嗎？能。評估模式控制前向行為，不負責關閉求導。

eval讓dropout這類訓練時隨機省略數值的操作改用評估行為；no_grad則讓此次計算不留下求導關係。先用沒有dropout的Linear看兩個開關。

```python
import torch
from torch import nn

layer = nn.Linear(2, 1)
x = torch.ones(1, 2)
layer.eval()
output = layer(x)
print("eval輸出可求導", output.requires_grad, output.grad_fn)
output.sum().backward()
with torch.no_grad():
    other = layer(x)
print("no_grad輸出可求導", other.requires_grad, other.grad_fn)
assert output.requires_grad and not other.requires_grad
```

輸入一筆兩個1，Linear參數預設可求導。`eval`後output仍連着權重計算，所以打印True與一個 `grad_fn`；`grad_fn` 是記錄這個結果由哪種運算來的內部標記，不需背它的具體名稱。`backward()` 能把影響傳回權重。`no_grad`內的other則打印False與None，表示沒有這一條求導路線。

Linear本身沒有dropout，因此eval不會讓它數值變動；我們選它正是為了專注證明「評估行為」與「能否求導」獨立。換成含dropout的網絡時eval會影響輸出方式，仍不表示它禁用梯度。相反，訓練模式下使用no_grad也會不記錄那次運算。

第三個設定requires_grad=False只凍結相應參數。固定權重仍是可微配方，輸入若需要梯度，輸出仍能對它求導。固定參考常同時eval、凍結和no_grad，各有目的；若還需穿過它訓練前面的輸入產生器，就不能把該段包成no_grad。

練習改為凍結參數、對輸入求導。另開一段獨立程式，以下完整取代原示範；原來的 `other`、`with torch.no_grad()` 比較區塊與末尾斷言都不保留。先預測：輸出仍可求導，`backward()` 後 x.grad 的兩格會分別等於固定的 w1、w2，因為每個輸入微增時，y 就按對應係數改變。

```python
import torch
from torch import nn

layer = nn.Linear(2, 1)
for parameter in layer.parameters():
    parameter.requires_grad_(False)
layer.eval()
x = torch.ones(1, 2, requires_grad=True)
output = layer(x)
print("輸出可求導", output.requires_grad)
output.sum().backward()
print("固定權重w1與w2", layer.weight)
print("輸入梯度", x.grad)
assert torch.allclose(x.grad, layer.weight)
```

輸入仍是一筆兩個1，但這次兩格需要梯度。迴圈逐項凍結 Linear 的權重與偏置，保留 eval，並讓前向計算留下求導關係。`layer.weight` 第一排的兩個數就是 w1、w2；它們在本次計算中固定。第一行應印 True，後兩行的兩個數應逐格相同，最後的斷言核對這件事。實際初始化值可不同，核對的是輸入敏感度是否等於同一次程式的固定係數。你凍結的是哪一組旋鈕，不等於整段計算停止可微。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>
