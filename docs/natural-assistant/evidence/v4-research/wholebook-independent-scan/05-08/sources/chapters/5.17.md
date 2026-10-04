## 5.17 eval 就會停止計算梯度嗎？

同事說「把模型切成評估模式，就不會訓練了」，這句話混了三件事：模組的行為、運算是否被記錄、某個參數是否可求導。評估時通常三者需要不同設置，不能只看一個開關就推斷參數有沒有梯度。本節用一層Linear，把三條規則分開驗證。`Linear(2,1)`把每筆兩個輸入數變成一個輸出：`y = w1*x1 + w2*x2 + b`。w1、w2是權重，b是混合完再加上的可調數字，叫bias或偏置；它們都是層的參數，`layer.parameters()`會包含權重與bias。背景可回看[2.3](02.md#2.3)。

`model.eval()` 改模塊的訓練/評估行為，例如dropout在訓練時隨機省略部分數值，評估時停用這種隨機省略。它不關閉autograd。`torch.no_grad()` 讓這次區塊內的運算不建立求導關係。參數的 `requires_grad=False` 則只說「不要對這個參數求導」，還可能保留輸入或其他參數的梯度。前向與反向概念見 [1.10](01.md#1.10)。

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

若只凍結權重，卻讓x需要梯度，輸出仍可對x求導，因為固定權重仍是一套可微的配方。這在固定一個模組、訓練前面的輸入產生器時很有用。`.requires_grad_(False)` 的結尾底線表示直接更改原參數設定；凍結權重的同時，偏移量也要考慮，否則尚可通過bias求導。

使用一個固定參考模型提供目標時，常同時eval、凍結參數、在無需回傳到輸入的預測中no_grad：各自負責穩定模塊行為、避免參數被訓練、減少此次求導記錄。若任務還要穿過固定模組回傳到輸入，就不能把整段包在no_grad裡。這要按我們需要哪條影響路線決定。

練習先對 `layer.parameters()` 逐項執行 `parameter.requires_grad_(False)`，再把x建立時改成 `torch.ones(1,2, requires_grad=True)`。保留eval但不加no_grad，先預測輸出仍True、backward後x.grad有兩個數，分別等於固定的w1與w2：輸入各增加一點時，y分別按這兩個係數變化。再執行核對。你凍結的是哪一組旋鈕，不等於整段計算停止可微。
