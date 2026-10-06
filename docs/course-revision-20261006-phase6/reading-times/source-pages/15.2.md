## 15.2 能否準備多組 FFN？

同一個輸入 `[1,2]`，我們想準備「保留、兩倍、交換特徵」三種獨立規則。在[Dense的規則](15.md#15.1)裡，所有token都使用同一組FFN；現在先建立三組各有自己權重的FFN，下一節再決定誰處理哪個輸入。每一組稱為expert（專家）。這個名字只是結構上的稱呼：剛建立時，它們只有初始化的數字，不會天生分成「數學專家」和「中文專家」。專長必須從訓練後的資料與行為觀察。

先用最簡單的線性層代表每位expert。輸入 `[1,2]`，第一位使用單位矩陣，輸出仍是 `[1,2]`；第二位把每個數字乘2，輸出 `[2,4]`；第三位交換兩個特徵，輸出 `[2,1]`。三者接收相同形狀，產生相同形狀，裡面的數字卻各自儲存。真實expert通常採用前節的兩層FFN；這裡縮成一層，只把「獨立權重」這件事隔離出來。

```python
import torch
from torch import nn

experts = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
tables = [torch.eye(2), 2 * torch.eye(2), torch.tensor([[0.0, 1.0], [1.0, 0.0]])]
with torch.no_grad():
    for expert, table in zip(experts, tables):
        expert.weight.copy_(table)
x = torch.tensor([[1.0, 2.0]])
print("各自輸出", [e(x).tolist() for e in experts])
print("同一權重", experts[0].weight is experts[1].weight)
print("權重總數", sum(p.numel() for p in experts.parameters()))
shared = nn.Linear(2, 2, bias=False)
repeated = nn.ModuleList([shared, shared, shared])
print("重複引用總數", sum(p.numel() for p in repeated.parameters()))
```

`ModuleList`是讓PyTorch登記一串子模組的容器；它本身不替你選expert，也不替你執行輸入。前面的列表推導式每次呼叫 `nn.Linear`，所以真的建立三個模組。`zip(experts,tables)`把兩串內容按位置配對：第一次取第0位expert和第0張table，接著是第1對、第2對；迴圈因此依序填好三張權重表。`copy_`只是在不記錄梯度時填入已知表格，方便核對，並沒有把三個權重綁成一個物件。第一行應是 `[[[1,2]],[[2,4]],[[2,1]]]`，第二行為False。

`experts.parameters()`依序提供這個容器裡登記的可學參數。`sum(p.numel() for p in experts.parameters())`可以逐步讀成：每次取一張參數表叫p，用 `p.numel()`數這張表的元素，再由 `sum`把每次得到的數字加起來。本例就是先得到4、4、4，再加成12；最後一行對repeated做同樣的計數。

第三行為12，因為三張二乘二表各有四個權重。最後一行卻為4：把同一個shared放進列表三次，PyTorch列出可學參數時會辨認這些引用屬於同一份數字。三個名稱不會創造三份獨立容量。這也呼應[權重共享](14.md#14.6)的差別：相同數值、相同形狀與同一物件並不是同一件事。

目前程式把三位都算了一次，只為檢查各自的規則，還沒有省計算。後面需要一個根據輸入做選擇的router，把token送往少數expert；如果始終讓三位都算，再把結果平均，參數增加，運算也增加。

練習在輸出前加一個 `no_grad` 區塊，將 `experts[0].weight` 乘3。先預測只有第一個輸出變成 `[3,6]`，再執行核對。這個手動修改只用到上面已有的工具，可以直接證明三份權重會獨立改變。

<details>
<summary>選讀：來源、量測條件與原始實報</summary>

正式MoE比較在兩層各放四份獨立FFN，共八份、264,704個expert參數；加上router與共享部分，總數340,608，詳見[15.10](15.md#15.10)。這些數量能證明多存了規則，不能證明八份已各有專長。我們的短訓報告只有路由選擇與文字評估，沒有按語言或學科核驗專業分工，因此不把expert命名為「英文」或「數學」專家。

</details>

